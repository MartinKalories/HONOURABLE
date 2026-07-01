"""
Phase-only least-squares fit:
    target wavefront phase image
        ≈ phase of a complex linear combination of Fourier-transformed LP modes

This is similar in structure to WAHLAH.py, but instead of fitting PSF intensity,
it fits phase only using the far-field LP modes.

Put this file in the same folder as lanternfiber.py, or make sure lanternfiber.py
is on your Python path.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.ndimage import zoom

from lanternfiber import lanternfiber

from datetime import datetime
# -------------------------------------------------------------------------
# Default settings
# -------------------------------------------------------------------------

DATADIR = "/home/manav//PL-NN-testdata_forDec2025/"
OUTDIR = DATADIR

# The user-mentioned wavefront cube file.
# If this exact file is not found, the loader will also try a few common variants.
WAVEFRONT_NPZ_FILENAME = "slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"

# Set this to a key name if you know it, for example "all_wfims".
# If None, the script prints all keys and automatically chooses a likely wavefront array.
WAVEFRONT_KEY = None

# Fibre parameters, copied from WAHLAH.py.
N_CORE = 1.44
N_CLADDING = 1.4345
WAVELENGTH = 1.55
CORE_RADIUS = 32.8 / 2

# Mode / fitting settings.
N_MODES = 10              # toggle this to choose how many far-field LP modes are used
N_TEST = 5                # how many wavefront images to fit
NPIX = 200                # near-field half-size; final near-field image is 2*NPIX by 2*NPIX
MAX_R = 2
PAD_FACTOR = 1            # far-field zero padding. 2 gives smoother far-field modes.
MAX_NFEV = 1000
N_RESTARTS = 3
RNG_SEED = 42

# Target phase units.
# Use "radians" if your wavefront images are already phase in radians.
# Use "waves" if your wavefront images are optical path in waves, so 1 means 2*pi phase.
# Use "degrees" if the wavefront phase is in degrees.
TARGET_PHASE_UNITS = "radians"

# Optional phase preprocessing.
REMOVE_TARGET_PISTON = False   # if True, subtracts circular mean phase from each target

# Optional centre crop after resizing modes to the target grid.
# None = fit the whole target image.
# 80   = fit a 160x160 square around the centre.
CENTRE_CROP_PIXELS = None

# Plotting.
PLOT_CROP_PIXELS = 80         # centre crop shown in saved example plot. None shows full image.

# Numerical stability.
FIELD_EPS = 1e-12


# -------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------

def wrap_phase(phi):
    """Wrap phase to [-pi, pi)."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def circular_mean_phase(phi):
    """Circular mean of a phase image."""
    return np.angle(np.mean(np.exp(1j * phi)))


def centre_crop(arr, crop_pixels):
    """
    Crop a square around the centre.

    crop_pixels is the half-width:
        crop_pixels=80 gives a 160 x 160 crop.
    """
    if crop_pixels is None:
        return arr

    cy, cx = np.array(arr.shape[-2:]) // 2

    y0 = max(cy - crop_pixels, 0)
    y1 = min(cy + crop_pixels, arr.shape[-2])
    x0 = max(cx - crop_pixels, 0)
    x1 = min(cx + crop_pixels, arr.shape[-1])

    return arr[..., y0:y1, x0:x1]


def resize_complex_image_to_shape(z, target_shape):
    """
    Resize a complex image by interpolating real and imaginary parts separately.
    """
    z = np.asarray(z, dtype=np.complex128)

    if z.shape == tuple(target_shape):
        return z

    zoom_y = target_shape[0] / z.shape[0]
    zoom_x = target_shape[1] / z.shape[1]

    real_resized = zoom(z.real, (zoom_y, zoom_x), order=1)
    imag_resized = zoom(z.imag, (zoom_y, zoom_x), order=1)

    return real_resized + 1j * imag_resized


def resize_phase_image_to_shape(phase, target_shape):
    """
    Resize a phase image safely by interpolating exp(i*phase), not phase directly.
    This avoids interpolation problems at the -pi/pi wrapping boundary.
    """
    phase = np.asarray(phase, dtype=float)

    if phase.shape == tuple(target_shape):
        return phase

    z = np.exp(1j * phase)
    z_resized = resize_complex_image_to_shape(z, target_shape)

    return np.angle(z_resized)


def normalise_power(field):
    """Normalise complex field so sum(|field|^2) = 1."""
    power = np.sum(np.abs(field) ** 2)
    if power > 0:
        return field / np.sqrt(power)
    return field


# -------------------------------------------------------------------------
# Load target wavefronts
# -------------------------------------------------------------------------

def possible_wavefront_paths(datadir, filename):
    """
    Try the user-given filename plus a few likely variants from the PL data naming.
    """
    base = filename
    if base.endswith(".npz"):
        stem = base[:-4]
    else:
        stem = base

    candidates = [
        os.path.join(datadir, base),
        os.path.join(datadir, stem + ".npz"),
        os.path.join(datadir, stem + "-WFs.npz"),
        os.path.join(datadir, stem + "-WF.npz"),
        os.path.join(datadir, "pllabdata_20240605_singlepsf_01_" + stem + ".npz"),
        os.path.join(datadir, "pllabdata_20240605_singlepsf_01_" + stem + "-WFs.npz"),
        os.path.join(datadir, "pllabdata_20240605_singlepsf_01_" + stem + "-WF.npz"),
    ]

    # Remove duplicates while preserving order.
    out = []
    for c in candidates:
        if c not in out:
            out.append(c)

    return out


def choose_wavefront_key(npz_file, requested_key=None):
    """
    Pick a wavefront-like array from an npz file.
    Prints the available keys to make debugging easier.
    """
    keys = list(npz_file.keys())

    print("\nAvailable keys in wavefront npz:")
    for key in keys:
        arr = npz_file[key]
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        print(f"  {key}: shape={shape}, dtype={dtype}")

    if requested_key is not None:
        if requested_key not in npz_file:
            raise KeyError(
                f"WAVEFRONT_KEY='{requested_key}' was not found. "
                f"Available keys are: {keys}"
            )
        return requested_key

    # Prefer names that look like wavefront arrays.
    priority_words = ["wf", "wavefront", "phase", "phi", "slm"]

    for word in priority_words:
        for key in keys:
            arr = npz_file[key]
            if word.lower() in key.lower() and hasattr(arr, "ndim") and arr.ndim >= 2:
                return key

    # Fallback: first numeric array with image-like dimensions.
    for key in keys:
        arr = npz_file[key]
        if hasattr(arr, "ndim") and arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
            return key

    raise RuntimeError("Could not find a suitable wavefront array in the npz file.")


def standardise_wavefront_array_shape(arr):
    """
    Convert wavefront data to shape:
        (n_images, ny, nx)

    Handles common cases:
        (ny, nx)
        (n_images, ny, nx)
        (ny, nx, n_images)
        (n_images, ny, nx, 1)
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[None, :, :]

    elif arr.ndim == 3:
        # If shape is probably (ny, nx, n_images), move the image axis to the front.
        # Example: (128, 128, 10000)
        if arr.shape[0] == arr.shape[1] and arr.shape[2] != arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)

    elif arr.ndim == 4:
        # Try to remove a single channel axis.
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        else:
            raise ValueError(f"Do not know how to handle 4D wavefront array shape {arr.shape}")

    else:
        raise ValueError(f"Do not know how to handle wavefront array shape {arr.shape}")

    if arr.ndim != 3:
        raise ValueError(f"Expected wavefront array to become 3D, got shape {arr.shape}")

    return arr.astype(float)


def load_wavefronts(datadir, filename, key=None):
    """
    Load wavefront phase images from the target npz file.
    """
    paths = possible_wavefront_paths(datadir, filename)

    wavefront_path = None
    for path in paths:
        if os.path.exists(path):
            wavefront_path = path
            break

    if wavefront_path is None:
        print("\nTried these wavefront paths:")
        for path in paths:
            print(" ", path)
        raise FileNotFoundError(
            "Could not find the wavefront npz file. "
            "Check WAVEFRONT_NPZ_FILENAME and DATADIR."
        )

    print("Loading wavefronts from:", wavefront_path)

    npz_file = np.load(wavefront_path, allow_pickle=True)
    chosen_key = choose_wavefront_key(npz_file, requested_key=key)

    print("Using wavefront key:", chosen_key)

    wavefronts = standardise_wavefront_array_shape(npz_file[chosen_key])

    print("Wavefront array shape after standardising:", wavefronts.shape)

    return wavefronts, wavefront_path, chosen_key


def prepare_target_phase(target_raw):
    """
    Convert one target wavefront image to wrapped phase in radians.
    """
    target = np.squeeze(target_raw).astype(float)

    if TARGET_PHASE_UNITS.lower() == "waves":
        target = target * 2 * np.pi
    elif TARGET_PHASE_UNITS.lower() == "degrees":
        target = np.deg2rad(target)
    elif TARGET_PHASE_UNITS.lower() == "radians":
        pass
    else:
        raise ValueError(
            "TARGET_PHASE_UNITS must be 'radians', 'waves', or 'degrees'."
        )

    target = wrap_phase(target)

    if REMOVE_TARGET_PISTON:
        target = wrap_phase(target - circular_mean_phase(target))

    return target


# -------------------------------------------------------------------------
# Far-field LP mode generation
# -------------------------------------------------------------------------

def near_to_far_field(near_field, pad_factor=1, normtosum=True):
    """
    Fourier transform a centred near-field mode into a centred far-field mode.

    Shift order:
        near-field centre -> array origin using ifftshift()
        fft2()
        far-field zero spatial frequency -> centre using fftshift()
    """
    near_field = np.asarray(near_field, dtype=np.complex128)

    if pad_factor > 1:
        ny, nx = near_field.shape

        new_ny = ny * pad_factor
        new_nx = nx * pad_factor

        pad_before_y = (new_ny - ny) // 2
        pad_after_y = new_ny - ny - pad_before_y

        pad_before_x = (new_nx - nx) // 2
        pad_after_x = new_nx - nx - pad_before_x

        near_field = np.pad(
            near_field,
            ((pad_before_y, pad_after_y), (pad_before_x, pad_after_x)),
            mode="constant",
            constant_values=0,
        )

    far_field = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(near_field)
        )
    )

    if normtosum:
        far_field = normalise_power(far_field)

    return far_field


def make_farfield_lp_modes(target_shape, n_modes, centre_crop_pixels=None):
    """
    Make Fourier-transformed LP modes and return the mode matrix.

    The mode matrix has shape:
        n_pixels x n_modes

    Each column is a flattened complex far-field LP mode.
    """
    lf = lanternfiber(N_CORE, N_CLADDING, CORE_RADIUS, WAVELENGTH)
    lf.find_fiber_modes()
    lf.make_fiber_modes(
        npix=NPIX,
        max_r=MAX_R,
        show_plots=False,
        normtosum=True,
    )

    near_fields_raw = np.array(lf.allmodefields_rsoftorder)
    n_available = near_fields_raw.shape[0]

    print("\nTotal available LP scalar modes:", n_available)

    if n_modes > n_available:
        raise ValueError(f"n_modes={n_modes} but only {n_available} LP modes are available.")

    far_modes = []
    labels = []
    lm_values = []

    for mode_num in range(n_modes):
        raw_mode = near_fields_raw[mode_num]

        # raw_mode is a signed real amplitude: negative values mean pi phase.
        near_field = lf.make_complex_fld(raw_mode)

        far_field = near_to_far_field(
            near_field,
            pad_factor=PAD_FACTOR,
            normtosum=True,
        )

        # Resize complex far-field mode to target wavefront grid.
        far_field = resize_complex_image_to_shape(far_field, target_shape)

        # Optional centre crop.
        far_field = centre_crop(far_field, centre_crop_pixels)

        # Normalise each mode after resizing/cropping.
        far_field = normalise_power(far_field)

        far_modes.append(far_field)

        if mode_num < len(lf.modelabels):
            labels.append(lf.modelabels[mode_num])
        else:
            labels.append(f"mode_{mode_num}")

        if hasattr(lf, "lp_mode_list") and mode_num < len(lf.lp_mode_list):
            lm_values.append(lf.lp_mode_list[mode_num])
        else:
            lm_values.append([np.nan, np.nan])

    far_modes = np.asarray(far_modes, dtype=np.complex128)

    n_modes, ny, nx = far_modes.shape
    mode_matrix = far_modes.reshape(n_modes, ny * nx).T

    # Normalise matrix columns.
    norms = np.sqrt(np.sum(np.abs(mode_matrix) ** 2, axis=0, keepdims=True))
    mode_matrix = mode_matrix / (norms + FIELD_EPS)

    print("Far-field LP mode image shape used for fit:", (ny, nx))
    print("Mode matrix shape:", mode_matrix.shape)

    return mode_matrix, ny, nx, labels, np.asarray(lm_values), far_modes


# -------------------------------------------------------------------------
# Phase-only least-squares fitting
# -------------------------------------------------------------------------

def unpack_coeffs(z, n_modes):
    real = z[:n_modes]
    imag = z[n_modes:]
    coeffs = real + 1j * imag

    # Phase-only fitting is invariant to overall amplitude, so normalise coeffs
    # to prevent the optimiser from wandering in amplitude scale.
    norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))
    if norm > 0:
        coeffs = coeffs / norm

    return coeffs


def fit_coeffs_to_target_phase(
    mode_matrix,
    target_phase,
    max_nfev=1000,
    n_restarts=3,
    rng=None,
):
    """
    Fit complex LP coefficients so that:
        angle(mode_matrix @ coeffs) matches target_phase

    The residual compares unit phasors, not raw phase values:
        exp(i * fit_phase) - exp(i * target_phase)

    This avoids problems at the -pi/pi phase wrapping boundary.
    """
    if rng is None:
        rng = np.random.default_rng()

    M = mode_matrix.astype(np.complex128)
    n_modes = M.shape[1]

    target_phase = wrap_phase(target_phase)
    target_unit = np.exp(1j * target_phase).reshape(-1)

    def residual(z):
        coeffs = unpack_coeffs(z, n_modes)

        field_flat = M @ coeffs
        fit_unit = field_flat / (np.abs(field_flat) + FIELD_EPS)

        diff = fit_unit - target_unit

        # least_squares expects a real-valued residual vector.
        return np.concatenate([diff.real, diff.imag])

    best_res = None

    for restart in range(n_restarts):
        if restart == 0:
            z0 = np.zeros(2 * n_modes)
            z0[0] = 1.0
        else:
            z0 = rng.normal(0.0, 1.0, size=2 * n_modes)

        res = least_squares(
            residual,
            z0,
            max_nfev=max_nfev,
            verbose=0,
        )

        if best_res is None or res.cost < best_res.cost:
            best_res = res

    coeffs_fit = unpack_coeffs(best_res.x, n_modes)

    field_flat = M @ coeffs_fit
    field_fit = field_flat.reshape(target_phase.shape)
    phase_fit = np.angle(field_fit)

    phase_residual = wrap_phase(target_phase - phase_fit)

    rms_phase_error = np.sqrt(np.mean(phase_residual ** 2))
    mean_abs_phase_error = np.mean(np.abs(phase_residual))

    return coeffs_fit, field_fit, phase_fit, phase_residual, rms_phase_error, mean_abs_phase_error, best_res


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_phase_fit_example(
    target_phase,
    phase_fit,
    phase_residual,
    title,
    outpath,
    plot_crop_pixels=None,
):
    """
    Save a target / fit / residual phase comparison plot.
    """
    target_plot = centre_crop(target_phase, plot_crop_pixels)
    fit_plot = centre_crop(phase_fit, plot_crop_pixels)
    residual_plot = centre_crop(phase_residual, plot_crop_pixels)

    plt.figure(figsize=(13, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(target_plot, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.title("Target wavefront phase")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(fit_plot, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.title("LP far-field phase fit")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(residual_plot, cmap="bwr", vmin=-np.pi, vmax=np.pi)
    plt.title("Wrapped phase residual")
    plt.colorbar()

    plt.suptitle(title)
    plt.tight_layout()

    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_coeff_plot(coeffs, labels, lm_values, outpath):
    """
    Save a simple amplitude/phase plot of the fitted LP coefficients.
    """
    amps = np.abs(coeffs)
    phases = np.angle(coeffs)

    x = np.arange(len(coeffs))

    tick_labels = []
    for k, label in enumerate(labels):
        l_val, m_val = lm_values[k]
        tick_labels.append(f"{label}\n(l={int(l_val)},m={int(m_val)})")

    plt.figure(figsize=(max(10, len(coeffs) * 0.7), 6))

    plt.subplot(2, 1, 1)
    plt.bar(x, amps)
    plt.ylabel("|coefficient|")
    plt.xticks(x, tick_labels, rotation=90, fontsize=8)

    plt.subplot(2, 1, 2)
    plt.bar(x, phases)
    plt.ylabel("phase [rad]")
    plt.xticks(x, tick_labels, rotation=90, fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-modes", type=int, default=N_MODES)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--max-nfev", type=int, default=MAX_NFEV)
    parser.add_argument("--n-restarts", type=int, default=N_RESTARTS)
    parser.add_argument("--crop-pixels", type=int, default=-1,
                        help="Half-width of centre crop for fitting. -1 means no crop.")
    parser.add_argument("--plot-crop-pixels", type=int, default=PLOT_CROP_PIXELS,
                        help="Half-width of centre crop shown in plots. -1 means full image.")
    parser.add_argument("--wavefront-key", type=str, default=WAVEFRONT_KEY)
    parser.add_argument("--wavefront-file", type=str, default=WAVEFRONT_NPZ_FILENAME)
    args = parser.parse_args()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_modes = args.n_modes
    n_test = args.n_test
    fit_crop_pixels = None if args.crop_pixels < 0 else args.crop_pixels
    plot_crop_pixels = None if args.plot_crop_pixels < 0 else args.plot_crop_pixels

    rng = np.random.default_rng(RNG_SEED)

    os.makedirs(OUTDIR, exist_ok=True)

    wavefronts, wavefront_path, wavefront_key = load_wavefronts(
        DATADIR,
        args.wavefront_file,
        key=args.wavefront_key,
    )

    n_test = min(n_test, wavefronts.shape[0])

    # Prepare the first target to define the fit grid.
    first_target_phase = prepare_target_phase(wavefronts[0])
    target_shape = first_target_phase.shape

    print("\nTarget phase image shape:", target_shape)

    mode_matrix, ny, nx, labels, lm_values, far_modes = make_farfield_lp_modes(
        target_shape=target_shape,
        n_modes=n_modes,
        centre_crop_pixels=fit_crop_pixels,
    )

    rms_phase_errors = []
    mean_abs_phase_errors = []
    costs = []
    nfevs = []
    success_flags = []
    coeffs_all = []

    example_target = None
    example_fit = None
    example_residual = None
    example_coeffs = None

    for i in range(n_test):
        print(f"\nFitting wavefront {i + 1}/{n_test}")

        target_phase = prepare_target_phase(wavefronts[i])

        # Fit uses the same centre crop as the modes, if enabled.
        target_phase = centre_crop(target_phase, fit_crop_pixels)

        coeffs_fit, field_fit, phase_fit, phase_residual, rms_err, mae_err, res = (
            fit_coeffs_to_target_phase(
                mode_matrix=mode_matrix,
                target_phase=target_phase,
                max_nfev=args.max_nfev,
                n_restarts=args.n_restarts,
                rng=rng,
            )
        )

        rms_phase_errors.append(rms_err)
        mean_abs_phase_errors.append(mae_err)
        costs.append(res.cost)
        nfevs.append(res.nfev)
        success_flags.append(res.success)
        coeffs_all.append(coeffs_fit)

        print("RMS wrapped phase error [rad]:", rms_err)
        print("Mean absolute wrapped phase error [rad]:", mae_err)
        print("Cost:", res.cost)
        print("nfev:", res.nfev)
        print("Success:", res.success)

        if i == 0:
            example_target = target_phase
            example_fit = phase_fit
            example_residual = phase_residual
            example_coeffs = coeffs_fit

    rms_phase_errors = np.asarray(rms_phase_errors)
    mean_abs_phase_errors = np.asarray(mean_abs_phase_errors)
    costs = np.asarray(costs)
    nfevs = np.asarray(nfevs)
    success_flags = np.asarray(success_flags)
    coeffs_all = np.asarray(coeffs_all)

    crop_label = "full" if fit_crop_pixels is None else f"crop{2 * fit_crop_pixels}px"

    results_path = os.path.join(
        OUTDIR,
        f"WF_phase_farfieldLP_fit_results_{n_test}wfs_{n_modes}modes_{crop_label}_{run_stamp}.npz",
    )

    np.savez_compressed(
        results_path,
        rms_phase_errors=rms_phase_errors,
        mean_abs_phase_errors=mean_abs_phase_errors,
        costs=costs,
        nfevs=nfevs,
        success_flags=success_flags,
        coeffs_all=coeffs_all,
        labels=np.asarray(labels),
        lm_values=lm_values,
        n_modes=n_modes,
        n_test=n_test,
        wavefront_path=wavefront_path,
        wavefront_key=wavefront_key,
        fit_crop_pixels=-1 if fit_crop_pixels is None else fit_crop_pixels,
        target_phase_units=TARGET_PHASE_UNITS,
        remove_target_piston=REMOVE_TARGET_PISTON,
    )

    print("\nSaved numerical results to:", results_path)

    csv_path = os.path.join(
        OUTDIR,
        f"WF_phase_farfieldLP_fit_summary_{n_test}wfs_{n_modes}modes_{crop_label}_{run_stamp}.csv",
    )

    summary = np.column_stack([
        np.arange(n_test),
        rms_phase_errors,
        mean_abs_phase_errors,
        costs,
        nfevs,
        success_flags.astype(int),
    ])

    np.savetxt(
        csv_path,
        summary,
        delimiter=",",
        header="sample,rms_phase_error_rad,mean_abs_phase_error_rad,cost,nfev,success",
        comments="",
    )

    print("Saved CSV summary to:", csv_path)

    print("\n==============================")
    print("Wavefront phase far-field LP fitting results")
    print("==============================")
    print("N modes:", n_modes)
    print("N wavefronts:", n_test)
    print("Fit crop:", crop_label)
    print("Mean RMS phase error [rad]:", np.mean(rms_phase_errors))
    print("Median RMS phase error [rad]:", np.median(rms_phase_errors))
    print("Mean absolute phase error [rad]:", np.mean(mean_abs_phase_errors))
    print("Success rate:", np.mean(success_flags))

    example_plot_path = os.path.join(
        OUTDIR,
        f"WF_phase_farfieldLP_fit_example_{n_modes}modes_{crop_label}_{run_stamp}.png",
    )

    plot_phase_fit_example(
        target_phase=example_target,
        phase_fit=example_fit,
        phase_residual=example_residual,
        title=f"Wavefront phase fit using {n_modes} far-field LP modes",
        outpath=example_plot_path,
        plot_crop_pixels=plot_crop_pixels,
    )

    print("Saved example phase plot to:", example_plot_path)

    coeff_plot_path = os.path.join(
        OUTDIR,
        f"WF_phase_farfieldLP_fit_coeffs_example_{n_modes}modes_{crop_label}_{run_stamp}.png",
    )

    save_coeff_plot(
        coeffs=example_coeffs,
        labels=labels,
        lm_values=lm_values,
        outpath=coeff_plot_path,
    )

    print("Saved coefficient plot to:", coeff_plot_path)

    plt.show()


if __name__ == "__main__":
    main()
