"""
Standalone demonstration: fit LP-mode fields to a smiley-face target using
nonlinear least squares.

This keeps the near-field LP-mode generation from the current wavefront-fitting
code, but replaces measured wavefront data with an image target.

The smiley image is treated as a target complex FIELD whose amplitude is the
image and whose phase is zero everywhere,

    E_target(x, y) = A_smiley(x, y) * exp(i * 0),

and the fitted field is

    E_fit(x, y) = sum_m c_m E_m(x, y).

scipy.optimize.least_squares adjusts the complex modal coefficients so that
E_fit best matches E_target. Because the residual is complex it is passed to
the optimiser as its real and imaginary parts stacked into one real vector,
so both amplitude and phase are constrained.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import shift as ndimage_shift, zoom
from scipy.optimize import least_squares

from lanternfiber import lanternfiber


# -------------------------------------------------------------------------
# Default settings
# -------------------------------------------------------------------------

# Change this to your folder if needed.
#DATADIR = "/home/manav/PL-NN-testdata_forDec2025/"
DATADIR = './' #'/Users/manavkalra/Downloads/PL-NN-testdata_forDec2025/'
OUTDIR = DATADIR

# Save the smiley image as this file, or change the path here.
SMILEY_PATH = os.path.join(DATADIR, "smiley3.jpeg")

# The supplied image is black-on-white. Inverting it makes the black smiley
# strokes correspond to high target intensity.
INVERT_SMILEY = True

# Downsample the image for fitting. 128x128 is plenty for demonstrating
# the modal fit and is much faster than fitting the original ~554x554 image.
TARGET_SIZE = 256

# Fibre parameters. Length units are micrometres.
N_CORE = 1.44
N_CLADDING = 1.4345
WAVELENGTH = 1.55
CORE_RADIUS = 32.8 / 2

# Number of LP modes to fit. Leave as None to use every mode the fibre supports,
# so that changing the fibre parameters above needs no change here.
N_MODES = None
NPIX = 256
MAX_R = 3
PAD_FACTOR = 1

# Nonlinear field fitting.
MAX_NFEV = 1000
N_RESTARTS = 3
RNG_SEED = 42

# How the fitted field and the target are scaled before they are compared:
#   "peak"  - divide each by its peak amplitude; fits the shape only.
#   "power" - divide each by sqrt(total power); fits the shape with the total
#             power of the fit constrained to equal that of the target.
#   "none"  - no normalisation; the absolute scale is fitted as well, so the
#             modal coefficients carry real amplitudes rather than fractions.
FIELD_NORMALISATION = "none"#"peak"

# The phase residual is meaningless where the target is dark, so it is masked
# out below this fraction of the peak target amplitude.
PHASE_MASK_FRACTION = 0.05

# Whether to shift the target onto the centre of the mode grid before fitting.
# The LP modes are all centred on the fibre core, so an off-centre target forces
# asymmetry into the residual that says nothing about the mode basis.
#   "none"     - leave the image where it is.
#   "centroid" - align the target's amplitude-weighted centroid with the modes.
#   "bbox"     - align the centre of the target's bright bounding box instead;
#                less sensitive to faint background, more sensitive to outliers.
CENTRE_TARGET = "bbox"

# Pixels above this fraction of the peak amplitude count as bright for "bbox".
CENTRE_BBOX_THRESHOLD = 0.5

# Fourier-grid mapping.
FOURIER_MAPPING_MODE = "custom"
TARGET_FMAX_X = 0.105
TARGET_FMAX_Y = 0.105

SAVE_FOURIER_DIAGNOSTIC = True
DIAGNOSTIC_MODE_NUM = 0
EDGE_WARNING_RATIO = 1e-3

FIELD_EPS = 1e-12


# -------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------

def field_centre(field: np.ndarray, mode: str = "centroid",
                 threshold: float = CENTRE_BBOX_THRESHOLD) -> Tuple[float, float]:
    """
    Locate the (y, x) centre of a field, either as its amplitude-weighted
    centroid or as the centre of the bounding box of its bright pixels.
    """
    amplitude = np.abs(np.asarray(field))
    ny, nx = amplitude.shape[-2:]
    grid_centre = ((ny - 1) / 2.0, (nx - 1) / 2.0)

    if mode == "centroid":
        total = np.sum(amplitude)

        if total <= 0:
            return grid_centre

        cy = float(np.sum(np.arange(ny) * np.sum(amplitude, axis=1)) / total)
        cx = float(np.sum(np.arange(nx) * np.sum(amplitude, axis=0)) / total)
        return cy, cx

    if mode == "bbox":
        bright = amplitude > threshold * np.max(amplitude)

        if not bright.any():
            return grid_centre

        ys, xs = np.nonzero(bright)
        return float((ys.min() + ys.max()) / 2.0), float((xs.min() + xs.max()) / 2.0)

    raise ValueError(f"Unknown centring mode: {mode!r}")


def centre_field_on(field: np.ndarray, reference_centre: Tuple[float, float],
                    mode: str = CENTRE_TARGET) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Shift a complex field so that its centre lands on reference_centre.

    Returns the shifted field and the (dy, dx) applied, in pixels. Real and
    imaginary parts are interpolated separately, as for the mode resizing.
    """
    field = np.asarray(field, dtype=np.complex128)

    if mode == "none":
        return field, (0.0, 0.0)

    cy, cx = field_centre(field, mode)
    dy = reference_centre[0] - cy
    dx = reference_centre[1] - cx

    real_shifted = ndimage_shift(field.real, (dy, dx), order=1, mode="constant", cval=0.0)
    imag_shifted = ndimage_shift(field.imag, (dy, dx), order=1, mode="constant", cval=0.0)

    return real_shifted + 1j * imag_shifted, (dy, dx)


def centre_crop(arr: np.ndarray, crop_pixels: Optional[int]) -> np.ndarray:
    """Crop a centred square; crop_pixels is the half-width."""
    if crop_pixels is None:
        return arr

    if crop_pixels <= 0:
        raise ValueError("crop_pixels must be positive or None.")

    cy, cx = np.array(arr.shape[-2:]) // 2
    y0 = max(cy - crop_pixels, 0)
    y1 = min(cy + crop_pixels, arr.shape[-2])
    x0 = max(cx - crop_pixels, 0)
    x1 = min(cx + crop_pixels, arr.shape[-1])
    return arr[..., y0:y1, x0:x1]


def resize_complex_image_to_shape(z: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize a complex image by interpolating real and imaginary parts.
    Used only for the legacy full-FFT mapping mode.
    """
    z = np.asarray(z, dtype=np.complex128)
    target_shape = tuple(int(v) for v in target_shape)

    if z.shape == target_shape:
        return z

    zoom_y = target_shape[0] / z.shape[0]
    zoom_x = target_shape[1] / z.shape[1]

    real_resized = zoom(z.real, (zoom_y, zoom_x), order=1)
    imag_resized = zoom(z.imag, (zoom_y, zoom_x), order=1)

    return real_resized + 1j * imag_resized


def normalise_power(field: np.ndarray) -> np.ndarray:
    """Normalise a complex field so that sum(|field|^2) = 1."""
    field = np.asarray(field, dtype=np.complex128)
    power = np.sum(np.abs(field) ** 2)

    if power > 0:
        return field / np.sqrt(power)

    return field


def normalise_field(field: np.ndarray, mode: str = FIELD_NORMALISATION) -> np.ndarray:
    """Scale a complex field according to the chosen normalisation mode."""
    field = np.asarray(field, dtype=np.complex128)

    if mode == "peak":
        return field / (np.max(np.abs(field)) + FIELD_EPS)

    if mode == "power":
        return normalise_power(field)

    if mode == "none":
        return field

    raise ValueError(f"Unknown field normalisation: {mode!r}")


def maximum_edge_amplitude_ratio(field: np.ndarray) -> float:
    """Maximum boundary amplitude divided by maximum field amplitude."""
    amplitude = np.abs(field)
    peak = np.max(amplitude)

    if peak == 0:
        return 0.0

    edge_values = np.concatenate([amplitude[0, :], amplitude[-1, :], amplitude[:, 0], amplitude[:, -1]])

    return float(np.max(edge_values) / peak)


# -------------------------------------------------------------------------
# Load smiley target
# -------------------------------------------------------------------------

def load_smiley_target(image_path: str, target_size: int = TARGET_SIZE, invert: bool = INVERT_SMILEY) -> np.ndarray:
    """
    Load the smiley image and turn it into a normalised target complex FIELD.

    For the supplied black-on-white smiley:
        black strokes -> amplitude 1
        white background -> amplitude 0

    The image sets the field amplitude; the phase is zero everywhere.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Could not find smiley image:\n{image_path}\n\nSave the supplied smiley as smiley.png in DATADIR, or change SMILEY_PATH."
        )

    image = plt.imread(image_path)

    # Convert RGB/RGBA to greyscale.
    if image.ndim == 3:
        rgb = image[..., :3]
        image = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    image = np.asarray(image, dtype=float)

    # plt.imread can return either 0..1 or 0..255 depending on format.
    if np.max(image) > 1.0:
        image = image / 255.0

    # Centre-crop to a square first.
    ny, nx = image.shape
    side = min(ny, nx)
    y0 = (ny - side) // 2
    x0 = (nx - side) // 2
    image = image[y0:y0 + side, x0:x0 + side]

    # Resize to the fitting grid.
    zoom_y = target_size / image.shape[0]
    zoom_x = target_size / image.shape[1]
    image = zoom(image, (zoom_y, zoom_x), order=1)

    if invert:
        image = 1.0 - image

    # Numerical cleanup and normalisation.
    image = np.clip(image, 0.0, None)
    image = image - np.min(image)
    image = image / (np.max(image) + FIELD_EPS)

    # Treat the image as an amplitude and give it zero phase everywhere,
    # i.e. E_target = A * exp(i * 0).
    return image.astype(np.complex128)


def build_lanternfiber(npix: int = NPIX, max_r: float = MAX_R) -> lanternfiber:
    """Construct lanternfiber object and generate supported LP modes."""
    lf = lanternfiber(N_CORE, N_CLADDING, CORE_RADIUS, WAVELENGTH)

    lf.find_fiber_modes()

    lf.make_fiber_modes(npix=npix, max_r=max_r, show_plots=False, normtosum=True)

    return lf


def make_nearfield_lp_modes(target_shape, n_modes: Optional[int] = N_MODES):
    """
    Generate complex NEAR-FIELD LP modes and construct the fitting matrix.

    n_modes is the number of modes to use; pass None to use every mode the
    fibre supports, which is what the fibre parameters themselves determine.

    No Fourier transform is performed.
    """

    lf = build_lanternfiber(npix=NPIX, max_r=MAX_R)

    near_fields_raw = np.asarray(lf.allmodefields_rsoftorder)

    n_available = near_fields_raw.shape[0]

    print(f"\nFibre: n_core={N_CORE}, n_cladding={N_CLADDING}, core radius={CORE_RADIUS} um, "
          f"wavelength={WAVELENGTH} um")
    print(f"NA = {lf.NA:.4f}, V = {lf.V:.4f}")
    print("Total available LP scalar modes:", n_available)

    if n_modes is None:
        n_modes = n_available
        print("Using all available modes.")

    elif n_modes > n_available:
        raise ValueError(f"n_modes={n_modes}, but only {n_available} modes are available.")

    else:
        print("Using the first", n_modes, "of them.")

    near_modes = []
    labels = []
    lm_values = []

    for mode_num in range(n_modes):

        raw_mode = near_fields_raw[mode_num]

        # Complex near-field LP mode
        near_field = lf.make_complex_fld(raw_mode)

        label = lf.modelabels[mode_num] if mode_num < len(lf.modelabels) else f"mode_{mode_num}"

        # Resize directly to same pixel grid as target smiley
        mode_field = resize_complex_image_to_shape(near_field, target_shape)

        # Normalise each mode to equal total power
        mode_field = normalise_power(mode_field)

        near_modes.append(mode_field)

        labels.append(label)

        if hasattr(lf, "lp_mode_list") and mode_num < len(lf.lp_mode_list):
            lm_values.append(lf.lp_mode_list[mode_num])
        else:
            lm_values.append([np.nan, np.nan])

    near_modes = np.asarray(near_modes, dtype=np.complex128)

    n_modes_actual, ny, nx = near_modes.shape

    # Each column = one complex LP mode
    mode_matrix = near_modes.reshape(n_modes_actual, ny * nx).T

    # Normalise mode columns
    norms = np.sqrt(np.sum(np.abs(mode_matrix) ** 2, axis=0, keepdims=True))

    mode_matrix = mode_matrix / (norms + FIELD_EPS)

    print("Near-field LP mode shape used for fit:", (ny, nx))

    print("Mode matrix shape:", mode_matrix.shape)

    return (mode_matrix, ny, nx, labels, np.asarray(lm_values), near_modes)


# -------------------------------------------------------------------------
# Nonlinear least-squares INTENSITY fitting
# -------------------------------------------------------------------------

def unpack_coeffs(z: np.ndarray, n_modes: int, normalise: bool = True) -> np.ndarray:
    """
    Convert the optimiser's real vector into complex modal coefficients.

    With normalise=True the coefficient vector is scaled to unit norm, because
    the fitted field is itself normalised before being compared with the target;
    the overall coefficient scale then carries no information. Pass normalise as
    False when the absolute scale is being fitted (FIELD_NORMALISATION="none").
    """
    coeffs = z[:n_modes] + 1j * z[n_modes:]

    if not normalise:
        return coeffs

    norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))

    if norm > 0:
        coeffs = coeffs / norm

    return coeffs


def fit_coeffs_to_target_field(
    mode_matrix: np.ndarray,
    target_field: np.ndarray,
    max_nfev: int = MAX_NFEV,
    n_restarts: int = N_RESTARTS,
    rng: Optional[np.random.Generator] = None,
    normalisation: str = FIELD_NORMALISATION,
):
    """
    Find complex LP coefficients that best reproduce the target complex FIELD.

        E_fit = M c

    Both the modes and the target are complex, so the residual E_fit - E_target
    is complex too. scipy.optimize.least_squares only accepts real residuals and
    real parameters, so the coefficients are passed in as 2*n_modes reals and the
    residual is returned as [real part, imaginary part] stacked end to end.

    normalisation selects how the fit and the target are scaled before they are
    compared; see FIELD_NORMALISATION. With "none" the absolute scale is fitted
    too, so the coefficients are left unnormalised.
    """
    if rng is None:
        rng = np.random.default_rng()

    M = np.asarray(mode_matrix, dtype=np.complex128)

    target = np.asarray(target_field, dtype=np.complex128)

    n_modes = M.shape[1]

    if M.shape[0] != target.size:
        raise ValueError(f"Mode matrix has {M.shape[0]} pixels, but target has {target.size}.")

    # "peak" and "power" discard the overall scale; "none" keeps it and fits it.
    normalise_coeffs = normalisation != "none"

    target = normalise_field(target, normalisation)

    target_flat = target.reshape(-1)

    def residual(z: np.ndarray) -> np.ndarray:
        coeffs = unpack_coeffs(z, n_modes, normalise=normalise_coeffs)

        # Coherently add the complex LP fields.
        field_flat = M @ coeffs

        field_flat = normalise_field(field_flat, normalisation)

        # Complex difference, split into real and imaginary parts for scipy.
        difference = field_flat - target_flat

        return np.concatenate([difference.real, difference.imag])

    best_result = None

    for restart in range(n_restarts):

        print(f"  least-squares restart {restart + 1}/{n_restarts}")

        if restart == 0:
            # First attempt starts from LP01.
            z0 = np.zeros(2 * n_modes, dtype=float)
            z0[0] = 1.0

        else:
            # Other attempts use random complex modal mixtures.
            z0 = rng.normal(0.0, 1.0, size=2 * n_modes)

        if not normalise_coeffs:
            # The absolute scale is a free parameter, so start it near the
            # target's scale rather than at an arbitrary unit amplitude.
            z0 = z0 / (np.linalg.norm(z0) + FIELD_EPS) * np.linalg.norm(target_flat)

        result = least_squares(residual, z0, max_nfev=max_nfev, verbose=0)

        print("    cost =", result.cost, "| nfev =", result.nfev, "| success =", result.success)

        if best_result is None or result.cost < best_result.cost:
            best_result = result

    coeffs_fit = unpack_coeffs(best_result.x, n_modes, normalise=normalise_coeffs)

    field_fit = (M @ coeffs_fit).reshape(target.shape)

    field_fit = normalise_field(field_fit, normalisation)

    field_residual = field_fit - target

    rms_field_error = np.sqrt(np.mean(np.abs(field_residual) ** 2))

    mean_abs_field_error = np.mean(np.abs(field_residual))

    return coeffs_fit, field_fit, field_residual, rms_field_error, mean_abs_field_error, best_result


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def save_fit_plot(
    target_amplitude: np.ndarray,
    amplitude_fit: np.ndarray,
    amplitude_residual: np.ndarray,
    phase_residual: np.ndarray,
    rms_error: float,
    rms_phase_error: float,
    n_modes: int,
    outpath: str,
) -> None:
    """Save target, LP fit, amplitude residual and phase residual."""
    vmax_residual = np.max(np.abs(amplitude_residual))

    if vmax_residual == 0:
        vmax_residual = 1.0

    # Both amplitude panels share a scale, which matters when the absolute
    # scale is fitted and the fit does not peak at 1.
    vmax_amplitude = max(np.max(target_amplitude), np.max(amplitude_fit))

    if vmax_amplitude == 0:
        vmax_amplitude = 1.0

    # Masked-out pixels (dark target, undefined phase) are drawn in grey.
    phase_cmap = plt.get_cmap("twilight_shifted").copy()
    phase_cmap.set_bad("0.8")

    # Scale the phase panel to the data, or a small residual is invisible on a
    # full +/-pi scale. Fall back to +/-pi when there is essentially no residual.
    vmax_phase = np.nanmax(np.abs(phase_residual)) if np.any(np.isfinite(phase_residual)) else np.pi

    if not np.isfinite(vmax_phase) or vmax_phase < 1e-6:
        vmax_phase = np.pi

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(target_amplitude, cmap="gray", origin="lower", vmin=0, vmax=vmax_amplitude)
    plt.title("Target smiley amplitude")
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.imshow(amplitude_fit, cmap="gray", origin="lower", vmin=0, vmax=vmax_amplitude)
    plt.title(f"Fitted amplitude\n{n_modes} near-field LP modes")
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.imshow(amplitude_residual, cmap="bwr", origin="lower", vmin=-vmax_residual, vmax=vmax_residual)
    plt.title(f"Fit - target amplitude residual\nRMS = {rms_error:.4f}")
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.imshow(phase_residual, cmap=phase_cmap, origin="lower", vmin=-vmax_phase, vmax=vmax_phase)
    plt.title(f"Fit - target phase residual [rad]\nRMS = {rms_phase_error:.4f}")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_coeff_plot(coeffs: np.ndarray, labels: list[str], lm_values: np.ndarray, outpath: str) -> None:
    """Save amplitude and phase of fitted complex LP coefficients."""
    amplitudes = np.abs(coeffs)
    phases = np.angle(coeffs)
    x = np.arange(len(coeffs))

    tick_labels = []

    for index, label in enumerate(labels):
        l_val, m_val = lm_values[index]

        if np.isfinite(l_val) and np.isfinite(m_val):
            tick_labels.append(f"{label}\n(l={int(l_val)},m={int(m_val)})")
        else:
            tick_labels.append(label)

    plt.figure(figsize=(max(10, len(coeffs) * 0.7), 6))

    plt.subplot(2, 1, 1)
    plt.bar(x, amplitudes)
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

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--smiley", type=str, default=SMILEY_PATH, help="Path to smiley image.")

    parser.add_argument("--target-size", type=int, default=TARGET_SIZE, help="Square image size used for fitting.")

    parser.add_argument("--n-modes", type=int, default=N_MODES,
                        help="Number of LP modes to fit. Omit to use every mode the fibre supports.")

    parser.add_argument("--max-nfev", type=int, default=MAX_NFEV)

    parser.add_argument("--n-restarts", type=int, default=N_RESTARTS)

    parser.add_argument("--pad-factor", type=int, default=PAD_FACTOR)

    parser.add_argument("--fourier-mapping", choices=["full_fft", "fibre_na", "custom"], default=FOURIER_MAPPING_MODE)

    parser.add_argument("--target-fmax-x", type=float, default=TARGET_FMAX_X)

    parser.add_argument("--target-fmax-y", type=float, default=TARGET_FMAX_Y)

    parser.add_argument("--centre-target", choices=["none", "centroid", "bbox"],
                        default=CENTRE_TARGET,
                        help="Shift the target onto the centre of the mode grid before fitting.")

    parser.add_argument("--field-normalisation", choices=["peak", "power", "none"],
                        default=FIELD_NORMALISATION,
                        help="How the fit and target are scaled before comparison.")

    parser.add_argument("--no-fourier-diagnostic", action="store_true")

    args = parser.parse_args()

    if args.n_modes is not None and args.n_modes <= 0:
        raise ValueError("--n-modes must be positive, or omitted to use all available modes.")

    if args.target_size <= 0:
        raise ValueError("--target-size must be positive.")

    if args.max_nfev <= 0:
        raise ValueError("--max-nfev must be positive.")

    if args.n_restarts <= 0:
        raise ValueError("--n-restarts must be positive.")

    if args.pad_factor < 1:
        raise ValueError("--pad-factor must be at least 1.")

    os.makedirs(OUTDIR, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rng = np.random.default_rng(RNG_SEED)

    # -----------------------------------------------------
    # Smiley target
    # -----------------------------------------------------

    target_field = load_smiley_target(image_path=args.smiley, target_size=args.target_size, invert=INVERT_SMILEY)

    target_shape = target_field.shape

    print("\nTarget smiley shape:", target_shape)

    # -----------------------------------------------------
    # Current Fourier LP-mode pipeline
    # -----------------------------------------------------

    diagnostic_path = None

    if SAVE_FOURIER_DIAGNOSTIC and not args.no_fourier_diagnostic:
        diagnostic_path = os.path.join(
            OUTDIR,
            f"smiley_fullFFT_diagnostic_mode{DIAGNOSTIC_MODE_NUM}_pad{args.pad_factor}_{run_stamp}.png",
        )

    mode_matrix, mode_ny, mode_nx, labels, lm_values, near_modes = make_nearfield_lp_modes(
        target_shape=target_shape, n_modes=args.n_modes)

    # However many modes the fibre turned out to support, that is what is fitted.
    n_modes_fitted = mode_matrix.shape[1]

    # The modes are all centred on the fibre core, so optionally move the target
    # there too. LP01 is the fundamental mode, so its centroid is that centre.
    mode_centre = field_centre(near_modes[0], "centroid")

    target_field, target_shift = centre_field_on(target_field, mode_centre, args.centre_target)

    if args.centre_target != "none":
        print(f"Centred target ({args.centre_target}) on mode centre "
              f"(y={mode_centre[0]:.2f}, x={mode_centre[1]:.2f}): "
              f"shifted by dy={target_shift[0]:+.2f}, dx={target_shift[1]:+.2f} pixels")

    # -----------------------------------------------------
    # Nonlinear least-squares complex field fit
    # -----------------------------------------------------

    print("\nFitting LP-mode field to smiley...")

    coeffs_fit, field_fit, field_residual, rms_error, mae_error, result = fit_coeffs_to_target_field(
        mode_matrix, target_field, args.max_nfev, args.n_restarts, rng,
        normalisation=args.field_normalisation)

    # The fit normalises the target internally, so do the same here for plotting.
    target_scaled = normalise_field(target_field, args.field_normalisation)

    # Amplitude-only views, for plotting and for comparison with an intensity fit.
    target_amplitude = np.abs(target_scaled)
    amplitude_fit = np.abs(field_fit)
    amplitude_residual = amplitude_fit - target_amplitude
    rms_amplitude_error = np.sqrt(np.mean(amplitude_residual ** 2))

    # Phase residual, wrapped to (-pi, pi]. The target phase is zero everywhere,
    # so this is just the phase the fit ended up with, but conj() keeps it
    # correct for any target. It is undefined where the target is dark.
    phase_residual = np.angle(field_fit * np.conj(target_scaled))
    bright = target_amplitude > PHASE_MASK_FRACTION * np.max(target_amplitude)
    phase_residual = np.where(bright, phase_residual, np.nan)
    rms_phase_error = np.sqrt(np.nanmean(phase_residual ** 2)) if bright.any() else np.nan
    max_fitted_phase = np.nanmax(np.abs(phase_residual)) if bright.any() else np.nan

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\n==============================")
    print("Smiley near-field LP field fit")
    print("==============================")
    print("N modes:", n_modes_fitted)
    print("Target size:", target_shape)
    print("Target centring:", args.centre_target)
    print("Field normalisation:", args.field_normalisation)
    print("Fourier mapping:", args.fourier_mapping)
    print("Padding factor:", args.pad_factor)
    print("RMS complex field error:", rms_error)
    print("Mean absolute complex field error:", mae_error)
    print("RMS amplitude-only error:", rms_amplitude_error)
    print("RMS phase residual [rad]:", rms_phase_error)
    print("Max |phase residual| [rad]:", max_fitted_phase)
    print("Peak fitted amplitude:", np.max(amplitude_fit), "| peak target amplitude:", np.max(target_amplitude))
    print("Total fitted power:", np.sum(amplitude_fit ** 2), "| total target power:", np.sum(target_amplitude ** 2))
    print("Least-squares cost:", result.cost)
    print("nfev:", result.nfev)
    print("Success:", result.success)

    print("\nFitted coefficients")

    for i, coeff in enumerate(coeffs_fit):
        print(f"{i:02d} {labels[i]:15s} |c|={np.abs(coeff):.6f} phase={np.angle(coeff):+.6f}")

    # -----------------------------------------------------
    # Save numerical result
    # -----------------------------------------------------

    results_path = os.path.join(
        OUTDIR,
        f"smiley_nearfieldLP_fit_{n_modes_fitted}modes_{args.target_size}px_{args.fourier_mapping}_{run_stamp}.npz",
    )

    np.savez_compressed(
        results_path,
        target_field=target_field,
        field_fit=field_fit,
        field_residual=field_residual,
        amplitude_fit=amplitude_fit,
        amplitude_residual=amplitude_residual,
        phase_residual=phase_residual,
        coeffs_fit=coeffs_fit,
        labels=np.asarray(labels),
        lm_values=lm_values,
        near_modes=near_modes,
        rms_field_error=rms_error,
        mean_abs_field_error=mae_error,
        rms_amplitude_error=rms_amplitude_error,
        rms_phase_error=rms_phase_error,
        field_normalisation=args.field_normalisation,
        n_modes=n_modes_fitted,
        centre_target=args.centre_target,
        target_shift=np.asarray(target_shift),
        least_squares_cost=result.cost,
        least_squares_nfev=result.nfev,
        least_squares_success=result.success,
        fourier_mapping=args.fourier_mapping,
    )

    print("\nSaved numerical result to:", results_path)

    # -----------------------------------------------------
    # Save plots
    # -----------------------------------------------------

    fit_plot_path = os.path.join(
        OUTDIR,
        f"smiley_nearfieldLP_fit_{n_modes_fitted}modes_{args.target_size}px_{run_stamp}.png",
    )

    save_fit_plot(target_amplitude, amplitude_fit, amplitude_residual, phase_residual, rms_amplitude_error,
                  rms_phase_error, n_modes_fitted, fit_plot_path)

    print("Saved fit plot to:", fit_plot_path)

    coeff_plot_path = os.path.join(OUTDIR, f"smiley_nearfieldLP_coeffs_{n_modes_fitted}modes_{run_stamp}.png")

    save_coeff_plot(coeffs=coeffs_fit, labels=labels, lm_values=lm_values, outpath=coeff_plot_path)

    print("Saved coefficient plot to:", coeff_plot_path)

    if diagnostic_path is not None:
        print("Saved Fourier diagnostic to:", diagnostic_path)


if __name__ == "__main__":
    main()
