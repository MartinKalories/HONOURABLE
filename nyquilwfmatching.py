"""
Phase-only least-squares fitting of measured wavefront phase images using
Fourier-transformed LP modes from lanternfiber.py.

The important Fourier-sampling choices are explicit:

1. NPIX and MAX_R determine the near-field pixel scale through
   lf.microns_per_pixel.
2. PAD_FACTOR changes the spacing of the FFT samples but not the Nyquist limit.
3. FOURIER_MAPPING_MODE determines which physical Fourier range is mapped onto
   the target wavefront image.

Put this file in the same directory as lanternfiber.py, or make sure that
lanternfiber.py is on the Python path.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from scipy.optimize import least_squares

from lanternfiber import lanternfiber


# -------------------------------------------------------------------------
# Default settings
# -------------------------------------------------------------------------

DATADIR = "/home/manav//PL-NN-testdata_forDec2025/"
OUTDIR = DATADIR

WAVEFRONT_NPZ_FILENAME = (
    "slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"
)
WAVEFRONT_KEY = None

# Fibre parameters. The length units are micrometres throughout this script.
N_CORE = 1.44
N_CLADDING = 1.4345
WAVELENGTH = 1.55
CORE_RADIUS = 32.8 / 2

# LP-mode generation and fitting.
N_MODES = 10
N_TEST = 5
NPIX = 200          # lanternfiber returns a 2*NPIX by 2*NPIX mode image
MAX_R = 1           # outer calculation radius in units of the core radius
PAD_FACTOR = 4      # zero-padding factor applied before the FFT
MAX_NFEV = 1000
N_RESTARTS = 3
RNG_SEED = 42

# Fourier-grid mapping.
#
# "full_fft": reproduces the old behaviour by resizing the complete FFT range.
#             It is useful for comparison, but it does not calibrate the target
#             image to a physical frequency range.
#
# "fibre_na": maps the target image edges to approximately +/- NA/lambda.
#             Use this only when the target image spans the full fibre output
#             angular range.
#
# "custom":   maps the target image edges to +/- TARGET_FMAX_X and
#             +/- TARGET_FMAX_Y, in cycles per micrometre.
FOURIER_MAPPING_MODE = "fibre_na"
TARGET_FMAX_X = None
TARGET_FMAX_Y = None

# Print and save one full-FFT diagnostic image with physical frequency axes.
SAVE_FOURIER_DIAGNOSTIC = True
DIAGNOSTIC_MODE_NUM = 0

# Warn when the near-field mode has appreciable amplitude at the array edge.
EDGE_WARNING_RATIO = 1e-3

# Target phase units.
TARGET_PHASE_UNITS = "radians"  # "radians", "waves", or "degrees"
REMOVE_TARGET_PISTON = False

# Optional centre crop after each far-field mode has been mapped to the target
# grid. crop_pixels is a half-width: 80 gives a 160 by 160 crop.
CENTRE_CROP_PIXELS = None
PLOT_CROP_PIXELS = 100

FIELD_EPS = 1e-12


# -------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------


def wrap_phase(phi: np.ndarray) -> np.ndarray:
    """Wrap phase to [-pi, pi)."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def circular_mean_phase(phi: np.ndarray) -> float:
    """Return the circular mean of a phase image."""
    return float(np.angle(np.mean(np.exp(1j * phi))))


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


def resize_complex_image_to_shape(
    z: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a complex image by interpolating the real and imaginary parts.

    This is retained for the legacy full-FFT mapping mode. For physically
    calibrated mapping, use resample_complex_fourier_field().
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


def maximum_edge_amplitude_ratio(field: np.ndarray) -> float:
    """Maximum boundary amplitude divided by the maximum field amplitude."""
    amplitude = np.abs(field)
    peak = np.max(amplitude)
    if peak == 0:
        return 0.0

    edge_values = np.concatenate(
        [
            amplitude[0, :],
            amplitude[-1, :],
            amplitude[:, 0],
            amplitude[:, -1],
        ]
    )
    return float(np.max(edge_values) / peak)


# -------------------------------------------------------------------------
# Load target wavefronts
# -------------------------------------------------------------------------


def possible_wavefront_paths(datadir: str, filename: str) -> list[str]:
    """Return likely paths for the supplied wavefront filename."""
    stem = filename[:-4] if filename.endswith(".npz") else filename

    candidates = [
        os.path.join(datadir, filename),
        os.path.join(datadir, stem + ".npz"),
        os.path.join(datadir, stem + "-WFs.npz"),
        os.path.join(datadir, stem + "-WF.npz"),
        os.path.join(datadir, "pllabdata_20240605_singlepsf_01_" + stem + ".npz"),
        os.path.join(datadir, "pllabdata_20240605_singlepsf_01_" + stem + "-WFs.npz"),
        os.path.join(datadir, "pllabdata_20240605_singlepsf_01_" + stem + "-WF.npz"),
    ]

    unique_paths = []
    for path in candidates:
        if path not in unique_paths:
            unique_paths.append(path)
    return unique_paths


def choose_wavefront_key(npz_file, requested_key: Optional[str] = None) -> str:
    """Choose a wavefront-like numeric array from an NPZ file."""
    keys = list(npz_file.keys())

    print("\nAvailable keys in wavefront npz:")
    for key in keys:
        arr = npz_file[key]
        print(
            f"  {key}: shape={getattr(arr, 'shape', None)}, "
            f"dtype={getattr(arr, 'dtype', None)}"
        )

    if requested_key is not None:
        if requested_key not in npz_file:
            raise KeyError(
                f"WAVEFRONT_KEY='{requested_key}' was not found. "
                f"Available keys are: {keys}"
            )
        return requested_key

    priority_words = ["wf", "wavefront", "phase", "phi", "slm"]
    for word in priority_words:
        for key in keys:
            arr = npz_file[key]
            if word in key.lower() and hasattr(arr, "ndim") and arr.ndim >= 2:
                return key

    for key in keys:
        arr = npz_file[key]
        if (
            hasattr(arr, "ndim")
            and arr.ndim >= 2
            and np.issubdtype(arr.dtype, np.number)
        ):
            return key

    raise RuntimeError("Could not find a suitable wavefront array in the NPZ file.")


def standardise_wavefront_array_shape(arr: np.ndarray) -> np.ndarray:
    """Convert common wavefront-array layouts to (n_images, ny, nx)."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        # Likely layout: (ny, nx, n_images), for example (128, 128, 10000).
        if arr.shape[0] == arr.shape[1] and arr.shape[2] != arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)
    elif arr.ndim == 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        else:
            raise ValueError(f"Do not know how to handle 4D shape {arr.shape}.")
    else:
        raise ValueError(f"Do not know how to handle wavefront shape {arr.shape}.")

    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D wavefront array, got {arr.shape}.")

    return arr.astype(float)


def load_wavefronts(
    datadir: str,
    filename: str,
    key: Optional[str] = None,
):
    """Load target wavefront phase images."""
    paths = possible_wavefront_paths(datadir, filename)
    wavefront_path = next((path for path in paths if os.path.exists(path)), None)

    if wavefront_path is None:
        print("\nTried these wavefront paths:")
        for path in paths:
            print(" ", path)
        raise FileNotFoundError(
            "Could not find the wavefront NPZ file. Check DATADIR and "
            "WAVEFRONT_NPZ_FILENAME."
        )

    print("Loading wavefronts from:", wavefront_path)
    npz_file = np.load(wavefront_path, allow_pickle=True)
    chosen_key = choose_wavefront_key(npz_file, requested_key=key)
    print("Using wavefront key:", chosen_key)

    wavefronts = standardise_wavefront_array_shape(npz_file[chosen_key])
    print("Wavefront array shape after standardising:", wavefronts.shape)
    return wavefronts, wavefront_path, chosen_key


def prepare_target_phase(target_raw: np.ndarray) -> np.ndarray:
    """Convert one target image to wrapped phase in radians."""
    target = np.squeeze(target_raw).astype(float)
    units = TARGET_PHASE_UNITS.lower()

    if units == "waves":
        target = target * 2 * np.pi
    elif units == "degrees":
        target = np.deg2rad(target)
    elif units != "radians":
        raise ValueError(
            "TARGET_PHASE_UNITS must be 'radians', 'waves', or 'degrees'."
        )

    target = wrap_phase(target)
    if REMOVE_TARGET_PISTON:
        target = wrap_phase(target - circular_mean_phase(target))
    return target


# -------------------------------------------------------------------------
# Fourier-domain LP-mode generation
# -------------------------------------------------------------------------


def near_to_far_field(
    near_field: np.ndarray,
    microns_per_pixel: float,
    pad_factor: int = 1,
    normtosum: bool = True,
):
    """
    Fourier transform a centred near-field mode and return physical frequency axes.

    Parameters
    ----------
    near_field
        Complex near-field mode.
    microns_per_pixel
        Near-field sampling in micrometres per pixel.
    pad_factor
        Integer zero-padding factor. Padding decreases FFT frequency spacing but
        does not alter the Nyquist frequency.

    Returns
    -------
    far_field
        Centred complex FFT.
    fy, fx
        Spatial-frequency axes in cycles per micrometre.
    """
    near_field = np.asarray(near_field, dtype=np.complex128)

    if not isinstance(pad_factor, (int, np.integer)):
        raise TypeError("pad_factor must be an integer.")
    if pad_factor < 1:
        raise ValueError("pad_factor must be at least 1.")
    if microns_per_pixel <= 0:
        raise ValueError("microns_per_pixel must be positive.")

    if pad_factor > 1:
        ny, nx = near_field.shape
        new_ny = ny * pad_factor
        new_nx = nx * pad_factor

        before_y = (new_ny - ny) // 2
        after_y = new_ny - ny - before_y
        before_x = (new_nx - nx) // 2
        after_x = new_nx - nx - before_x

        near_field = np.pad(
            near_field,
            ((before_y, after_y), (before_x, after_x)),
            mode="constant",
            constant_values=0,
        )

    far_field = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(near_field))
    )

    ny_fft, nx_fft = far_field.shape
    fx = np.fft.fftshift(
        np.fft.fftfreq(nx_fft, d=microns_per_pixel)
    )
    fy = np.fft.fftshift(
        np.fft.fftfreq(ny_fft, d=microns_per_pixel)
    )

    if normtosum:
        far_field = normalise_power(far_field)

    return far_field, fy, fx


def resolve_target_frequency_range(
    lf: lanternfiber,
    mapping_mode: str,
    target_fmax_x: Optional[float],
    target_fmax_y: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve the target image's positive x and y edge frequencies."""
    mapping_mode = mapping_mode.lower()

    if mapping_mode == "full_fft":
        return None, None

    if mapping_mode == "fibre_na":
        fmax = float(lf.NA / lf.wavelength)
        return fmax, fmax

    if mapping_mode == "custom":
        if target_fmax_x is None or target_fmax_y is None:
            raise ValueError(
                "FOURIER_MAPPING_MODE='custom' requires both TARGET_FMAX_X "
                "and TARGET_FMAX_Y."
            )
        if target_fmax_x <= 0 or target_fmax_y <= 0:
            raise ValueError("TARGET_FMAX_X and TARGET_FMAX_Y must be positive.")
        return float(target_fmax_x), float(target_fmax_y)

    raise ValueError(
        "FOURIER_MAPPING_MODE must be 'full_fft', 'fibre_na', or 'custom'."
    )


def resample_complex_fourier_field(
    far_field: np.ndarray,
    fy_source: np.ndarray,
    fx_source: np.ndarray,
    target_shape: Tuple[int, int],
    target_fmax_y: float,
    target_fmax_x: float,
):
    """
    Interpolate a complex FFT onto a physically specified target-frequency grid.

    The target grid spans approximately [-fmax, +fmax) along each axis.
    """
    target_ny, target_nx = (int(target_shape[0]), int(target_shape[1]))

    target_dfx = 2.0 * target_fmax_x / target_nx
    target_dfy = 2.0 * target_fmax_y / target_ny

    fx_target = (np.arange(target_nx) - target_nx // 2) * target_dfx
    fy_target = (np.arange(target_ny) - target_ny // 2) * target_dfy

    source_min_x, source_max_x = float(fx_source[0]), float(fx_source[-1])
    source_min_y, source_max_y = float(fy_source[0]), float(fy_source[-1])

    if fx_target[0] < source_min_x or fx_target[-1] > source_max_x:
        raise ValueError(
            "Target x-frequency range exceeds the available FFT range. "
            "Increase NPIX or decrease the requested target range."
        )
    if fy_target[0] < source_min_y or fy_target[-1] > source_max_y:
        raise ValueError(
            "Target y-frequency range exceeds the available FFT range. "
            "Increase NPIX or decrease the requested target range."
        )

    target_fy_grid, target_fx_grid = np.meshgrid(
        fy_target,
        fx_target,
        indexing="ij",
    )
    points = np.column_stack(
        [target_fy_grid.ravel(), target_fx_grid.ravel()]
    )

    real_interp = RegularGridInterpolator(
        (fy_source, fx_source),
        far_field.real,
        bounds_error=True,
    )
    imag_interp = RegularGridInterpolator(
        (fy_source, fx_source),
        far_field.imag,
        bounds_error=True,
    )

    real_target = real_interp(points).reshape(target_shape)
    imag_target = imag_interp(points).reshape(target_shape)
    return real_target + 1j * imag_target, fy_target, fx_target


def recommended_padding_factor(
    near_field_shape: Tuple[int, int],
    microns_per_pixel: float,
    target_shape: Tuple[int, int],
    target_fmax_x: float,
    target_fmax_y: float,
) -> int:
    """
    Padding required for FFT spacing no larger than the target-grid spacing.

    This is a diagnostic, not an automatic requirement. The result may be large.
    """
    near_ny, near_nx = near_field_shape
    target_ny, target_nx = target_shape

    target_dfx = 2.0 * target_fmax_x / target_nx
    target_dfy = 2.0 * target_fmax_y / target_ny
    base_dfx = 1.0 / (near_nx * microns_per_pixel)
    base_dfy = 1.0 / (near_ny * microns_per_pixel)

    pad_x = int(np.ceil(base_dfx / target_dfx))
    pad_y = int(np.ceil(base_dfy / target_dfy))
    return max(1, pad_x, pad_y)


def print_fourier_diagnostics(
    lf: lanternfiber,
    near_field_shape: Tuple[int, int],
    fx: np.ndarray,
    fy: np.ndarray,
    target_shape: Tuple[int, int],
    target_fmax_x: Optional[float],
    target_fmax_y: Optional[float],
    pad_factor: int,
) -> None:
    """Print the physical near-field and FFT sampling information."""
    pixel_scale = float(lf.microns_per_pixel)
    nyquist = 1.0 / (2.0 * pixel_scale)
    dfx = float(fx[1] - fx[0])
    dfy = float(fy[1] - fy[0])

    print("\nFourier sampling diagnostics")
    print("----------------------------")
    print("Near-field shape:", near_field_shape)
    print("Near-field pixel scale [um/pixel]:", pixel_scale)
    print("Nyquist frequency [cycles/um]:", nyquist)
    print("Padding factor:", pad_factor)
    print("Padded FFT shape:", (len(fy), len(fx)))
    print("FFT x spacing [cycles/um]:", dfx)
    print("FFT y spacing [cycles/um]:", dfy)
    print("FFT x range [cycles/um]:", (float(fx[0]), float(fx[-1])))
    print("FFT y range [cycles/um]:", (float(fy[0]), float(fy[-1])))
    print("Fibre NA:", float(lf.NA))
    print("Fibre NA/lambda [cycles/um]:", float(lf.NA / lf.wavelength))

    if target_fmax_x is not None and target_fmax_y is not None:
        print("Mapped target x range [cycles/um]:", (-target_fmax_x, target_fmax_x))
        print("Mapped target y range [cycles/um]:", (-target_fmax_y, target_fmax_y))

        rec_pad = recommended_padding_factor(
            near_field_shape=near_field_shape,
            microns_per_pixel=pixel_scale,
            target_shape=target_shape,
            target_fmax_x=target_fmax_x,
            target_fmax_y=target_fmax_y,
        )
        print("Diagnostic padding for one FFT sample per target pixel:", rec_pad)
        print(
            "Current FFT samples across mapped x width:",
            2.0 * target_fmax_x / dfx,
        )
        print(
            "Current FFT samples across mapped y width:",
            2.0 * target_fmax_y / dfy,
        )


def save_full_fft_diagnostic(
    far_field: np.ndarray,
    fy: np.ndarray,
    fx: np.ndarray,
    label: str,
    outpath: str,
    target_fmax_x: Optional[float] = None,
    target_fmax_y: Optional[float] = None,
) -> None:
    """Save a far-field intensity plot with physical frequency axes."""
    intensity = np.abs(far_field) ** 2
    maximum = np.max(intensity)
    if maximum > 0:
        intensity = intensity / maximum

    plt.figure(figsize=(7, 6))
    plt.imshow(
        intensity,
        extent=[fx[0], fx[-1], fy[0], fy[-1]],
        origin="lower",
        aspect="equal",
    )

    if target_fmax_x is not None and target_fmax_y is not None:
        rectangle = plt.Rectangle(
            (-target_fmax_x, -target_fmax_y),
            2 * target_fmax_x,
            2 * target_fmax_y,
            fill=False,
            linewidth=1.5,
        )
        plt.gca().add_patch(rectangle)

    plt.xlabel(r"$f_x$ [cycles/$\mu$m]")
    plt.ylabel(r"$f_y$ [cycles/$\mu$m]")
    plt.title(f"{label}: full FFT intensity")
    plt.colorbar(label="Normalised intensity")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def build_lanternfiber(
    npix: int = NPIX,
    max_r: float = MAX_R,
) -> lanternfiber:
    """Construct a lanternfiber object and generate all supported LP modes."""
    lf = lanternfiber(N_CORE, N_CLADDING, CORE_RADIUS, WAVELENGTH)
    lf.find_fiber_modes()
    lf.make_fiber_modes(
        npix=npix,
        max_r=max_r,
        show_plots=False,
        normtosum=True,
    )
    return lf


def get_farfield_mode_for_plot(
    mode_num: int = 0,
    npix: int = NPIX,
    max_r: float = MAX_R,
    pad_factor: int = PAD_FACTOR,
):
    """
    Import-friendly helper for a separate plotting script.

    Returns
    -------
    lf, near_field, far_field, fy, fx, label
    """
    lf = build_lanternfiber(npix=npix, max_r=max_r)
    near_fields_raw = np.asarray(lf.allmodefields_rsoftorder)

    if not 0 <= mode_num < len(near_fields_raw):
        raise IndexError(
            f"mode_num={mode_num} is invalid; use 0 to {len(near_fields_raw) - 1}."
        )

    near_field = lf.make_complex_fld(near_fields_raw[mode_num])
    far_field, fy, fx = near_to_far_field(
        near_field=near_field,
        microns_per_pixel=lf.microns_per_pixel,
        pad_factor=pad_factor,
        normtosum=True,
    )
    label = lf.modelabels[mode_num]
    return lf, near_field, far_field, fy, fx, label


def make_farfield_lp_modes(
    target_shape: Tuple[int, int],
    n_modes: int,
    centre_crop_pixels: Optional[int] = None,
    pad_factor: int = PAD_FACTOR,
    mapping_mode: str = FOURIER_MAPPING_MODE,
    target_fmax_x: Optional[float] = TARGET_FMAX_X,
    target_fmax_y: Optional[float] = TARGET_FMAX_Y,
    diagnostic_outpath: Optional[str] = None,
):
    """Generate mapped complex far-field LP modes and the fitting matrix."""
    lf = build_lanternfiber(npix=NPIX, max_r=MAX_R)
    near_fields_raw = np.asarray(lf.allmodefields_rsoftorder)
    n_available = near_fields_raw.shape[0]

    print("\nTotal available LP scalar modes:", n_available)
    if n_modes > n_available:
        raise ValueError(
            f"n_modes={n_modes}, but only {n_available} modes are available."
        )

    resolved_fmax_x, resolved_fmax_y = resolve_target_frequency_range(
        lf=lf,
        mapping_mode=mapping_mode,
        target_fmax_x=target_fmax_x,
        target_fmax_y=target_fmax_y,
    )

    far_modes = []
    labels = []
    lm_values = []
    diagnostics_printed = False

    for mode_num in range(n_modes):
        raw_mode = near_fields_raw[mode_num]
        near_field = lf.make_complex_fld(raw_mode)
        label = (
            lf.modelabels[mode_num]
            if mode_num < len(lf.modelabels)
            else f"mode_{mode_num}"
        )

        edge_ratio = maximum_edge_amplitude_ratio(near_field)
        if edge_ratio > EDGE_WARNING_RATIO:
            print(
                f"Warning: {label} edge amplitude ratio is {edge_ratio:.3e}. "
                "Consider increasing MAX_R."
            )

        far_field_full, fy_source, fx_source = near_to_far_field(
            near_field=near_field,
            microns_per_pixel=lf.microns_per_pixel,
            pad_factor=pad_factor,
            normtosum=True,
        )

        if not diagnostics_printed:
            print_fourier_diagnostics(
                lf=lf,
                near_field_shape=near_field.shape,
                fx=fx_source,
                fy=fy_source,
                target_shape=target_shape,
                target_fmax_x=resolved_fmax_x,
                target_fmax_y=resolved_fmax_y,
                pad_factor=pad_factor,
            )
            diagnostics_printed = True

        if diagnostic_outpath is not None and mode_num == DIAGNOSTIC_MODE_NUM:
            save_full_fft_diagnostic(
                far_field=far_field_full,
                fy=fy_source,
                fx=fx_source,
                label=label,
                outpath=diagnostic_outpath,
                target_fmax_x=resolved_fmax_x,
                target_fmax_y=resolved_fmax_y,
            )

        if mapping_mode.lower() == "full_fft":
            far_field = resize_complex_image_to_shape(
                far_field_full,
                target_shape,
            )
        else:
            far_field, _, _ = resample_complex_fourier_field(
                far_field=far_field_full,
                fy_source=fy_source,
                fx_source=fx_source,
                target_shape=target_shape,
                target_fmax_y=resolved_fmax_y,
                target_fmax_x=resolved_fmax_x,
            )

        far_field = centre_crop(far_field, centre_crop_pixels)
        far_field = normalise_power(far_field)
        far_modes.append(far_field)
        labels.append(label)

        if hasattr(lf, "lp_mode_list") and mode_num < len(lf.lp_mode_list):
            lm_values.append(lf.lp_mode_list[mode_num])
        else:
            lm_values.append([np.nan, np.nan])

    far_modes = np.asarray(far_modes, dtype=np.complex128)
    n_modes_actual, ny, nx = far_modes.shape
    mode_matrix = far_modes.reshape(n_modes_actual, ny * nx).T

    norms = np.sqrt(
        np.sum(np.abs(mode_matrix) ** 2, axis=0, keepdims=True)
    )
    mode_matrix = mode_matrix / (norms + FIELD_EPS)

    print("Fourier mapping mode:", mapping_mode)
    print("Far-field LP mode shape used for fit:", (ny, nx))
    print("Mode matrix shape:", mode_matrix.shape)

    metadata = {
        "mapping_mode": mapping_mode,
        "target_fmax_x": resolved_fmax_x,
        "target_fmax_y": resolved_fmax_y,
        "microns_per_pixel": float(lf.microns_per_pixel),
        "NA": float(lf.NA),
        "V": float(lf.V),
        "pad_factor": int(pad_factor),
        "max_r": float(MAX_R),
        "npix": int(NPIX),
    }

    return (
        mode_matrix,
        ny,
        nx,
        labels,
        np.asarray(lm_values),
        far_modes,
        metadata,
    )


# -------------------------------------------------------------------------
# Phase-only least-squares fitting
# -------------------------------------------------------------------------


def unpack_coeffs(z: np.ndarray, n_modes: int) -> np.ndarray:
    """Convert a real optimisation vector into normalised complex coefficients."""
    coeffs = z[:n_modes] + 1j * z[n_modes:]
    norm = np.sqrt(np.sum(np.abs(coeffs) ** 2))
    if norm > 0:
        coeffs = coeffs / norm
    return coeffs


def fit_coeffs_to_target_phase(
    mode_matrix: np.ndarray,
    target_phase: np.ndarray,
    max_nfev: int = 1000,
    n_restarts: int = 3,
    rng: Optional[np.random.Generator] = None,
):
    """Fit complex LP coefficients to a wrapped target phase image."""
    if rng is None:
        rng = np.random.default_rng()

    M = np.asarray(mode_matrix, dtype=np.complex128)
    n_modes = M.shape[1]
    target_phase = wrap_phase(target_phase)
    target_unit = np.exp(1j * target_phase).reshape(-1)

    if M.shape[0] != target_unit.size:
        raise ValueError(
            f"Mode matrix has {M.shape[0]} pixels, but target has "
            f"{target_unit.size}."
        )

    def residual(z: np.ndarray) -> np.ndarray:
        coeffs = unpack_coeffs(z, n_modes)
        field_flat = M @ coeffs
        fit_unit = field_flat / (np.abs(field_flat) + FIELD_EPS)
        difference = fit_unit - target_unit
        return np.concatenate([difference.real, difference.imag])

    best_result = None
    for restart in range(n_restarts):
        if restart == 0:
            z0 = np.zeros(2 * n_modes)
            z0[0] = 1.0
        else:
            z0 = rng.normal(0.0, 1.0, size=2 * n_modes)

        result = least_squares(
            residual,
            z0,
            max_nfev=max_nfev,
            verbose=0,
        )

        if best_result is None or result.cost < best_result.cost:
            best_result = result

    coeffs_fit = unpack_coeffs(best_result.x, n_modes)
    field_fit = (M @ coeffs_fit).reshape(target_phase.shape)
    phase_fit = np.angle(field_fit)
    phase_residual = wrap_phase(target_phase - phase_fit)

    rms_phase_error = float(np.sqrt(np.mean(phase_residual ** 2)))
    mean_abs_phase_error = float(np.mean(np.abs(phase_residual)))

    return (
        coeffs_fit,
        field_fit,
        phase_fit,
        phase_residual,
        rms_phase_error,
        mean_abs_phase_error,
        best_result,
    )


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------


def plot_phase_fit_example(
    target_phase: np.ndarray,
    phase_fit: np.ndarray,
    phase_residual: np.ndarray,
    title: str,
    outpath: str,
    plot_crop_pixels: Optional[int] = None,
) -> None:
    """Save target, fitted and residual phase images."""
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


def save_coeff_plot(
    coeffs: np.ndarray,
    labels: list[str],
    lm_values: np.ndarray,
    outpath: str,
) -> None:
    """Save amplitude and phase plots for the fitted LP coefficients."""
    amplitudes = np.abs(coeffs)
    phases = np.angle(coeffs)
    x = np.arange(len(coeffs))

    tick_labels = []
    for index, label in enumerate(labels):
        l_val, m_val = lm_values[index]
        if np.isfinite(l_val) and np.isfinite(m_val):
            tick_labels.append(
                f"{label}\n(l={int(l_val)},m={int(m_val)})"
            )
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
    parser.add_argument("--n-modes", type=int, default=N_MODES)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--max-nfev", type=int, default=MAX_NFEV)
    parser.add_argument("--n-restarts", type=int, default=N_RESTARTS)
    parser.add_argument("--pad-factor", type=int, default=PAD_FACTOR)
    parser.add_argument(
        "--fourier-mapping",
        choices=["full_fft", "fibre_na", "custom"],
        default=FOURIER_MAPPING_MODE,
    )
    parser.add_argument("--target-fmax-x", type=float, default=TARGET_FMAX_X)
    parser.add_argument("--target-fmax-y", type=float, default=TARGET_FMAX_Y)
    parser.add_argument(
        "--crop-pixels",
        type=int,
        default=-1,
        help="Half-width of centre crop for fitting; -1 means no crop.",
    )
    parser.add_argument(
        "--plot-crop-pixels",
        type=int,
        default=PLOT_CROP_PIXELS,
        help="Half-width shown in plots; -1 means the full image.",
    )
    parser.add_argument("--wavefront-key", type=str, default=WAVEFRONT_KEY)
    parser.add_argument(
        "--wavefront-file",
        type=str,
        default=WAVEFRONT_NPZ_FILENAME,
    )
    parser.add_argument(
        "--no-fourier-diagnostic",
        action="store_true",
        help="Do not save the full-FFT diagnostic image.",
    )
    args = parser.parse_args()

    if args.n_modes <= 0:
        raise ValueError("--n-modes must be positive.")
    if args.n_test <= 0:
        raise ValueError("--n-test must be positive.")
    if args.pad_factor < 1:
        raise ValueError("--pad-factor must be at least 1.")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fit_crop_pixels = None if args.crop_pixels < 0 else args.crop_pixels
    plot_crop_pixels = (
        None if args.plot_crop_pixels < 0 else args.plot_crop_pixels
    )
    rng = np.random.default_rng(RNG_SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    wavefronts, wavefront_path, wavefront_key = load_wavefronts(
        DATADIR,
        args.wavefront_file,
        key=args.wavefront_key,
    )
    n_test = min(args.n_test, wavefronts.shape[0])

    first_target_phase = prepare_target_phase(wavefronts[0])
    target_shape = first_target_phase.shape
    print("\nTarget phase image shape:", target_shape)

    diagnostic_path = None
    if SAVE_FOURIER_DIAGNOSTIC and not args.no_fourier_diagnostic:
        diagnostic_path = os.path.join(
            OUTDIR,
            f"WF_fullFFT_diagnostic_mode{DIAGNOSTIC_MODE_NUM}_"
            f"pad{args.pad_factor}_{run_stamp}.png",
        )

    (
        mode_matrix,
        ny,
        nx,
        labels,
        lm_values,
        far_modes,
        fourier_metadata,
    ) = make_farfield_lp_modes(
        target_shape=target_shape,
        n_modes=args.n_modes,
        centre_crop_pixels=fit_crop_pixels,
        pad_factor=args.pad_factor,
        mapping_mode=args.fourier_mapping,
        target_fmax_x=args.target_fmax_x,
        target_fmax_y=args.target_fmax_y,
        diagnostic_outpath=diagnostic_path,
    )

    if diagnostic_path is not None:
        print("Saved Fourier diagnostic to:", diagnostic_path)

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

    for index in range(n_test):
        print(f"\nFitting wavefront {index + 1}/{n_test}")
        target_phase = prepare_target_phase(wavefronts[index])
        target_phase = centre_crop(target_phase, fit_crop_pixels)

        (
            coeffs_fit,
            field_fit,
            phase_fit,
            phase_residual,
            rms_error,
            mean_abs_error,
            result,
        ) = fit_coeffs_to_target_phase(
            mode_matrix=mode_matrix,
            target_phase=target_phase,
            max_nfev=args.max_nfev,
            n_restarts=args.n_restarts,
            rng=rng,
        )

        rms_phase_errors.append(rms_error)
        mean_abs_phase_errors.append(mean_abs_error)
        costs.append(result.cost)
        nfevs.append(result.nfev)
        success_flags.append(result.success)
        coeffs_all.append(coeffs_fit)

        print("RMS wrapped phase error [rad]:", rms_error)
        print("Mean absolute wrapped phase error [rad]:", mean_abs_error)
        print("Cost:", result.cost)
        print("nfev:", result.nfev)
        print("Success:", result.success)

        if index == 0:
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

    crop_label = (
        "full" if fit_crop_pixels is None else f"crop{2 * fit_crop_pixels}px"
    )
    mapping_label = args.fourier_mapping

    results_path = os.path.join(
        OUTDIR,
        f"WF_phase_farfieldLP_fit_results_{n_test}wfs_"
        f"{args.n_modes}modes_{crop_label}_{mapping_label}_{run_stamp}.npz",
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
        far_modes=far_modes,
        n_modes=args.n_modes,
        n_test=n_test,
        wavefront_path=wavefront_path,
        wavefront_key=wavefront_key,
        fit_crop_pixels=-1 if fit_crop_pixels is None else fit_crop_pixels,
        target_phase_units=TARGET_PHASE_UNITS,
        remove_target_piston=REMOVE_TARGET_PISTON,
        fourier_mapping_mode=fourier_metadata["mapping_mode"],
        target_fmax_x=(
            np.nan
            if fourier_metadata["target_fmax_x"] is None
            else fourier_metadata["target_fmax_x"]
        ),
        target_fmax_y=(
            np.nan
            if fourier_metadata["target_fmax_y"] is None
            else fourier_metadata["target_fmax_y"]
        ),
        microns_per_pixel=fourier_metadata["microns_per_pixel"],
        fibre_NA=fourier_metadata["NA"],
        fibre_V=fourier_metadata["V"],
        pad_factor=fourier_metadata["pad_factor"],
        max_r=fourier_metadata["max_r"],
        npix=fourier_metadata["npix"],
    )
    print("\nSaved numerical results to:", results_path)

    csv_path = os.path.join(
        OUTDIR,
        f"WF_phase_farfieldLP_fit_summary_{n_test}wfs_"
        f"{args.n_modes}modes_{crop_label}_{mapping_label}_{run_stamp}.csv",
    )
    summary = np.column_stack(
        [
            np.arange(n_test),
            rms_phase_errors,
            mean_abs_phase_errors,
            costs,
            nfevs,
            success_flags.astype(int),
        ]
    )
    np.savetxt(
        csv_path,
        summary,
        delimiter=",",
        header=(
            "sample,rms_phase_error_rad,mean_abs_phase_error_rad,"
            "cost,nfev,success"
        ),
        comments="",
    )
    print("Saved CSV summary to:", csv_path)

    print("\n==============================")
    print("Wavefront phase far-field LP fitting results")
    print("==============================")
    print("N modes:", args.n_modes)
    print("N wavefronts:", n_test)
    print("Fit crop:", crop_label)
    print("Fourier mapping:", args.fourier_mapping)
    print("Padding factor:", args.pad_factor)
    print("Mean RMS phase error [rad]:", np.mean(rms_phase_errors))
    print("Median RMS phase error [rad]:", np.median(rms_phase_errors))
    print("Mean absolute phase error [rad]:", np.mean(mean_abs_phase_errors))
    print("Success rate:", np.mean(success_flags))

    example_plot_path = os.path.join(
        OUTDIR,
        f"WF_phase_farfieldLP_fit_example_{args.n_modes}modes_"
        f"{crop_label}_{mapping_label}_{run_stamp}.png",
    )
    plot_phase_fit_example(
        target_phase=example_target,
        phase_fit=example_fit,
        phase_residual=example_residual,
        title=(
            f"Wavefront phase fit using {args.n_modes} far-field LP modes "
            f"({mapping_label})"
        ),
        outpath=example_plot_path,
        plot_crop_pixels=plot_crop_pixels,
    )
    print("Saved example phase plot to:", example_plot_path)

    coeff_plot_path = os.path.join(
        OUTDIR,
        f"WF_phase_farfieldLP_fit_coeffs_example_{args.n_modes}modes_"
        f"{crop_label}_{mapping_label}_{run_stamp}.png",
    )
    save_coeff_plot(
        coeffs=example_coeffs,
        labels=labels,
        lm_values=lm_values,
        outpath=coeff_plot_path,
    )
    print("Saved coefficient plot to:", coeff_plot_path)


if __name__ == "__main__":
    main()
