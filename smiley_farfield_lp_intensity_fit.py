"""
Standalone demonstration: fit Fourier-transformed LP-mode intensities to a
smiley-face target using nonlinear least squares.

This keeps the Fourier-domain LP-mode generation and physical frequency mapping
from the current wavefront-fitting code, but replaces measured wavefront data
with an image target.

The fitted field is

    E_fit(x, y) = sum_m c_m E_m(x, y)

and the fitted intensity is

    I_fit(x, y) = |E_fit(x, y)|^2.

scipy.optimize.least_squares adjusts the complex modal coefficients so that
I_fit best matches the smiley target intensity.
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

# Change this to your folder if needed.
#DATADIR = "/home/manav/PL-NN-testdata_forDec2025/"
DATADIR = '/Users/manavkalra/Downloads/PL-NN-testdata_forDec2025/'
OUTDIR = DATADIR

# Save the smiley image as this file, or change the path here.
SMILEY_PATH = os.path.join(DATADIR, "smiley.png")

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

# Keep these consistent with the current Fourier-transformed mode code.
N_MODES = 15
NPIX = 256
MAX_R = 3
PAD_FACTOR = 1

# Nonlinear intensity fitting.
MAX_NFEV = 1000
N_RESTARTS = 3
RNG_SEED = 42

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


def maximum_edge_amplitude_ratio(field: np.ndarray) -> float:
    """Maximum boundary amplitude divided by maximum field amplitude."""
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
# Load smiley target
# -------------------------------------------------------------------------

def load_smiley_target(
    image_path: str,
    target_size: int = TARGET_SIZE,
    invert: bool = INVERT_SMILEY,
) -> np.ndarray:
    """
    Load the smiley image and turn it into a normalised target intensity.

    For the supplied black-on-white smiley:
        black strokes -> intensity 1
        white background -> intensity 0
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Could not find smiley image:\n{image_path}\n\n"
            "Save the supplied smiley as smiley.png in DATADIR, "
            "or change SMILEY_PATH."
        )

    image = plt.imread(image_path)

    # Convert RGB/RGBA to greyscale.
    if image.ndim == 3:
        rgb = image[..., :3]
        image = (
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        )

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

    return image


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
    Fourier transform a centred near-field mode and return physical
    spatial-frequency axes.
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

    # Same centred Fourier-transform convention as the current code.
    far_field = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(near_field)
        )
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
                "FOURIER_MAPPING_MODE='custom' requires TARGET_FMAX_X "
                "and TARGET_FMAX_Y."
            )

        if target_fmax_x <= 0 or target_fmax_y <= 0:
            raise ValueError(
                "TARGET_FMAX_X and TARGET_FMAX_Y must be positive."
            )

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
    Interpolate a complex FFT onto the specified target-frequency grid.
    """
    target_ny, target_nx = (
        int(target_shape[0]),
        int(target_shape[1]),
    )

    target_dfx = 2.0 * target_fmax_x / target_nx
    target_dfy = 2.0 * target_fmax_y / target_ny

    fx_target = (
        np.arange(target_nx) - target_nx // 2
    ) * target_dfx

    fy_target = (
        np.arange(target_ny) - target_ny // 2
    ) * target_dfy

    source_min_x = float(fx_source[0])
    source_max_x = float(fx_source[-1])
    source_min_y = float(fy_source[0])
    source_max_y = float(fy_source[-1])

    if fx_target[0] < source_min_x or fx_target[-1] > source_max_x:
        raise ValueError(
            "Target x-frequency range exceeds the available FFT range. "
            "Increase NPIX or decrease TARGET_FMAX_X."
        )

    if fy_target[0] < source_min_y or fy_target[-1] > source_max_y:
        raise ValueError(
            "Target y-frequency range exceeds the available FFT range. "
            "Increase NPIX or decrease TARGET_FMAX_Y."
        )

    target_fy_grid, target_fx_grid = np.meshgrid(
        fy_target,
        fx_target,
        indexing="ij",
    )

    points = np.column_stack(
        [
            target_fy_grid.ravel(),
            target_fx_grid.ravel(),
        ]
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

    return (
        real_target + 1j * imag_target,
        fy_target,
        fx_target,
    )


def recommended_padding_factor(
    near_field_shape: Tuple[int, int],
    microns_per_pixel: float,
    target_shape: Tuple[int, int],
    target_fmax_x: float,
    target_fmax_y: float,
) -> int:
    """Diagnostic padding estimate."""
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
    """Print physical near-field and FFT sampling information."""
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
    print(
        "FFT x range [cycles/um]:",
        (float(fx[0]), float(fx[-1])),
    )
    print(
        "FFT y range [cycles/um]:",
        (float(fy[0]), float(fy[-1])),
    )
    print("Fibre NA:", float(lf.NA))
    print(
        "Fibre NA/lambda [cycles/um]:",
        float(lf.NA / lf.wavelength),
    )

    if target_fmax_x is not None and target_fmax_y is not None:
        print(
            "Mapped target x range [cycles/um]:",
            (-target_fmax_x, target_fmax_x),
        )
        print(
            "Mapped target y range [cycles/um]:",
            (-target_fmax_y, target_fmax_y),
        )

        rec_pad = recommended_padding_factor(
            near_field_shape=near_field_shape,
            microns_per_pixel=pixel_scale,
            target_shape=target_shape,
            target_fmax_x=target_fmax_x,
            target_fmax_y=target_fmax_y,
        )

        print(
            "Diagnostic padding for one FFT sample per target pixel:",
            rec_pad,
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
        extent=[
            fx[0],
            fx[-1],
            fy[0],
            fy[-1],
        ],
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
    """Construct lanternfiber object and generate supported LP modes."""
    lf = lanternfiber(
        N_CORE,
        N_CLADDING,
        CORE_RADIUS,
        WAVELENGTH,
    )

    lf.find_fiber_modes()

    lf.make_fiber_modes(
        npix=npix,
        max_r=max_r,
        show_plots=False,
        normtosum=True,
    )

    return lf


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
    """
    Generate mapped complex far-field LP modes and construct fitting matrix.

    This is the same basic Fourier-mode pipeline as the current wavefront code.
    """
    lf = build_lanternfiber(
        npix=NPIX,
        max_r=MAX_R,
    )

    near_fields_raw = np.asarray(
        lf.allmodefields_rsoftorder
    )

    n_available = near_fields_raw.shape[0]

    print("\nTotal available LP scalar modes:", n_available)

    if n_modes > n_available:
        raise ValueError(
            f"n_modes={n_modes}, but only {n_available} modes are available."
        )

    resolved_fmax_x, resolved_fmax_y = (
        resolve_target_frequency_range(
            lf=lf,
            mapping_mode=mapping_mode,
            target_fmax_x=target_fmax_x,
            target_fmax_y=target_fmax_y,
        )
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

        edge_ratio = maximum_edge_amplitude_ratio(
            near_field
        )

        if edge_ratio > EDGE_WARNING_RATIO:
            print(
                f"Warning: {label} edge amplitude ratio is "
                f"{edge_ratio:.3e}. Consider increasing MAX_R."
            )

        far_field_full, fy_source, fx_source = (
            near_to_far_field(
                near_field=near_field,
                microns_per_pixel=lf.microns_per_pixel,
                pad_factor=pad_factor,
                normtosum=True,
            )
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

        if (
            diagnostic_outpath is not None
            and mode_num == DIAGNOSTIC_MODE_NUM
        ):
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

            far_field, _, _ = (
                resample_complex_fourier_field(
                    far_field=far_field_full,
                    fy_source=fy_source,
                    fx_source=fx_source,
                    target_shape=target_shape,
                    target_fmax_y=resolved_fmax_y,
                    target_fmax_x=resolved_fmax_x,
                )
            )

        far_field = centre_crop(
            far_field,
            centre_crop_pixels,
        )

        far_field = normalise_power(far_field)

        far_modes.append(far_field)
        labels.append(label)

        if (
            hasattr(lf, "lp_mode_list")
            and mode_num < len(lf.lp_mode_list)
        ):
            lm_values.append(
                lf.lp_mode_list[mode_num]
            )
        else:
            lm_values.append(
                [np.nan, np.nan]
            )

    far_modes = np.asarray(
        far_modes,
        dtype=np.complex128,
    )

    n_modes_actual, ny, nx = far_modes.shape

    # Each column is one complex far-field LP mode.
    mode_matrix = far_modes.reshape(
        n_modes_actual,
        ny * nx,
    ).T

    norms = np.sqrt(
        np.sum(
            np.abs(mode_matrix) ** 2,
            axis=0,
            keepdims=True,
        )
    )

    mode_matrix = mode_matrix / (
        norms + FIELD_EPS
    )

    print("Fourier mapping mode:", mapping_mode)
    print(
        "Far-field LP mode shape used for fit:",
        (ny, nx),
    )
    print(
        "Mode matrix shape:",
        mode_matrix.shape,
    )

    metadata = {
        "mapping_mode": mapping_mode,
        "target_fmax_x": resolved_fmax_x,
        "target_fmax_y": resolved_fmax_y,
        "microns_per_pixel": float(
            lf.microns_per_pixel
        ),
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
# Nonlinear least-squares INTENSITY fitting
# -------------------------------------------------------------------------

def unpack_coeffs(
    z: np.ndarray,
    n_modes: int,
) -> np.ndarray:
    """
    Convert the optimiser's real vector into complex modal coefficients.

    The coefficient vector is normalised because the fitted image is also
    normalised by its peak intensity. The overall coefficient scale therefore
    carries no useful information in this demonstration.
    """
    coeffs = (
        z[:n_modes]
        + 1j * z[n_modes:]
    )

    norm = np.sqrt(
        np.sum(np.abs(coeffs) ** 2)
    )

    if norm > 0:
        coeffs = coeffs / norm

    return coeffs


def fit_coeffs_to_target_intensity(
    mode_matrix: np.ndarray,
    target_intensity: np.ndarray,
    max_nfev: int = MAX_NFEV,
    n_restarts: int = N_RESTARTS,
    rng: Optional[np.random.Generator] = None,
):
    """
    Find complex LP coefficients that best reproduce the target INTENSITY.

    IMPORTANT:
    This is nonlinear because

        I_fit = |M c|^2

    so np.linalg.lstsq cannot be used directly for intensity-only fitting.
    scipy.optimize.least_squares is used instead.
    """
    if rng is None:
        rng = np.random.default_rng()

    M = np.asarray(
        mode_matrix,
        dtype=np.complex128,
    )

    target = np.asarray(
        target_intensity,
        dtype=float,
    )

    n_modes = M.shape[1]

    if M.shape[0] != target.size:
        raise ValueError(
            f"Mode matrix has {M.shape[0]} pixels, "
            f"but target has {target.size}."
        )

    target = target - np.min(target)
    target = target / (
        np.max(target) + FIELD_EPS
    )

    target_flat = target.reshape(-1)

    def residual(z: np.ndarray) -> np.ndarray:
        coeffs = unpack_coeffs(
            z,
            n_modes,
        )

        # Coherently add complex LP fields FIRST.
        field_flat = M @ coeffs

        # Then convert the combined field to intensity.
        intensity_flat = np.abs(
            field_flat
        ) ** 2

        # Compare shapes rather than arbitrary overall power.
        intensity_flat = intensity_flat / (
            np.max(intensity_flat)
            + FIELD_EPS
        )

        return (
            intensity_flat
            - target_flat
        )

    best_result = None

    for restart in range(n_restarts):

        print(
            f"  least-squares restart "
            f"{restart + 1}/{n_restarts}"
        )

        if restart == 0:
            # First attempt starts from LP01.
            z0 = np.zeros(
                2 * n_modes,
                dtype=float,
            )
            z0[0] = 1.0

        else:
            # Other attempts use random complex modal mixtures.
            z0 = rng.normal(
                0.0,
                1.0,
                size=2 * n_modes,
            )

        result = least_squares(
            residual,
            z0,
            max_nfev=max_nfev,
            verbose=0,
        )

        print(
            "    cost =",
            result.cost,
            "| nfev =",
            result.nfev,
            "| success =",
            result.success,
        )

        if (
            best_result is None
            or result.cost < best_result.cost
        ):
            best_result = result

    coeffs_fit = unpack_coeffs(
        best_result.x,
        n_modes,
    )

    field_fit = (
        M @ coeffs_fit
    ).reshape(target.shape)

    intensity_fit = (
        np.abs(field_fit) ** 2
    )

    intensity_fit = intensity_fit / (
        np.max(intensity_fit)
        + FIELD_EPS
    )

    intensity_residual = (
        intensity_fit
        - target
    )

    rms_intensity_error = np.sqrt(
        np.mean(
            intensity_residual ** 2
        )
    )

    mean_abs_intensity_error = np.mean(
        np.abs(intensity_residual)
    )

    return (
        coeffs_fit,
        field_fit,
        intensity_fit,
        intensity_residual,
        rms_intensity_error,
        mean_abs_intensity_error,
        best_result,
    )


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def save_fit_plot(
    target_intensity: np.ndarray,
    intensity_fit: np.ndarray,
    intensity_residual: np.ndarray,
    rms_error: float,
    n_modes: int,
    outpath: str,
) -> None:
    """Save target, LP fit, and residual."""
    vmax_residual = np.max(
        np.abs(intensity_residual)
    )

    if vmax_residual == 0:
        vmax_residual = 1.0

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(
        target_intensity,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    plt.title("Target smiley intensity")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(
        intensity_fit,
        cmap="gray",
        origin="lower",
        vmin=0,
        vmax=1,
    )
    plt.title(
        f"Fitted intensity\n"
        f"{n_modes} far-field LP modes"
    )
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(
        intensity_residual,
        cmap="bwr",
        origin="lower",
        vmin=-vmax_residual,
        vmax=vmax_residual,
    )
    plt.title(
        f"Fit - target residual\n"
        f"RMS = {rms_error:.4f}"
    )
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_coeff_plot(
    coeffs: np.ndarray,
    labels: list[str],
    lm_values: np.ndarray,
    outpath: str,
) -> None:
    """Save amplitude and phase of fitted complex LP coefficients."""
    amplitudes = np.abs(coeffs)
    phases = np.angle(coeffs)
    x = np.arange(len(coeffs))

    tick_labels = []

    for index, label in enumerate(labels):
        l_val, m_val = lm_values[index]

        if np.isfinite(l_val) and np.isfinite(m_val):
            tick_labels.append(
                f"{label}\n"
                f"(l={int(l_val)},m={int(m_val)})"
            )
        else:
            tick_labels.append(label)

    plt.figure(
        figsize=(
            max(10, len(coeffs) * 0.7),
            6,
        )
    )

    plt.subplot(2, 1, 1)
    plt.bar(x, amplitudes)
    plt.ylabel("|coefficient|")
    plt.xticks(
        x,
        tick_labels,
        rotation=90,
        fontsize=8,
    )

    plt.subplot(2, 1, 2)
    plt.bar(x, phases)
    plt.ylabel("phase [rad]")
    plt.xticks(
        x,
        tick_labels,
        rotation=90,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(
        outpath,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smiley",
        type=str,
        default=SMILEY_PATH,
        help="Path to smiley image.",
    )

    parser.add_argument(
        "--target-size",
        type=int,
        default=TARGET_SIZE,
        help="Square image size used for fitting.",
    )

    parser.add_argument(
        "--n-modes",
        type=int,
        default=N_MODES,
    )

    parser.add_argument(
        "--max-nfev",
        type=int,
        default=MAX_NFEV,
    )

    parser.add_argument(
        "--n-restarts",
        type=int,
        default=N_RESTARTS,
    )

    parser.add_argument(
        "--pad-factor",
        type=int,
        default=PAD_FACTOR,
    )

    parser.add_argument(
        "--fourier-mapping",
        choices=[
            "full_fft",
            "fibre_na",
            "custom",
        ],
        default=FOURIER_MAPPING_MODE,
    )

    parser.add_argument(
        "--target-fmax-x",
        type=float,
        default=TARGET_FMAX_X,
    )

    parser.add_argument(
        "--target-fmax-y",
        type=float,
        default=TARGET_FMAX_Y,
    )

    parser.add_argument(
        "--no-fourier-diagnostic",
        action="store_true",
    )

    args = parser.parse_args()

    if args.n_modes <= 0:
        raise ValueError(
            "--n-modes must be positive."
        )

    if args.target_size <= 0:
        raise ValueError(
            "--target-size must be positive."
        )

    if args.max_nfev <= 0:
        raise ValueError(
            "--max-nfev must be positive."
        )

    if args.n_restarts <= 0:
        raise ValueError(
            "--n-restarts must be positive."
        )

    if args.pad_factor < 1:
        raise ValueError(
            "--pad-factor must be at least 1."
        )

    os.makedirs(
        OUTDIR,
        exist_ok=True,
    )

    run_stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    rng = np.random.default_rng(
        RNG_SEED
    )

    # -----------------------------------------------------
    # Smiley target
    # -----------------------------------------------------

    target_intensity = load_smiley_target(
        image_path=args.smiley,
        target_size=args.target_size,
        invert=INVERT_SMILEY,
    )

    target_shape = target_intensity.shape

    print(
        "\nTarget smiley shape:",
        target_shape,
    )

    # -----------------------------------------------------
    # Current Fourier LP-mode pipeline
    # -----------------------------------------------------

    diagnostic_path = None

    if (
        SAVE_FOURIER_DIAGNOSTIC
        and not args.no_fourier_diagnostic
    ):
        diagnostic_path = os.path.join(
            OUTDIR,
            f"smiley_fullFFT_diagnostic_mode"
            f"{DIAGNOSTIC_MODE_NUM}_"
            f"pad{args.pad_factor}_"
            f"{run_stamp}.png",
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
        centre_crop_pixels=None,
        pad_factor=args.pad_factor,
        mapping_mode=args.fourier_mapping,
        target_fmax_x=args.target_fmax_x,
        target_fmax_y=args.target_fmax_y,
        diagnostic_outpath=diagnostic_path,
    )

    # -----------------------------------------------------
    # Nonlinear least-squares intensity fit
    # -----------------------------------------------------

    print(
        "\nFitting LP-mode intensity to smiley..."
    )

    (
        coeffs_fit,
        field_fit,
        intensity_fit,
        intensity_residual,
        rms_error,
        mae_error,
        result,
    ) = fit_coeffs_to_target_intensity(
        mode_matrix=mode_matrix,
        target_intensity=target_intensity,
        max_nfev=args.max_nfev,
        n_restarts=args.n_restarts,
        rng=rng,
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\n==============================")
    print("Smiley far-field LP fit")
    print("==============================")
    print("N modes:", args.n_modes)
    print(
        "Target size:",
        target_shape,
    )
    print(
        "Fourier mapping:",
        args.fourier_mapping,
    )
    print(
        "Target fmax x:",
        fourier_metadata["target_fmax_x"],
    )
    print(
        "Target fmax y:",
        fourier_metadata["target_fmax_y"],
    )
    print(
        "Padding factor:",
        args.pad_factor,
    )
    print(
        "RMS intensity error:",
        rms_error,
    )
    print(
        "Mean absolute intensity error:",
        mae_error,
    )
    print(
        "Least-squares cost:",
        result.cost,
    )
    print(
        "nfev:",
        result.nfev,
    )
    print(
        "Success:",
        result.success,
    )

    print("\nFitted coefficients")

    for i, coeff in enumerate(
        coeffs_fit
    ):
        print(
            f"{i:02d} "
            f"{labels[i]:15s} "
            f"|c|={np.abs(coeff):.6f} "
            f"phase={np.angle(coeff):+.6f} rad"
        )

    # -----------------------------------------------------
    # Save numerical result
    # -----------------------------------------------------

    results_path = os.path.join(
        OUTDIR,
        f"smiley_farfieldLP_fit_"
        f"{args.n_modes}modes_"
        f"{args.target_size}px_"
        f"{args.fourier_mapping}_"
        f"{run_stamp}.npz",
    )

    np.savez_compressed(
        results_path,
        target_intensity=target_intensity,
        field_fit=field_fit,
        intensity_fit=intensity_fit,
        intensity_residual=intensity_residual,
        coeffs_fit=coeffs_fit,
        labels=np.asarray(labels),
        lm_values=lm_values,
        far_modes=far_modes,
        rms_intensity_error=rms_error,
        mean_abs_intensity_error=mae_error,
        least_squares_cost=result.cost,
        least_squares_nfev=result.nfev,
        least_squares_success=result.success,
        fourier_mapping_mode=fourier_metadata[
            "mapping_mode"
        ],
        target_fmax_x=fourier_metadata[
            "target_fmax_x"
        ],
        target_fmax_y=fourier_metadata[
            "target_fmax_y"
        ],
        microns_per_pixel=fourier_metadata[
            "microns_per_pixel"
        ],
        fibre_NA=fourier_metadata[
            "NA"
        ],
        fibre_V=fourier_metadata[
            "V"
        ],
        pad_factor=fourier_metadata[
            "pad_factor"
        ],
        max_r=fourier_metadata[
            "max_r"
        ],
        npix=fourier_metadata[
            "npix"
        ],
    )

    print(
        "\nSaved numerical result to:",
        results_path,
    )

    # -----------------------------------------------------
    # Save plots
    # -----------------------------------------------------

    fit_plot_path = os.path.join(
        OUTDIR,
        f"smiley_farfieldLP_fit_"
        f"{args.n_modes}modes_"
        f"{args.target_size}px_"
        f"{run_stamp}.png",
    )

    save_fit_plot(
        target_intensity=target_intensity,
        intensity_fit=intensity_fit,
        intensity_residual=intensity_residual,
        rms_error=rms_error,
        n_modes=args.n_modes,
        outpath=fit_plot_path,
    )

    print(
        "Saved fit plot to:",
        fit_plot_path,
    )

    coeff_plot_path = os.path.join(
        OUTDIR,
        f"smiley_farfieldLP_coeffs_"
        f"{args.n_modes}modes_"
        f"{run_stamp}.png",
    )

    save_coeff_plot(
        coeffs=coeffs_fit,
        labels=labels,
        lm_values=lm_values,
        outpath=coeff_plot_path,
    )

    print(
        "Saved coefficient plot to:",
        coeff_plot_path,
    )

    if diagnostic_path is not None:
        print(
            "Saved Fourier diagnostic to:",
            diagnostic_path,
        )


if __name__ == "__main__":
    main()
