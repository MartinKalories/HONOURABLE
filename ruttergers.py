"""
Fourier transform and display LP modes from example_fiber_modes.py style code.
"""

from lanternfiber import lanternfiber
import numpy as np
import matplotlib.pyplot as plt
import os

datadir = "/home/manav//PL-NN-testdata_forDec2025/"
outdir = datadir
os.makedirs(outdir, exist_ok=True)
# ============================================================
# Fibre parameters from example_fiber_modes.py
# ============================================================

n_core = 1.44
n_cladding = 1.4345
wavelength = 1.55       # microns
core_radius = 32.8 / 2  # microns

show_plots = False

# Plot settings
MAX_L = 20
MAX_M = 8

NPIX = 256
MAX_R = 2

CELL_SIZE = 0.75
FFT_CROP_FRAC = 0.45

DISPLAY = "real"
# Options:
# "real"      -> red/blue Fourier field, closest to your example image
# "abs"       -> Fourier amplitude
# "intensity" -> Fourier intensity
# "phase"     -> Fourier phase

OUTFILE = "fourier_lp_modes_grid.png"


# ============================================================
# Fourier helper functions
# ============================================================

def fft_centred_mode(E):
    """
    Fourier transform a mode field that is centred in the image.

    ifftshift moves the image centre to array index (0, 0),
    which is what np.fft.fft2 expects.

    fftshift moves the zero-frequency component back to the centre
    for plotting.
    """
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(E),
            norm="ortho"
        )
    )


def centre_crop(img, crop_frac):
    """
    Crop around the centre of the Fourier plane.
    """
    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2

    half_y = int((ny * crop_frac) / 2)
    half_x = int((nx * crop_frac) / 2)

    return img[
        cy - half_y:cy + half_y,
        cx - half_x:cx + half_x
    ]


def make_display_image(F, display):
    """
    Convert complex Fourier field into a plottable image.
    """
    if display == "real":
        img = np.real(F)
        vmax = np.max(np.abs(img))
        if vmax > 0:
            img = img / vmax
        return img, "bwr", -1, 1

    elif display == "abs":
        img = np.abs(F)
        vmax = np.max(img)
        if vmax > 0:
            img = img / vmax
        return img, "inferno", 0, 1

    elif display == "intensity":
        img = np.abs(F) ** 2
        vmax = np.max(img)
        if vmax > 0:
            img = img / vmax
        return img, "inferno", 0, 1

    elif display == "phase":
        img = np.angle(F)
        return img, "twilight", -np.pi, np.pi

    else:
        raise ValueError("DISPLAY must be: real, abs, intensity, or phase")


# ============================================================
# Generate modes using the same structure as example_fiber_modes.py
# ============================================================

f = lanternfiber(n_core, n_cladding, core_radius, wavelength)

f.find_fiber_modes()
f.make_fiber_modes(
    max_r=MAX_R,
    npix=NPIX,
    show_plots=show_plots,
    plot_pausetime=0.5
)

print("V-number:", f.V)
print("Total number of scalar modes:", f.nmodes)

# This is useful for checking what was generated
print("Supported LP mode groups:")
for l_val, m_val in zip(f.allmodes_l, f.allmodes_m):
    print(f"  LP{l_val}{m_val}")


# ============================================================
# Build lookup table for LP(l, m) -> mode index
# ============================================================

mode_lookup = {}

for idx, (l_val, m_val) in enumerate(zip(f.allmodes_l, f.allmodes_m)):
    mode_lookup[(int(l_val), int(m_val))] = idx


# ============================================================
# Plot Fourier-transformed LP modes
# ============================================================

fig, ax = plt.subplots(figsize=(7, 13))

for l in range(0, MAX_L + 1):
    for m in range(1, MAX_M + 1):

        if (l, m) not in mode_lookup:
            continue

        mode_idx = mode_lookup[(l, m)]

        # Use cos-oriented Cartesian field from example_fiber_modes.py
        E = f.allmodefields_cos_cart[mode_idx]

        # The LP modes are real signed amplitudes.
        # Negative values represent pi phase, so keeping the sign is important.
        E_complex = E.astype(np.complex128)

        # Fourier transform
        F = fft_centred_mode(E_complex)

        # Crop central Fourier region for clearer plotting
        F_crop = centre_crop(F, FFT_CROP_FRAC)

        # Convert complex Fourier result into visible image
        img, cmap, vmin, vmax = make_display_image(F_crop, DISPLAY)

        x0 = m - CELL_SIZE / 2
        x1 = m + CELL_SIZE / 2
        y0 = l - CELL_SIZE / 2
        y1 = l + CELL_SIZE / 2

        ax.imshow(
            img,
            extent=(x0, x1, y0, y1),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            zorder=3
        )

ax.set_xlim(0.5, MAX_M + 0.5)
ax.set_ylim(-0.5, MAX_L + 0.5)

ax.set_xticks(np.arange(1, MAX_M + 1))
ax.set_yticks(np.arange(0, MAX_L + 1))

ax.set_xlabel("mode index m", fontweight="bold")
ax.set_ylabel("mode index l", fontweight="bold")

ax.grid(True, linestyle="--", alpha=0.45)
ax.set_axisbelow(True)

ax.set_title(
    f"Fourier-transformed LP modes, V = {f.V:.2f}",
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(OUTFILE, dpi=300)
plt.show()
phase_plot_save_path = os.path.join(
    outdir,
    f"ruttegers attempt.png"
)

print(f"Saved figure to: {OUTFILE}")
