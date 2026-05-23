import numpy as np
import matplotlib.pyplot as plt

from lanternfiber import lanternfiber


# ============================================================
# SETTINGS
# ============================================================

# Replace these with your actual fibre values
N_CORE = 1.479
N_CLADDING = 1.444
CORE_RADIUS = 25.0      # microns
WAVELENGTH = 1.55      # microns

MAX_L = 20             # vertical axis
MAX_M = 8              # horizontal axis

NPIX = 256             # half-width of mode image
MAX_R = 2              # mode image extends to 2 core radii

DISPLAY = "real"       # "real", "abs", "intensity", or "phase"
FFT_CROP_FRAC = 0.45   # crop centre of Fourier plane

CELL_SIZE = 0.75
OUTFILE = "fourier_lp_mode_grid.png"


# ============================================================
# FOURIER HELPERS
# ============================================================

def fft_mode(E):
    """
    Fourier transform a centred LP mode image.

    ifftshift moves the centre of the image to index (0, 0),
    which is what numpy's FFT expects.
    """
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(E),
            norm="ortho"
        )
    )


def centre_crop(img, crop_frac=0.45):
    n = img.shape[0]
    c = n // 2
    half = int((n * crop_frac) / 2)

    return img[c - half:c + half, c - half:c + half]


def display_image(F, display="real"):
    if display == "real":
        img = np.real(F)
        vmax = np.max(np.abs(img))
        if vmax > 0:
            img = img / vmax
        return img, "seismic", -1, 1

    if display == "abs":
        img = np.abs(F)
        vmax = np.max(img)
        if vmax > 0:
            img = img / vmax
        return img, "inferno", 0, 1

    if display == "intensity":
        img = np.abs(F) ** 2
        vmax = np.max(img)
        if vmax > 0:
            img = img / vmax
        return img, "inferno", 0, 1

    if display == "phase":
        img = np.angle(F) / np.pi
        return img, "twilight", -1, 1

    raise ValueError("DISPLAY must be 'real', 'abs', 'intensity', or 'phase'")


# ============================================================
# MAKE LP MODES USING YOUR lanternfibre.py FILE
# ============================================================

fib = lanternfiber(
    n_core=N_CORE,
    n_cladding=N_CLADDING,
    core_radius=CORE_RADIUS,
    wavelength=WAVELENGTH
)

fib.find_fiber_modes(max_l=MAX_L + 1, verbose=True)

fib.make_fiber_modes(
    max_r=MAX_R,
    npix=NPIX,
    show_plots=False,
    normtosum=True
)

print(f"V-number = {fib.V:.3f}")


# Make lookup table:
# key = (l, m), value = index into fib.allmodefields_cos_cart
mode_lookup = {}

for idx, (l_val, m_val) in enumerate(zip(fib.allmodes_l, fib.allmodes_m)):
    mode_lookup[(int(l_val), int(m_val))] = idx


# ============================================================
# PLOT FOURIER-TRANSFORMED LP MODE GRID
# ============================================================

fig, ax = plt.subplots(figsize=(7, 13))

for l in range(0, MAX_L + 1):
    for m in range(1, MAX_M + 1):

        if (l, m) not in mode_lookup:
            continue

        mode_idx = mode_lookup[(l, m)]

        # This is the real-space LP mode from lanternfibre.py
        E = fib.allmodefields_cos_cart[mode_idx]

        # Convert signed real field into complex field
        E_complex = fib.make_complex_fld(E)

        # Fourier transform
        F = fft_mode(E_complex)

        # Crop around zero spatial frequency
        F_crop = centre_crop(F, FFT_CROP_FRAC)

        # Convert complex Fourier field into something visible
        img, cmap, vmin, vmax = display_image(F_crop, DISPLAY)

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

ax.set_xticks(np.arange(1, MAX_M + 1, 1))
ax.set_yticks(np.arange(0, MAX_L + 1, 1))

ax.set_xlabel("mode index m", fontweight="bold")
ax.set_ylabel("mode index l", fontweight="bold")

ax.grid(True, linestyle="--", alpha=0.45)
ax.set_axisbelow(True)

ax.set_title(
    f"Fourier-transformed LP modes, V = {fib.V:.2f}",
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(OUTFILE, dpi=300)
plt.show()

print(f"Saved image to {OUTFILE}")
