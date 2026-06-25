"""
Standalone script that imports lanternfiber.py, generates LP near-field modes,
then Fourier transforms them to get far-field modes.

Put this file in the same folder as lanternfiber.py, or add the folder containing
lanternfiber.py to sys.path below.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from lanternfiber import lanternfiber


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def near_to_far_field(near_field, pad_factor=1, normtosum=True):
    """
    Convert a near-field complex mode into a far-field complex mode using FFT.

    The near-field mode from lanternfiber is centred in the middle of the image.
    For the FFT convention, we first move that centre to array index (0, 0)
    using ifftshift(). After fft2(), we shift the far-field zero-frequency
    component back to the middle using fftshift() for plotting.
    """

    near_field = np.asarray(near_field, dtype=np.complex128)

    # Optional zero-padding. This does not add real information, but makes the
    # far-field image look smoother / more finely sampled.
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
            constant_values=0
        )

    # Important FFT shift order:
    # 1. Move centred near-field mode to array origin.
    # 2. Fourier transform.
    # 3. Move far-field zero spatial frequency back to the centre for display.
    far_field = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(near_field)
        )
    )

    if normtosum:
        power = np.sum(np.abs(far_field) ** 2)
        if power > 0:
            far_field = far_field / np.sqrt(power)

    return far_field


def make_all_far_field_modes(lf, pad_factor=1, normtosum=True):
    """
    Fourier transform all modes stored in lf.allmodefields_rsoftorder.

    Returns
    -------
    far_fields : list of 2D complex arrays
        Complex far-field mode for each LP mode.
    """

    if lf.allmodefields_rsoftorder is None:
        raise RuntimeError(
            "No near-field modes found. Run lf.find_fiber_modes() and "
            "lf.make_fiber_modes() before calling this."
        )

    far_fields = []

    for raw_mode in lf.allmodefields_rsoftorder:
        # raw_mode is the red/blue signed amplitude mode.
        # lanternfiber.make_complex_fld() turns negative values into phase pi.
        near_field = lf.make_complex_fld(raw_mode)

        far_field = near_to_far_field(
            near_field,
            pad_factor=pad_factor,
            normtosum=normtosum
        )

        far_fields.append(far_field)

    return far_fields


def plot_far_field_mode(far_field, title="", outpath=None, log_intensity=True):
    """
    Plot far-field intensity and phase.
    """

    intensity = np.abs(far_field) ** 2
    phase = np.angle(far_field)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    if log_intensity:
        # Small floor avoids log10(0)
        im = np.log10(intensity / np.max(intensity) + 1e-12)
        plt.imshow(im, cmap="inferno")
        plt.title("Far-field log intensity")
    else:
        plt.imshow(intensity, cmap="inferno")
        plt.title("Far-field intensity")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.title("Far-field phase")
    plt.colorbar()

    plt.suptitle(title)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


# -------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------

def main():
    # Change this to the folder where you want the far-field plots saved.
    outdir = "./far_field_modes"
    os.makedirs(outdir, exist_ok=True)

    # Use the same fibre parameters that you used for your near-field modes.
    # These are example values only, so replace them with your actual values.
    lf = lanternfiber(
        n_core=1.45,
        n_cladding=1.44,
        core_radius=25,      # microns
        wavelength=1.55,     # microns
        nmodes=19
    )

    # Make the near-field LP modes.
    lf.find_fiber_modes()
    lf.make_fiber_modes(
        npix=200,
        max_r=2,
        show_plots=False,
        normtosum=True
    )

    # Fourier transform all near-field modes into far-field modes.
    far_fields = make_all_far_field_modes(
        lf,
        pad_factor=2,
        normtosum=True
    )

    # Save all far-field plots.
    for mode_num, far_field in enumerate(far_fields):
        if mode_num < len(lf.modelabels):
            label = lf.modelabels[mode_num]
        else:
            label = f"mode_{mode_num}"

        outfile = os.path.join(outdir, f"far_field_{mode_num:02d}_{label}.png")

        plot_far_field_mode(
            far_field,
            title=f"{label} far field",
            outpath=outfile,
            log_intensity=True
        )

        print(f"Saved {outfile}")

    # Optional: save the complex far-field arrays too.
    np.savez(
        os.path.join(outdir, "far_field_modes_complex.npz"),
        far_fields=np.asarray(far_fields),
        modelabels=np.asarray(lf.modelabels)
    )

    print("Done.")


if __name__ == "__main__":
    main()
