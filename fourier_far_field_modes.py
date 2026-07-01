"""
fourier_far_field_modes.py

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


def plot_far_field_mode(
    far_field,
    title="",
    outpath=None,
    log_intensity=True,
    zoom_pixels=25
):
    """
    Plot zoomed far-field intensity and zoomed far-field phase.
    """

    intensity = np.abs(far_field) ** 2
    phase = np.angle(far_field)

    # Centre pixel of the image
    cy, cx = np.array(intensity.shape) // 2

    # Crop intensity and phase around the centre
    intensity_zoom = intensity[
        cy - zoom_pixels: cy + zoom_pixels,
        cx - zoom_pixels: cx + zoom_pixels
    ]

    phase_zoom = phase[
        cy - zoom_pixels: cy + zoom_pixels,
        cx - zoom_pixels: cx + zoom_pixels
    ]

    plt.figure(figsize=(10, 4))

    # -------------------------------------------------
    # Zoomed intensity plot
    # -------------------------------------------------
    plt.subplot(1, 2, 1)

    if log_intensity:
        im = np.log10(intensity_zoom / np.max(intensity_zoom) + 1e-12)
        plt.imshow(im, cmap="inferno")
        plt.title("Far-field log intensity, centre zoom")
    else:
        plt.imshow(intensity_zoom, cmap="inferno")
        plt.title("Far-field intensity, centre zoom")

    plt.colorbar()

    # -------------------------------------------------
    # Zoomed phase plot
    # -------------------------------------------------
    plt.subplot(1, 2, 2)

    plt.imshow(
        phase_zoom,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi
    )

    plt.title("Far-field phase, centre zoom")
    plt.colorbar()

    plt.suptitle(title)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_all_far_field_modes_grid(
    far_fields,
    labels=None,
    lm_values=None,
    outpath=None,
    log_intensity=False,
    zoom_pixels=None,
    modes_per_row=5
):
    """
    Plot all far-field modes in a wrapped grid.

    For each block:
        top row    = intensity
        bottom row = phase

    modes_per_row controls how many modes appear before wrapping.
    """

    n_modes = len(far_fields)

    n_blocks = int(np.ceil(n_modes / modes_per_row))
    n_rows = 2 * n_blocks
    n_cols = modes_per_row

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 5 * n_blocks),
        squeeze=False
    )

    im_top = None
    im_bot = None

    for i, far_field in enumerate(far_fields):
        block = i // modes_per_row
        col = i % modes_per_row

        intensity_row = 2 * block
        phase_row = 2 * block + 1

        intensity = np.abs(far_field) ** 2
        phase = np.angle(far_field)

        if zoom_pixels is not None:
            cy, cx = np.array(intensity.shape) // 2

            intensity = intensity[
                cy - zoom_pixels: cy + zoom_pixels,
                cx - zoom_pixels: cx + zoom_pixels
            ]

            phase = phase[
                cy - zoom_pixels: cy + zoom_pixels,
                cx - zoom_pixels: cx + zoom_pixels
            ]

        # Make title label
        if labels is not None and i < len(labels):
            mode_label = labels[i]
        else:
            mode_label = f"Mode {i}"

        if lm_values is not None and i < len(lm_values):
            l_val, m_val = lm_values[i]
            title = f"{mode_label}\nl={l_val}, m={m_val}"
        else:
            title = mode_label

        # -------------------------
        # Intensity plot
        # -------------------------
        ax_top = axes[intensity_row, col]

        if log_intensity:
            im_top = ax_top.imshow(
                np.log10(intensity / np.max(intensity) + 1e-12),
                cmap="inferno"
            )
        else:
            im_top = ax_top.imshow(intensity, cmap="inferno")

        ax_top.set_title(title, fontsize=10)
        ax_top.set_xticks([])
        ax_top.set_yticks([])

        if col == 0:
            ax_top.set_ylabel("Intensity", fontsize=12)

        # -------------------------
        # Phase plot
        # -------------------------
        ax_bot = axes[phase_row, col]

        im_bot = ax_bot.imshow(
            phase,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi
        )

        ax_bot.set_xticks([])
        ax_bot.set_yticks([])

        if col == 0:
            ax_bot.set_ylabel("Phase", fontsize=12)

    # Hide unused empty axes at the end
    for j in range(n_modes, n_blocks * modes_per_row):
        block = j // modes_per_row
        col = j % modes_per_row

        axes[2 * block, col].axis("off")
        axes[2 * block + 1, col].axis("off")

    # Colourbars
    if im_top is not None:
        cbar1 = fig.colorbar(im_top, ax=axes[0::2, :], shrink=0.8)

        if log_intensity:
            cbar1.set_label("log10 relative intensity")
        else:
            cbar1.set_label("intensity")

    if im_bot is not None:
        cbar2 = fig.colorbar(im_bot, ax=axes[1::2, :], shrink=0.8)
        cbar2.set_label("phase [rad]")

    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# -------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------

def main():
    # Change this to the folder where you want the far-field plots saved.
    outdir = "/home/manav//PL-NN-testdata_forDec2025/far_field_modes"
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

    # Build l,m list for titles
    lm_values = []
    for mode_num in range(len(far_fields)):
        if hasattr(lf, "lp_mode_list") and mode_num < len(lf.lp_mode_list):
            lm_values.append(lf.lp_mode_list[mode_num])
        else:
            lm_values.append((None, None))

    # Save all modes in one big figure
    all_modes_plot_path = os.path.join(outdir, "all_far_field_modes_grid.png")

    plot_all_far_field_modes_grid(
        far_fields=far_fields,
        labels=lf.modelabels,
        lm_values=lm_values,
        outpath=all_modes_plot_path,
        log_intensity=False,   # change to True if you want log intensity
        zoom_pixels=25, # or None for full image
        modes_per_row=10
    )

    print(f"Saved combined grid to {all_modes_plot_path}")

    # Save all individual far-field plots.
    for mode_num, far_field in enumerate(far_fields):
        if mode_num < len(lf.modelabels):
            label = lf.modelabels[mode_num]
        else:
            label = f"mode_{mode_num}"

        # Include the LP l,m values in the title and filename when available.
        if hasattr(lf, "lp_mode_list") and mode_num < len(lf.lp_mode_list):
            l_val, m_val = lf.lp_mode_list[mode_num]
            title = f"{label} far field, l={l_val}, m={m_val}"
            outfile = os.path.join(
                outdir,
                f"far_field_{mode_num:02d}_{label}_l{l_val}_m{m_val}.png"
            )
        else:
            title = f"{label} far field"
            outfile = os.path.join(outdir, f"far_field_{mode_num:02d}_{label}.png")

        plot_far_field_mode(
            far_field,
            title=title,
            outpath=outfile,
            log_intensity=False,
            zoom_pixels=25
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
