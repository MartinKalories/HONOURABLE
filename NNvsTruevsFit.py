import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


# ============================================================
# SETTINGS
# ============================================================

datadir = "/home/manav/PL-NN-testdata_forDec2025/"
# datadir = "/Users/manavkalra/Downloads/PL-NN-testdata_forDec2025/"

model_filename = (
    "pl2wf2psf_data202407_model01_20260917-2026.keras"
)

normfacts_filename = (
    "pl2wf2psf_data202407_model01_20260917-2026_normfacts.npz"
)

PL_filename = (
      "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_proc2_files-combined.npz"
)

WF_filename = (
    "slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_files-64px_combined.npz"
)

LP_RESULTS_FILE = (
 "WF_phase_farfieldLP_fit_results_100000wfs_27modes_full_custom_20261003_220237.npz"
)

PUPIL_RADIUS_PIXELS = 31

# Which image to display
IMAGE_INDEX = 73832

# Save figure?
SAVE_FIGURE = True

OUTPUT_FILENAME = "WF_NN_true_LP_comparison.png"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_phase(phi):
    """Wrap phase to [-pi, pi]."""
    return np.angle(np.exp(1j * phi))


def make_circular_mask(shape, radius_pixels):
    """Create circular pupil mask."""

    ny, nx = shape

    cy = (ny - 1) / 2
    cx = (nx - 1) / 2

    y, x = np.ogrid[:ny, :nx]

    return (
        (x - cx)**2 +
        (y - cy)**2
    ) <= radius_pixels**2


def wrapped_rmse(image1, image2, mask):
    """Wrapped RMSE between two phase images inside pupil."""

    residual = wrap_phase(image1 - image2)

    return np.sqrt(
        np.mean(
            residual[mask]**2
        )
    )


# ============================================================
# LOAD NEURAL NETWORK
# ============================================================

print("Loading neural network...")

model = keras.models.load_model(
    datadir + model_filename
)


# ============================================================
# LOAD NORMALISATION VALUES
# ============================================================

print("Loading normalisation values...")

normfacts = np.load(
    datadir + normfacts_filename
)

PL_normfacts = normfacts["PL"]
WF_normfacts = normfacts["WF"]

PL_mean = PL_normfacts[0]
PL_std = PL_normfacts[2]

WF_mean = WF_normfacts[0]
WF_std = WF_normfacts[2]


# ============================================================
# LOAD PL INPUT DATA
# ============================================================

print("Loading PL data...")

npf = np.load(
    datadir + PL_filename,
    allow_pickle=True
)

X_test = npf["all_plims"]


# ============================================================
# LOAD TRUE WAVEFRONTS
# ============================================================

print("Loading true wavefronts...")

npf = np.load(
    datadir + WF_filename,
    allow_pickle=True
)

true_wf = npf["all_pupphase"]


# ============================================================
# LOAD LP FIT RESULTS
# ============================================================

print("Loading LP fitting results...")

lp_results = np.load(
    datadir + LP_RESULTS_FILE,
    allow_pickle=True
)

coeffs_all = lp_results["coeffs_all"]
far_modes = lp_results["far_modes"]


# ============================================================
# MAKE SURE EVERYTHING USES SAME NUMBER OF IMAGES
# ============================================================

n_compare = min(
    len(X_test),
    len(true_wf),
    len(coeffs_all)
)

X_test = X_test[:n_compare]
true_wf = true_wf[:n_compare]
coeffs_all = coeffs_all[:n_compare]

print("Number of comparable images:", n_compare)


# ============================================================
# CHECK IMAGE INDEX
# ============================================================

if IMAGE_INDEX >= n_compare:
    raise ValueError(
        f"IMAGE_INDEX = {IMAGE_INDEX} is too large. "
        f"Maximum index is {n_compare - 1}."
    )


# ============================================================
# RUN NN
# ============================================================

print("Running neural network...")

X_test_norm = (
    X_test - PL_mean
) / PL_std

predictions = model.predict(
    X_test_norm,
    verbose=1
)

# Model output:
# predictions[0] = PSF
# predictions[1] = WF

predictions_wf_norm = predictions[1]

if predictions_wf_norm.ndim == 4:
    predictions_wf_norm = predictions_wf_norm[..., 0]


# Convert NN WF back into radians
nn_wf = (
    predictions_wf_norm * WF_std
) + WF_mean


# ============================================================
# RECONSTRUCT LP FITTED WAVEFRONTS
# ============================================================

print("Reconstructing LP fitted wavefronts...")

lp_fields = np.einsum(
    "nm,mhw->nhw",
    coeffs_all,
    far_modes
)

lp_wf = np.angle(
    lp_fields
)


# ============================================================
# CREATE PUPIL MASK
# ============================================================

pupil_mask = make_circular_mask(
    true_wf.shape[-2:],
    PUPIL_RADIUS_PIXELS
)


# ============================================================
# SELECT IMAGE
# ============================================================

i = IMAGE_INDEX

true_image = true_wf[i]
nn_image = nn_wf[i]
lp_image = lp_wf[i]


# ============================================================
# WRAP PHASE IMAGES
# ============================================================

true_wrapped = wrap_phase(
    true_image
)

nn_wrapped = wrap_phase(
    nn_image
)

lp_wrapped = wrap_phase(
    lp_image
)


# ============================================================
# CALCULATE WRAPPED RESIDUALS
# ============================================================

nn_true_residual = wrap_phase(
    nn_image - true_image
)

nn_lp_residual = wrap_phase(
    nn_image - lp_image
)


# ============================================================
# CALCULATE RMSE
# ============================================================

nn_true_rmse = wrapped_rmse(
    nn_image,
    true_image,
    pupil_mask
)

nn_lp_rmse = wrapped_rmse(
    nn_image,
    lp_image,
    pupil_mask
)


print("\n======================================")
print("IMAGE", i)
print("======================================")

print(
    f"NN vs True WF RMSE = "
    f"{nn_true_rmse:.4f} rad"
)

print(
    f"NN vs LP Fit RMSE  = "
    f"{nn_lp_rmse:.4f} rad"
)

# ============================================================
# FULL DATASET RMSE COMPARISON
# ============================================================

nn_true_residual_all = wrap_phase(
    nn_wf - true_wf
)

nn_lp_residual_all = wrap_phase(
    nn_wf - lp_wf
)

# Per-image RMSE
nn_true_per_image = np.sqrt(
    np.mean(
        nn_true_residual_all[:, pupil_mask]**2,
        axis=1
    )
)

nn_lp_per_image = np.sqrt(
    np.mean(
        nn_lp_residual_all[:, pupil_mask]**2,
        axis=1
    )
)

# Mean per-image RMSE
nn_true_mean = np.mean(nn_true_per_image)
nn_lp_mean = np.mean(nn_lp_per_image)

# Global RMSE
nn_true_global = np.sqrt(
    np.mean(
        nn_true_residual_all[:, pupil_mask]**2
    )
)

nn_lp_global = np.sqrt(
    np.mean(
        nn_lp_residual_all[:, pupil_mask]**2
    )
)

print("\nFULL DATASET RESULTS")
print("-----------------------------")
print(f"Number of images: {n_compare}")

print(f"NN vs True mean RMSE: {nn_true_mean:.4f}")
print(f"NN vs LP mean RMSE:   {nn_lp_mean:.4f}")

print(f"NN vs True global RMSE: {nn_true_global:.4f}")
print(f"NN vs LP global RMSE:   {nn_lp_global:.4f}")

print("\nSINGLE IMAGE RESULTS")
print("-----------------------------")
print(f"NN vs True: {nn_true_rmse:.4f}")
print(f"NN vs LP:   {nn_lp_rmse:.4f}")


# ============================================================
# MASK OUTSIDE PUPIL
# ============================================================

true_plot = np.where(
    pupil_mask,
    true_wrapped,
    np.nan
)

nn_plot = np.where(
    pupil_mask,
    nn_wrapped,
    np.nan
)

lp_plot = np.where(
    pupil_mask,
    lp_wrapped,
    np.nan
)

nn_true_res_plot = np.where(
    pupil_mask,
    nn_true_residual,
    np.nan
)

nn_lp_res_plot = np.where(
    pupil_mask,
    nn_lp_residual,
    np.nan
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(13, 8)
)


# ------------------------------------------------------------
# TOP ROW: TRUE vs NN
# ------------------------------------------------------------

im = axes[0, 0].imshow(
    true_plot,
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi
)

axes[0, 0].set_title(
    "True Wavefront"
)

axes[0, 0].set_axis_off()


axes[0, 1].imshow(
    nn_plot,
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi
)

axes[0, 1].set_title(
    "NN Prediction"
)

axes[0, 1].set_axis_off()


res1 = axes[0, 2].imshow(
    nn_true_res_plot,
    cmap="bwr",
    vmin=-np.pi,
    vmax=np.pi
)

axes[0, 2].set_title(
    "NN − True Residual\n"
    f"RMSE = {nn_true_rmse:.3f} rad"
)

axes[0, 2].set_axis_off()


# ------------------------------------------------------------
# BOTTOM ROW: LP FIT vs NN
# ------------------------------------------------------------

axes[1, 0].imshow(
    lp_plot,
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi
)

axes[1, 0].set_title(
    "Best-Case LP Fit"
)

axes[1, 0].set_axis_off()


axes[1, 1].imshow(
    nn_plot,
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi
)

axes[1, 1].set_title(
    "NN Prediction"
)

axes[1, 1].set_axis_off()


res2 = axes[1, 2].imshow(
    nn_lp_res_plot,
    cmap="bwr",
    vmin=-np.pi,
    vmax=np.pi
)

axes[1, 2].set_title(
    "NN − LP Fit Residual\n"
    f"RMSE = {nn_lp_rmse:.3f} rad"
)

axes[1, 2].set_axis_off()


# ============================================================
# COLOUR BARS
# ============================================================

# Shared phase colourbar
cbar_phase = fig.colorbar(
    im,
    ax=[
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1]
    ],
    fraction=0.025,
    pad=0.02
)

cbar_phase.set_label(
    "Phase [rad]"
)


# Shared residual colourbar
cbar_res = fig.colorbar(
    res1,
    ax=[
        axes[0, 2],
        axes[1, 2]
    ],
    fraction=0.05,
    pad=0.03
)

cbar_res.set_label(
    "Wrapped phase residual [rad]"
)


# ============================================================
# ROW LABELS
# ============================================================

fig.text(
    0.02,
    0.73,
    "NN vs True WF",
    rotation=90,
    va="center",
    ha="center",
    fontsize=14,
    fontweight="bold"
)

fig.text(
    0.02,
    0.28,
    "NN vs LP Fit",
    rotation=90,
    va="center",
    ha="center",
    fontsize=14,
    fontweight="bold"
)


# ============================================================
# FINAL FORMATTING
# ============================================================

fig.suptitle(
    f"Wavefront Comparison — Image {i}",
    fontsize=16,
    fontweight="bold"
)

plt.subplots_adjust(
    left=0.08,
    right=0.88,
    top=0.90,
    bottom=0.06,
    wspace=0.08,
    hspace=0.18
)


# ============================================================
# SAVE
# ============================================================

if SAVE_FIGURE:

    save_path = (
        datadir +
        OUTPUT_FILENAME
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nSaved figure to:",
        save_path
    )


plt.show()
