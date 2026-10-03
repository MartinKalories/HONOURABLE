import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import hashlib

def fingerprint(arr):
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# ============================================================
# SETTINGS
# ============================================================

datadir = "/home/manav/PL-NN-testdata_forDec2025/"
#datadir = '/Users/manavkalra/Downloads/PL-NN-testdata_forDec2025/'
# ------------------------------------------------------------
# NN
# ------------------------------------------------------------

model_filename = (
    "pl2wf2psf_data202407_model01_20260914-2022.keras"
)

normfacts_filename = (
    "pl2wf2psf_data202407_model01_20260914-2022_normfacts.npz"
)
#scl1:pl2wf2psf_data202407_model01_20260914-2022.keras
#scl0.5: pl2wf2psf_data202407_model01_20260917-2026.keras
PL_filename = (
    "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"
)

WF_filename = (
    "slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"
)

#scl1
#"pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"
#psf_file_1 = "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined-PSFs.npz"
#wf_file_1 = "slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"
#scl0.5
#"pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_proc2_files-combined.npz"
#psf_file_2 = "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_proc2_files-combined-PSFs.npz"
#wf_file_2 = "slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_files-64px_combined.npz"

# ------------------------------------------------------------
# LP FIT RESULTS
#
# IMPORTANT:
# This must be an LP-fit result generated from the SAME
# WF dataset, in the SAME sample order.
# ------------------------------------------------------------

LP_RESULTS_FILE = (
    "WF_phase_farfieldLP_fit_results_10000wfs_27modes_full_custom_20261003_180229.npz"
)


PUPIL_RADIUS_PIXELS = 31


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_phase(phi):
    return np.angle(np.exp(1j * phi))


def make_circular_mask(shape, radius_pixels):
    ny, nx = shape

    cy = (ny - 1) / 2
    cx = (nx - 1) / 2

    y, x = np.ogrid[:ny, :nx]

    return (
        (x - cx)**2 +
        (y - cy)**2
    ) <= radius_pixels**2


def wrapped_pupil_rmse(pred, reference, radius_pixels=31):

    pred = np.squeeze(pred)
    reference = np.squeeze(reference)

    mask = make_circular_mask(
        pred.shape[-2:],
        radius_pixels
    )

    residual = wrap_phase(
        pred - reference
    )

    per_image_rmse = np.sqrt(
        np.mean(
            residual[:, mask]**2,
            axis=1
        )
    )

    mean_rmse = np.mean(
        per_image_rmse
    )

    global_rmse = np.sqrt(
        np.mean(
            residual[:, mask]**2
        )
    )

    return (
        mean_rmse,
        per_image_rmse,
        global_rmse,
        residual,
        mask
    )


# ============================================================
# LOAD NN
# ============================================================

print("Loading neural network...")

model = keras.models.load_model(
    datadir + model_filename
)

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
# LOAD NN TEST DATA
# ============================================================

print("Loading PL data...")

npf = np.load(
    datadir + PL_filename,
    allow_pickle=True
)

X_test = npf["all_plims"]


print("Loading true WF data...")

npf = np.load(
    datadir + WF_filename,
    allow_pickle=True
)

true_wf = npf["all_pupphase"]


# ============================================================
# LOAD LP RESULTS
# ============================================================

print("Loading LP fitting results...")

lp_results = np.load(
    datadir + LP_RESULTS_FILE,
    allow_pickle=True
)

coeffs_all = lp_results["coeffs_all"]
far_modes = lp_results["far_modes"]

n_lp = len(coeffs_all)

print("Number of LP fits:", n_lp)
print("LP modes shape:", far_modes.shape)


# ============================================================
# USE EXACTLY SAME NUMBER OF IMAGES
# ============================================================

n_compare = min(
    n_lp,
    len(X_test),
    len(true_wf)
)

print("Comparing", n_compare, "wavefronts")

X_test = X_test[:n_compare]
true_wf = true_wf[:n_compare]
coeffs_all = coeffs_all[:n_compare]


# ============================================================
# RUN NN
# ============================================================

X_test_norm = (
    X_test - PL_mean
) / PL_std

print("Running NN predictions...")

predictions = model.predict(
    X_test_norm,
    verbose=1
)

predictions_wf_norm = predictions[1]

if predictions_wf_norm.ndim == 4:
    predictions_wf_norm = predictions_wf_norm[..., 0]


# Convert back to radians

nn_wf = (
    predictions_wf_norm * WF_std
) + WF_mean


print("\n======================================")
print("ARRAY FINGERPRINTS")
print("======================================")

print("X_test          :", fingerprint(X_test))
print("X_test_norm     :", fingerprint(X_test_norm))
print("true_wf         :", fingerprint(true_wf))
print("prediction_norm :", fingerprint(predictions_wf_norm))
print("prediction_wf   :", fingerprint(nn_wf))

print("\nNORMALISATION")
print("PL_mean:", PL_mean)
print("PL_std :", PL_std)
print("WF_mean:", WF_mean)
print("WF_std :", WF_std)
# ============================================================
# EXACT NN SANITY CHECK
# This should reproduce the standalone NN script exactly
# ============================================================

check_mask = make_circular_mask(
    true_wf.shape[-2:],
    radius_pixels=31
)

check_residual = np.angle(
    np.exp(
        1j * (nn_wf - true_wf)
    )
)

check_global_rmse = np.sqrt(
    np.mean(
        check_residual[:, check_mask] ** 2
    )
)

print("\n======================================")
print("STANDALONE NN SANITY CHECK")
print("======================================")

print("N samples:", len(nn_wf))
print("NN shape :", nn_wf.shape)
print("True shape:", true_wf.shape)

print(
    "Direct global wrapped pupil RMSE:",
    check_global_rmse
)

# ============================================================
# RECONSTRUCT LP FITTED WAVEFRONTS
# ============================================================

print("Reconstructing LP fitted fields...")

# coeffs_all:
#       (N images, N modes)
#
# far_modes:
#       (N modes, H, W)
#
# Output:
#       (N images, H, W)

lp_fields = np.einsum(
    "nm,mhw->nhw",
    coeffs_all,
    far_modes
)

lp_wf = np.angle(
    lp_fields
)


# ============================================================
# CHECK SHAPES
# ============================================================

print("\nShapes")
print("NN WF :", nn_wf.shape)
print("LP WF :", lp_wf.shape)
print("True WF:", true_wf.shape)

if nn_wf.shape != lp_wf.shape:
    raise ValueError(
        "NN and LP fitted wavefronts have different shapes."
    )


# ============================================================
# NN vs LP BEST-CASE FIT
# ============================================================

(
    nn_lp_mean_rmse,
    nn_lp_per_image_rmse,
    nn_lp_global_rmse,
    nn_lp_residual,
    pupil_mask
) = wrapped_pupil_rmse(
    nn_wf,
    lp_wf,
    radius_pixels=PUPIL_RADIUS_PIXELS
)


print("\n======================================")
print("NN vs LP BEST-CASE FIT")
print("======================================")

print(
    "Mean wrapped pupil RMSE [rad]:",
    nn_lp_mean_rmse
)

print(
    "Global wrapped pupil RMSE [rad]:",
    nn_lp_global_rmse
)

print(
    "Median per-image RMSE [rad]:",
    np.median(nn_lp_per_image_rmse)
)


# ============================================================
# NN vs TRUE
# ============================================================

(
    nn_true_mean,
    nn_true_per_image,
    nn_true_global,
    _,
    _
) = wrapped_pupil_rmse(
    nn_wf,
    true_wf,
    radius_pixels=PUPIL_RADIUS_PIXELS
)


# ============================================================
# LP vs TRUE
# ============================================================

(
    lp_true_mean,
    lp_true_per_image,
    lp_true_global,
    _,
    _
) = wrapped_pupil_rmse(
    lp_wf,
    true_wf,
    radius_pixels=PUPIL_RADIUS_PIXELS
)

# ============================================================
# GLOBAL RMSE COMPARISON
# ============================================================

print("\n======================================")
print("GLOBAL WRAPPED PUPIL RMSE COMPARISON")
print("======================================")

print(
    "NN vs true WF [rad]:",
    nn_true_global
)

print(
    "LP fit vs true WF [rad]:",
    lp_true_global
)

print(
    "NN vs LP fit [rad]:",
    nn_lp_global_rmse
)

print("Standalone/available PL count:", len(X_test))
print("Available WF count:", len(true_wf))
print("LP count:", n_lp)
print("Actual n_compare:", n_compare)
# ============================================================
# SHOW ONE EXAMPLE
# ============================================================

i = 0

nn_plot = np.where(
    pupil_mask,
    wrap_phase(nn_wf[i]),
    np.nan
)

lp_plot = np.where(
    pupil_mask,
    wrap_phase(lp_wf[i]),
    np.nan
)

res_plot = np.where(
    pupil_mask,
    nn_lp_residual[i],
    np.nan
)


plt.figure(
    figsize=(15, 5)
)


plt.subplot(1, 3, 1)

plt.imshow(
    nn_plot,
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi
)

plt.title("NN predicted WF")
plt.colorbar()


plt.subplot(1, 3, 2)

plt.imshow(
    lp_plot,
    cmap="twilight",
    vmin=-np.pi,
    vmax=np.pi
)

plt.title("Best-case LP fit")
plt.colorbar()


plt.subplot(1, 3, 3)

plt.imshow(
    res_plot,
    cmap="bwr",
    vmin=-np.pi,
    vmax=np.pi
)

plt.title(
    "NN − LP wrapped residual\n"
    f"RMSE = "
    f"{nn_lp_per_image_rmse[i]:.3f} rad"
)

plt.colorbar()


plt.tight_layout()
plt.show()
