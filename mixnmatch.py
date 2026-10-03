import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import hashlib
import os

def fingerprint(arr):
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]

print("\nRunning script:")
print(os.path.abspath(__file__))

# ============================================================
# SETTINGS
# ============================================================

datadir = '/home/manav/PL-NN-testdata_forDec2025/'

# ------------------------------------------------------------
# MODEL TRAINED ON DATASET A
# ------------------------------------------------------------

model_filename = (
    'pl2wf2psf_data202407_model01_20260914-2022.keras'
)

normfacts_filename = ('pl2wf2psf_data202407_model01_20260914-2022_normfacts.npz')


# ------------------------------------------------------------
# NEW DATASET B
# ------------------------------------------------------------

#PL_filename =  "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1-scl0.5_mixed_files-combined.npz"
#PSF_filename ="pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1-scl0.5_mixed_files-combined-PSFs.npz"
#WF_filename = "slmcube_20240605_seeing_0.4-10-scl1-scl0.5_mixed_files-combined.npz"
#sc1
PL_filename= "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"
PSF_filename = "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined-PSFs.npz"
WF_filename = "slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz"
 

print("\n======================================")
print("FILES ACTUALLY LOADED")
print("======================================")

print("Model:")
print(os.path.abspath(datadir + model_filename))

print("\nNormfacts:")
print(os.path.abspath(datadir + normfacts_filename))

print("\nPL:")
print(os.path.abspath(datadir + PL_filename))

print("\nWF:")
print(os.path.abspath(datadir + WF_filename))

print("\nPL exists:",
      os.path.exists(datadir + PL_filename))

print("WF exists:",
      os.path.exists(datadir + WF_filename))

print("\nPL file size:",
      os.path.getsize(datadir + PL_filename))

print("WF file size:",
      os.path.getsize(datadir + WF_filename))


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

model = keras.models.load_model(
    datadir + model_filename
)

print("Model loaded.")


# ============================================================
# LOAD NORMALISATION FACTORS FROM DATASET A
# ============================================================

print("Loading training normalisation factors...")

normfacts = np.load(
   datadir + normfacts_filename
)

PL_normfacts = normfacts['PL']
WF_normfacts = normfacts['WF']
PSF_normfacts = normfacts['PSF']


# Extract individual values

PL_mean = PL_normfacts[0]
PL_std = PL_normfacts[2]

WF_mean = WF_normfacts[0]
WF_std = WF_normfacts[2]

PSF_min = PSF_normfacts[0]
PSF_max = PSF_normfacts[1]


print("\nNormalisation factors from Dataset A:")
print("PL mean:", PL_mean)
print("PL std :", PL_std)

print("WF mean:", WF_mean)
print("WF std :", WF_std)

print("PSF min:", PSF_min)
print("PSF max:", PSF_max)


# ============================================================
# LOAD NEW DATASET B
# ============================================================

print("\nLoading Dataset B...")

def wrapped_pupil_rmse(pred, true, radius_pixels=31):
    """
    pred, true: arrays shaped (N, H, W)

    Returns:
        mean_rmse      = mean of per-image RMS values
        per_image_rmse = RMS for each individual wavefront
        global_rmse    = pooled RMS over all images/pupil pixels
    """

    pred = np.squeeze(pred)
    true = np.squeeze(true)

    _, H, W = true.shape

    yy, xx = np.ogrid[:H, :W]

    cy = (H - 1) / 2
    cx = (W - 1) / 2

    pupil_mask = (
        (xx - cx)**2 +
        (yy - cy)**2
    ) <= radius_pixels**2

    # Phase-aware residual in [-pi, pi]
    residual = np.angle(
        np.exp(1j * (pred - true))
    )

    # RMS for each wavefront
    per_image_rmse = np.sqrt(
        np.mean(
            residual[:, pupil_mask]**2,
            axis=1
        )
    )

    # This matches your current LP "Mean RMS phase error"
    mean_rmse = np.mean(per_image_rmse)

    # Optional pooled/global RMSE
    global_rmse = np.sqrt(
        np.mean(
            residual[:, pupil_mask]**2
        )
    )

    return mean_rmse, per_image_rmse, global_rmse


# ------------------------------------------------------------
# PL INPUT
# ------------------------------------------------------------

npf = np.load(
    datadir + PL_filename,
    allow_pickle=True
)

X_test = npf['all_plims']


# ------------------------------------------------------------
# TRUE PSF
# ------------------------------------------------------------

npf = np.load(
    datadir + PSF_filename,
    allow_pickle=True
)

y_test_psf = npf['all_psfims']


# ------------------------------------------------------------
# TRUE WAVEFRONT
# ------------------------------------------------------------

npf = np.load(
    datadir + WF_filename,
    allow_pickle=True
)

y_test_wf = npf['all_pupphase']


print("Dataset B loaded.")

print("PL shape :", X_test.shape)
print("PSF shape:", y_test_psf.shape)
print("WF shape :", y_test_wf.shape)

max_samples = 100000

n_samples = min(
    max_samples,
    len(X_test),
    len(y_test_psf),
    len(y_test_wf)
)

X_test = X_test[:n_samples]
y_test_psf = y_test_psf[:n_samples]
y_test_wf = y_test_wf[:n_samples]

# ============================================================
# CHECK DATA LENGTHS
# ============================================================

if not (
    len(X_test)
    == len(y_test_psf)
    == len(y_test_wf)
):
    raise ValueError(
        "PL, PSF and WF datasets do not contain the same "
        "number of samples."
    )


# ============================================================
# NORMALISE DATASET B USING DATASET A NORMALISATION
# ============================================================

print("\nNormalising Dataset B using Dataset A factors...")


# ------------------------------------------------------------
# PL
# ------------------------------------------------------------

X_test_norm = (
    X_test - PL_mean
) / PL_std


# ------------------------------------------------------------
# WF
# ------------------------------------------------------------

y_test_wf_norm = (
    y_test_wf - WF_mean
) / WF_std


# ------------------------------------------------------------
# PSF
# ------------------------------------------------------------

y_test_psf_norm = (
    y_test_psf - PSF_min
) / PSF_max


# ============================================================
# RUN TRAINED MODEL ON DATASET B
# ============================================================

print("\nRunning model on Dataset B...")

predictions = model.predict(
    X_test_norm,
    
    verbose=1
)

predictions_psf_norm = predictions[0]
predictions_wf_norm = predictions[1]


# Remove final channel dimension if required
# e.g. (N, H, W, 1) -> (N, H, W)

if predictions_psf_norm.ndim == 4:
    predictions_psf_norm = predictions_psf_norm[..., 0]

if predictions_wf_norm.ndim == 4:
    predictions_wf_norm = predictions_wf_norm[..., 0]


print("Prediction complete.")


# ============================================================
# NORMALISED RMSE
# ============================================================

rmse_psf_norm = np.sqrt(
    np.mean(
        (predictions_psf_norm - y_test_psf_norm) ** 2 # _psf_norm instead of y_test if this doesnt work
    )
)

rmse_wf_norm = np.sqrt(
    np.mean(
        (predictions_wf_norm - y_test_wf_norm) ** 2 # _wf_norm as well
    )
)


print("\n===================================")
print("NORMALISED TEST RESULTS")
print("===================================")

print("PSF RMSE:", rmse_psf_norm)
print("WF RMSE :", rmse_wf_norm)



# ============================================================
# CONVERT PREDICTIONS BACK TO ORIGINAL UNITS
# ============================================================

predictions_psf = (predictions_psf_norm * PSF_max) + PSF_min

predictions_wf = (predictions_wf_norm * WF_std) + WF_mean

print("\n======================================")
print("ARRAY FINGERPRINTS")
print("======================================")

print("X_test          :", fingerprint(X_test))
print("X_test_norm     :", fingerprint(X_test_norm))
print("true_wf         :", fingerprint(y_test_wf))
print("prediction_norm :", fingerprint(predictions_wf_norm))
print("prediction_wf   :", fingerprint(predictions_wf))

print("\nNORMALISATION")
print("PL_mean:", PL_mean)
print("PL_std :", PL_std)
print("WF_mean:", WF_mean)
print("WF_std :", WF_std)

# ============================================================
# RMSE IN ORIGINAL DATA UNITS
# ============================================================

rmse_psf = np.sqrt(np.mean((predictions_psf - y_test_psf) ** 2))

rmse_wf = np.sqrt(np.mean((predictions_wf - y_test_wf) ** 2))

mean_wf_rmse, per_image_wf_rmse, global_wf_rmse = \
    wrapped_pupil_rmse(
        predictions_wf,
        y_test_wf,
        radius_pixels=31
    )

print(
    "Mean wrapped pupil WF RMSE [rad]:",
    mean_wf_rmse
)

print(
    "Global wrapped pupil WF RMSE [rad]:",
    global_wf_rmse
)


# ============================================================
# SAME RESULT DIRECTLY FROM NORMALISED RMSE
# ============================================================

rmse_psf_from_norm = rmse_psf_norm * PSF_max
rmse_wf_from_norm = rmse_wf_norm * WF_std


print("\n===================================")
print("NORMALISED TEST RESULTS")
print("===================================")

print("PSF RMSE (normalised):", rmse_psf_norm)
print("WF RMSE  (normalised):", rmse_wf_norm)


print("\n===================================")
print("ORIGINAL SCALE TEST RESULTS")
print("===================================")

print("PSF RMSE:", rmse_psf)
print("WF RMSE [rad]:", rmse_wf)


print("\nCross-check from normalised RMSE:")
print("PSF RMSE:", rmse_psf_from_norm)
print("WF RMSE [rad]:", rmse_wf_from_norm)


# ============================================================
# PLOT EXAMPLE PREDICTIONS
# ============================================================

num_examples = 5

for i in range(
    min(num_examples, len(X_test))
):

    plt.figure(figsize=(10, 8))


    # --------------------------------------------------------
    # TRUE WF
    # --------------------------------------------------------

    plt.subplot(2, 2, 1)

    plt.imshow(
        y_test_wf[i]
    )

    plt.title(
        "True WF"
    )

    plt.colorbar()


    # --------------------------------------------------------
    # PREDICTED WF
    # --------------------------------------------------------

    plt.subplot(2, 2, 2)

    plt.imshow(
        predictions_wf[i]
    )

    plt.title(
        "Predicted WF"
    )

    plt.colorbar()


    # --------------------------------------------------------
    # TRUE PSF
    # --------------------------------------------------------

    plt.subplot(2, 2, 3)

    plt.imshow(
        y_test_psf[i]
    )

    plt.title(
        "True PSF"
    )

    plt.colorbar()


    # --------------------------------------------------------
    # PREDICTED PSF
    # --------------------------------------------------------

    plt.subplot(2, 2, 4)

    plt.imshow(
        predictions_psf[i]
    )

    plt.title(
        "Predicted PSF"
    )

    plt.colorbar()


    plt.tight_layout()
    save_path = datadir + f"prediction_example_{i}.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("Saved:", save_path)

    plt.show()
    plt.show()

