import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

# ============================================================
# SETTINGS
# ============================================================

datadir = '/home/manav/PL-NN-testdata_forDec2025/'

# ------------------------------------------------------------
# MODEL TRAINED ON DATASET A
# ------------------------------------------------------------

model_filename = (
    'pl2wf2psf_data202407_model01_20260917-2026.keras'
)

normfacts_filename = ('pl2wf2psf_data202407_model01_20260917-2026_normfacts.npz')


# ------------------------------------------------------------
# NEW DATASET B
# ------------------------------------------------------------

PL_filename =  "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1-scl0.5_mixed_files-combined.npz"
PSF_filename = "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1-scl0.5_mixed_files-combined-PSFs.npz"
WF_filename = "slmcube_20240605_seeing_0.4-10-scl1-scl0.5_mixed_files-combined.npz"



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


# ============================================================
# RMSE IN ORIGINAL DATA UNITS
# ============================================================

rmse_psf = np.sqrt(np.mean((predictions_psf - y_test_psf) ** 2))

rmse_wf = np.sqrt(np.mean((predictions_wf - y_test_wf) ** 2))


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

