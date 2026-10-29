
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras

# ============================================================
# 1. FILE PATHS AND SETTINGS
# ============================================================

DATA_DIR = Path("/home/manav/PL-NN-testdata_forDec2025/")
OUTPUT_DIR = Path("/home/manav/HONOURABLE")

PUPIL_RADIUS = 31
TEST_FRACTION = 0.2
BATCH_SIZE = 16
N_BINS = 100

# Replace filenames if yours are different.
# IMPORTANT: Replace the mixed-seeing model placeholder.

CASES = {
    "Better seeing": {
        "pl": "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_proc2_files-combined.npz",
        "wf": "slmcube_20240605_seeing_0.4-10-scl0.5_rand_10K_01_files-64px_combined.npz",
        "model": "pl2wf2psf_data202407_model01_20260917-2026.keras",
    },
    "Worse seeing": {
        "pl": "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz",
        "wf": "slmcube_20240605_seeing_0.4-10-scl1_rand_10K_01_files-combined.npz",
        #"model": "pl2wf2psf_data202407_model01_20260914-2022.keras",
        "model": "pl2wf2psf_data202407_model01_20260917-2026.keras",
    },
    "Mixed seeing": {
        "pl": "pllabdata_20240605_singlepsf_01_slmcube_20240605_seeing_0.4-10-scl1-scl0.5_mixed_files-combined.npz",
        "wf": "slmcube_20240605_seeing_0.4-10-scl1-scl0.5_mixed_files-combined.npz",
        #"model": "pl2wf2psf_data202407_model01_20260922-2318.keras",
        "model": "pl2wf2psf_data202407_model01_20260917-2026.keras",
        "max_samples": 100000,
    },
}


# ============================================================
# 2. CIRCULAR PUPIL MASK
# ============================================================

def pupil_mask(shape, radius=31):
    ny, nx = shape
    yy, xx = np.ogrid[:ny, :nx]

    return (
        (yy - (ny - 1) / 2)**2
        + (xx - (nx - 1) / 2)**2
    ) <= radius**2


# ============================================================
# 3. INPUT WAVEFRONT ERROR
# ============================================================

def input_wfe(phase, mask):
    """
    Standard deviation of each wavefront within the pupil.
    Output in radians.
    """
    return np.std(phase[:, mask], axis=1)


# ============================================================
# 4. NEURAL NETWORK RECONSTRUCTION RMSE
# ============================================================

def reconstruction_rmse(pred, true, mask):
    """
    Calculate wrapped pupil RMSE for each image.
    """

    residual = np.angle(
        np.exp(1j * (pred - true))
    )

    return np.sqrt(
        np.mean(
            residual[:, mask]**2,
            axis=1
        )
    )


# ============================================================
# 5. LOAD DATA AND RUN NN
# ============================================================

def load_case(spec):

    pl_path = DATA_DIR / spec["pl"]
    wf_path = DATA_DIR / spec["wf"]
    model_path = DATA_DIR / spec["model"]

    norm_path = DATA_DIR / (
        model_path.stem + "_normfacts.npz"
    )

    for path in [pl_path, wf_path, model_path, norm_path]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

    with np.load(pl_path, allow_pickle=True) as data:
        pl = np.asarray(
            data["all_plims"],
            dtype=np.float32
        )

    with np.load(wf_path, allow_pickle=True) as data:
        wf = np.asarray(
            data["all_pupphase"],
            dtype=np.float32
        )

    if len(pl) != len(wf):
        raise ValueError(
            "PL and WF datasets have different numbers of images."
        )

    # IMPORTANT:
    # This assumes the first 20% are the held-out test set.
    # Change this if your original split was different.

    # Limit mixed-seeing dataset to 200,000 samples
    if spec.get("max_samples") is not None:
        max_samples = spec["max_samples"]
        pl = pl[:max_samples]
        wf = wf[:max_samples]

# Select the test fraction
    n = int(len(pl) * TEST_FRACTION)

    pl = pl[:n]
    wf = wf[:n]


    with np.load(norm_path) as norm:
        pl_mean, _, pl_std = norm["PL"]
        wf_mean, _, wf_std = norm["WF"]

    x = (pl - pl_mean) / pl_std

    if x.ndim == 3:
        x = x[..., np.newaxis]

    model = keras.models.load_model(
        model_path,
        compile=False
    )

    prediction = model.predict(
        x,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # In your previous model, WF was the second output.
    pred_wf = (
        prediction[1]
        if isinstance(prediction, (list, tuple))
        else prediction["outlayer_wf"]
    )

    if pred_wf.ndim == 4:
        pred_wf = np.squeeze(pred_wf, axis=-1)

    pred_wf = pred_wf * wf_std + wf_mean

    if pred_wf.shape != wf.shape:
        raise ValueError(
            f"Prediction shape {pred_wf.shape} "
            f"does not match WF shape {wf.shape}"
        )

    return wf, pred_wf


# ============================================================
# 6. HISTOGRAM PLOTTING
# ============================================================

def plot_histogram(groups, xlabel, title, filename):

    fig, ax = plt.subplots(figsize=(9, 5.5))

    all_values = np.concatenate(list(groups.values()))

    bins = np.linspace(
        all_values.min(),
        all_values.max(),
        N_BINS + 1
    )

    colours = {
        "Better seeing": "tab:blue",
        "Worse seeing": "tab:orange",
        "Mixed seeing": "tab:purple"
    }

    for name, values in groups.items():

        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.25,
            color=colours[name],
            label=f"{name} (n={len(values)})"
        )

# Thin outline following the histogram shape
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.2,
            color=colours[name]
        )

        ax.axvline(
            np.median(values),
            color=colours[name],
            linestyle="--",
            linewidth=2
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density (rad$^{-1}$)")
    ax.set_title(title)

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 7. MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    wfe_results = {}
    rmse_results = {}

    for name, spec in CASES.items():

        print(f"\nProcessing: {name}")

        true_wf, predicted_wf = load_case(spec)

        mask = pupil_mask(
            true_wf.shape[-2:],
            PUPIL_RADIUS
        )

        wfe_results[name] = input_wfe(
            true_wf,
            mask
        )

        rmse_results[name] = reconstruction_rmse(
            predicted_wf,
            true_wf,
            mask
        )

        print(
            "Median input WFE:",
            np.median(wfe_results[name]),
            "rad"
        )

        print(
            "Median reconstruction RMSE:",
            np.median(rmse_results[name]),
            "rad"
        )

    # HISTOGRAM 1: INPUT WFE

    plot_histogram(
        wfe_results,
        xlabel="Input wavefront RMS error [rad]",
        title="Input WFE by Seeing Condition",
        filename="input_wfe_histogram.png"
    )

    # HISTOGRAM 2: RECONSTRUCTION RMSE

    plot_histogram(
        rmse_results,
        xlabel="NN reconstruction RMSE [rad]",
        title="Wavefront Reconstruction RMSE for NNs trained on Worse Seeing Data",
        filename="nn_reconstruction_rmse_histogram.png"
    )

    print("\nBoth histograms generated successfully.")


if __name__ == "__main__":
    main()
