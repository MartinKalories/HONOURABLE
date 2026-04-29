import copy
import json
import csv
import datetime
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from skopt import Optimizer, dump, load
from skopt.space import Real, Integer, Categorical

from thirdopt import train_one_run, get_base_pdict, datadir


# ------------------------------------------------------------
# Run ID
# Keep this the same to resume a crashed run.
# Change this when starting a new experiment.
# ------------------------------------------------------------
RUN_ID = "bayesopt_current_60ksub"

save_dir = datadir
os.makedirs(save_dir, exist_ok=True)

checkpoint_npz = os.path.join(save_dir, f"{RUN_ID}_checkpoint.npz")
optimizer_checkpoint = os.path.join(save_dir, f"{RUN_ID}_optimizer.pkl")
all_trials_filename = os.path.join(save_dir, f"{RUN_ID}_all_trials.csv")


# ------------------------------------------------------------
# Search space
# ------------------------------------------------------------
space = [
    Real(1e-5, 5e-3, prior="log-uniform", name="learningRate"),
    Real(0.0, 0.4, name="dropout_rate"),
    Real(0.0, 0.6, name="dropout_rate_dense"),
    Real(0.0, 0.8, name="dropout_rate_psf"),
    Integer(512, 4096, name="n_units_dense"),
    Categorical([16, 32, 64], name="batchSize"),
    Categorical([3, 5, 7], name="ksz_enc"),
    Categorical([3, 5], name="ksz_psf"),
    Categorical([3, 5], name="ksz_wf"),
    Categorical([64, 96, 128], name="nfilts_enc"),
    Categorical([32, 64, 96], name="nfilts_psf"),
    Categorical([32, 64, 96], name="nfilts_wf"),
    Real(0.5, 3.0, name="loss_weight"),
    Categorical(["relu", "elu", "gelu"], name="actFunc"),
]

space_param_keys = [dim.name for dim in space]


# ------------------------------------------------------------
# Live state
# ------------------------------------------------------------
trial_log = []
live_losses = []
live_min_losses = []

plt.ion()
fig_live, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_checkpoint(opt):
    np.savez(
        checkpoint_npz,
        trial_log=np.array(trial_log, dtype=object),
        live_losses=np.array(live_losses),
        live_min_losses=np.array(live_min_losses),
    )

    dump(opt, optimizer_checkpoint, store_objective=False)
    print("Checkpoint saved.")


def load_checkpoint_if_available():
    global trial_log, live_losses, live_min_losses

    if os.path.exists(checkpoint_npz) and os.path.exists(optimizer_checkpoint):
        print("Loading checkpoint...")

        data = np.load(checkpoint_npz, allow_pickle=True)

        trial_log = list(data["trial_log"])
        live_losses = list(data["live_losses"])
        live_min_losses = list(data["live_min_losses"])

        opt = load(optimizer_checkpoint)

        print(f"Resumed from {len(trial_log)} completed trials.")
        return opt

    print("No checkpoint found. Starting fresh.")

    return Optimizer(
        dimensions=space,
        base_estimator="GP",
        acq_func="EI",
        random_state=42,
        n_initial_points=2,
    )


def append_trial_to_csv(trial_entry, filename):
    fieldnames = [
        "trial",
        "objective_val_loss",
        "final_val_loss",
    ] + space_param_keys

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {
            "trial": trial_entry["trial"],
            "objective_val_loss": trial_entry["objective_val_loss"],
            "final_val_loss": trial_entry["final_val_loss"],
        }

        for k in space_param_keys:
            row[k] = trial_entry["params"].get(k, None)

        writer.writerow(row)


def save_full_trial_log_csv(trial_log, filename):
    if not trial_log:
        return

    fieldnames = [
        "trial",
        "objective_val_loss",
        "final_val_loss",
    ] + space_param_keys

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial_entry in trial_log:
            row = {
                "trial": trial_entry["trial"],
                "objective_val_loss": trial_entry["objective_val_loss"],
                "final_val_loss": trial_entry["final_val_loss"],
            }

            for k in space_param_keys:
                row[k] = trial_entry["params"].get(k, None)

            writer.writerow(row)


def save_best_results():
    if not trial_log:
        return

    best_trial = min(trial_log, key=lambda x: x["objective_val_loss"])
    best_params = make_json_safe(copy.deepcopy(best_trial["params"]))

    best_pdict = get_base_pdict()
    best_pdict.update(best_params)
    best_pdict = make_json_safe(best_pdict)

    best_loss = float(best_trial["objective_val_loss"])

    best_json_path = os.path.join(save_dir, f"{RUN_ID}_best_results.json")
    best_npz_path = os.path.join(save_dir, f"{RUN_ID}_best_results.npz")

    with open(best_json_path, "w") as f:
        json.dump(
            make_json_safe(
                {
                    "best_objective_val_loss": best_loss,
                    "best_params": best_pdict,
                    "n_trials_completed": len(trial_log),
                }
            ),
            f,
            indent=2,
        )

    np.savez(
        best_npz_path,
        best_objective_val_loss=best_loss,
        best_params=np.array([best_pdict], dtype=object),
        all_trials=np.array(trial_log, dtype=object),
    )

    save_full_trial_log_csv(trial_log, all_trials_filename)

    print("\nBest result so far")
    print("Best objective:", best_loss)
    print("Best parameters:")
    for k, v in best_pdict.items():
        print(f"{k}: {v}")


def update_live_plot():
    ax1.clear()
    ax2.clear()

    iterations = np.arange(1, len(live_losses) + 1)

    ax1.plot(iterations, live_losses, marker="o")
    ax1.set_title("Iteration vs loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Objective loss")

    ax2.plot(iterations, live_min_losses, marker="o")
    ax2.set_title("Iteration vs min loss")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Running min loss")

    fig_live.tight_layout()
    fig_live.canvas.draw()
    fig_live.canvas.flush_events()
    plt.pause(0.01)

    fig_live.savefig(
        os.path.join(save_dir, f"{RUN_ID}_live_progress.png"),
        dpi=300,
        bbox_inches="tight",
    )


# ------------------------------------------------------------
# Main optimisation loop
# ------------------------------------------------------------
def main():
    n_calls = 50
    opt = load_checkpoint_if_available()

    try:
        while len(trial_log) < n_calls:
            x = opt.ask()

            params = {
                dim.name: make_json_safe(val)
                for dim, val in zip(space, x)
            }

            result = None

            try:
                result = train_one_run(
                    pdict_override=params,
                    do_predictions=False,
                    do_plotting=False,
                    save_model=False,
                    save_preds=False,
                    save_movie=False,
                    verbose=0,
                )

                objective_val = float(result["objective_val_loss"])
                final_val_loss = float(result["final_val_loss"])

                # ------------------------------------------------
                # NaN / Inf guard
                # ------------------------------------------------
                if not np.isfinite(objective_val):
                    print("NaN or Inf loss detected. Skipping this trial.")
                    print("Params:", params)

                    if result is not None:
                        del result
                        result = None

                    tf.keras.backend.clear_session()
                    gc.collect()

                    continue

            except Exception as e:
                print("Trial crashed with params:", params)
                print("Error:", repr(e))
                print("Stopping. Rerun the file to resume from the last completed trial.")

                if result is not None:
                    del result
                    result = None

                tf.keras.backend.clear_session()
                gc.collect()

                save_checkpoint(opt)
                return

            finally:
                if result is not None:
                    del result
                    result = None

                tf.keras.backend.clear_session()
                gc.collect()

            # ------------------------------------------------
            # Only successful, finite-loss trials reach here
            # ------------------------------------------------
            opt.tell(x, objective_val)

            trial_entry = {
                "trial": int(len(trial_log) + 1),
                "params": make_json_safe(copy.deepcopy(params)),
                "objective_val_loss": float(objective_val),
                "final_val_loss": float(final_val_loss),
            }

            trial_log.append(trial_entry)

            live_losses.append(objective_val)
            live_min_losses.append(min(live_losses))

            append_trial_to_csv(trial_entry, all_trials_filename)
            update_live_plot()
            save_checkpoint(opt)

            print(f"Trial {len(trial_log)} objective: {objective_val:.8f}")
            print(f"Params: {params}")
            print(f"Running best loss: {min(live_losses):.8f}")
            print("-" * 60)

    except KeyboardInterrupt:
        print("\nOptimisation stopped manually.")
        save_checkpoint(opt)

    save_best_results()

    dump(
        opt,
        os.path.join(save_dir, f"{RUN_ID}_optimizer_final.pkl"),
        store_objective=False,
    )

    fig_live.savefig(
        os.path.join(save_dir, f"{RUN_ID}_final_progress.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
