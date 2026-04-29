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
# Run ID (CHANGE THIS ONLY WHEN YOU WANT A NEW EXPERIMENT)
# ------------------------------------------------------------
RUN_ID = "bayesopt_10ksub"

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


def update_live_plot():
    ax1.clear()
    ax2.clear()

    iterations = np.arange(1, len(live_losses) + 1)

    ax1.plot(iterations, live_losses, marker="o")
    ax1.set_title("Iteration vs loss")

    ax2.plot(iterations, live_min_losses, marker="o")
    ax2.set_title("Iteration vs min loss")

    fig_live.tight_layout()
    fig_live.canvas.draw()
    fig_live.canvas.flush_events()
    plt.pause(0.01)


# ------------------------------------------------------------
# Main loop
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
                if not np.isfinite(objective_val):
                    print("NaN loss → skipping trial")

                    tf.keras.backend.clear_session()
                    gc.collect()

                    continue   # skip this trial

            except Exception as e:
                print("Trial failed with params:", params)
                print("Error:", repr(e))
                print("Stopping → will resume from last successful trial")

                if result is not None:
                    del result

                tf.keras.backend.clear_session()
                gc.collect()

                save_checkpoint(opt)
                return   # 🚨 STOP HERE

            finally:
                if result is not None:
                    del result

                tf.keras.backend.clear_session()
                gc.collect()

            # ONLY SUCCESSFUL TRIALS REACH HERE
            opt.tell(x, objective_val)

            trial_entry = {
                "trial": len(trial_log) + 1,
                "params": make_json_safe(copy.deepcopy(params)),
                "objective_val_loss": objective_val,
                "final_val_loss": final_val_loss,
            }

            trial_log.append(trial_entry)

            live_losses.append(objective_val)
            live_min_losses.append(min(live_losses))

            update_live_plot()

            save_checkpoint(opt)

            print(f"Trial {len(trial_log)}: {objective_val:.6f}")
            print("-" * 50)

    except KeyboardInterrupt:
        print("Stopped manually")
        save_checkpoint(opt)

    print("Finished optimisation.")


if __name__ == "__main__":
    main()
