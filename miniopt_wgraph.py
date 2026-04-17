import copy
import json
import csv
import pickle
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.callbacks import VerboseCallback

from thirdopt import train_one_run, get_base_pdict, datadir


# ------------------------------------------------------------
# Save directory
# ------------------------------------------------------------
save_dir = datadir
os.makedirs(save_dir, exist_ok=True)

# ------------------------------------------------------------
# Search space
# ------------------------------------------------------------
space = [
    Real(1e-5, 1e-4, prior='log-uniform', name='learningRate'),
    Real(0.02, 0.15, name='dropout_rate'),
    Real(0.0, 0.32, name='dropout_rate_dense'),
    Real(0.0, 0.8, name='dropout_rate_psf'),
]

trial_log = []

# Live plotting state
plt.ion()
fig_live, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
live_losses = []
live_min_losses = []

run_date = datetime.datetime.now().strftime("%Y%m%d")
run_time = datetime.datetime.now().strftime("%H%M")
timestamp = f"{run_date}-{run_time}"
run_prefix = f"bayesopt_{timestamp}"

all_trials_filename = os.path.join(save_dir, "all_trial_results.csv")


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


def save_trial_log_csv(trial_log, filename):
    if not trial_log:
        return

    param_keys = sorted(trial_log[0]["params"].keys())

    fieldnames = [
        "trial",
        "objective_val_loss",
        "final_val_loss",
    ] + param_keys

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in trial_log:
            out_row = {
                "trial": row["trial"],
                "objective_val_loss": row["objective_val_loss"],
                "final_val_loss": row["final_val_loss"],
            }
            for k in param_keys:
                out_row[k] = row["params"].get(k, None)
            writer.writerow(out_row)


def append_trial_to_master_csv(trial_entry, filename, run_prefix, run_date, run_time):
    param_keys = sorted(trial_entry["params"].keys())

    fieldnames = [
        "run_prefix",
        "run_date",
        "run_time",
        "trial",
        "objective_val_loss",
        "final_val_loss",
    ] + param_keys

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        out_row = {
            "run_prefix": run_prefix,
            "run_date": run_date,
            "run_time": run_time,
            "trial": trial_entry["trial"],
            "objective_val_loss": trial_entry["objective_val_loss"],
            "final_val_loss": trial_entry["final_val_loss"],
        }

        for k in param_keys:
            out_row[k] = trial_entry["params"].get(k, None)

        writer.writerow(out_row)


def save_best_results(best_pdict, best_loss, trial_log, prefix):
    with open(os.path.join(save_dir, f"{prefix}_best_results.json"), "w") as f:
        json.dump(
            make_json_safe({
                "best_objective_val_loss": float(best_loss),
                "best_params": best_pdict,
                "n_trials_completed": len(trial_log),
                "run_date": run_date,
                "run_time": run_time,
            }),
            f,
            indent=2,
        )

    np.savez(
        os.path.join(save_dir, f"{prefix}_best_results.npz"),
        best_objective_val_loss=float(best_loss),
        best_params=np.array([best_pdict], dtype=object),
        all_trials=np.array(trial_log, dtype=object),
        run_date=run_date,
        run_time=run_time,
    )

    save_trial_log_csv(trial_log, os.path.join(save_dir, f"{prefix}_all_trials.csv"))


def save_skopt_result(res, prefix):
    with open(os.path.join(save_dir, f"{prefix}_skopt_result.pkl"), "wb") as f:
        pickle.dump(res, f)


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
        os.path.join(save_dir, f"{run_prefix}_live_progress.png"),
        dpi=300,
        bbox_inches="tight"
    )


@use_named_args(space)
def objective(**params):
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

        trial_entry = {
            "trial": int(len(trial_log) + 1),
            "params": make_json_safe(copy.deepcopy(result["pdict"])),
            "objective_val_loss": float(objective_val),
            "final_val_loss": float(result["final_val_loss"]),
            "history_val_loss": [float(x) for x in result["history_val_loss"]],
        }
        trial_log.append(trial_entry)

        append_trial_to_master_csv(
            trial_entry=trial_entry,
            filename=all_trials_filename,
            run_prefix=run_prefix,
            run_date=run_date,
            run_time=run_time
        )

        live_losses.append(objective_val)
        live_min_losses.append(min(live_losses))
        update_live_plot()

        save_trial_log_csv(
            trial_log,
            os.path.join(save_dir, f"{run_prefix}_all_trials.csv")
        )

        np.savez(
            os.path.join(save_dir, f"{run_prefix}_all_trials.npz"),
            all_trials=np.array(trial_log, dtype=object),
            live_losses=np.array(live_losses),
            live_min_losses=np.array(live_min_losses),
            run_date=run_date,
            run_time=run_time,
        )

        print(f"Trial {len(trial_log)} objective (min val_loss): {objective_val:.8f}")
        print(f"Params: {params}")
        print(f"Running best loss: {min(live_losses):.8f}")
        print("-" * 60)

        return objective_val

    except Exception as e:
        print("Trial failed with params:", params)
        print("Error:", repr(e))
        print("-" * 60)
        return 1e6


def main():
    res = None

    try:
        res = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=40,
            n_initial_points=8,
            acq_func='EI',
            random_state=42,
            callback=[VerboseCallback(n_total=40)]
        )

    except KeyboardInterrupt:
        print("\nOptimisation stopped manually.")

    if trial_log:
        best_trial = min(trial_log, key=lambda x: x["objective_val_loss"])
        best_pdict = make_json_safe(copy.deepcopy(best_trial["params"]))
        best_loss = float(best_trial["objective_val_loss"])

        print("\nBest result so far")
        print("Best objective (min val_loss):", best_loss)
        print("Best parameters:")
        for k, v in best_pdict.items():
            print(f"{k}: {v}")

        save_best_results(best_pdict, best_loss, trial_log, run_prefix)

    if res is not None:
        best_params = {dim.name: make_json_safe(val) for dim, val in zip(space, res.x)}
        best_pdict = get_base_pdict()
        best_pdict.update(best_params)
        best_pdict = make_json_safe(best_pdict)

        print("\nFinal best result")
        print("Best objective (min val_loss):", float(res.fun))
        print("Best parameters:")
        for k, v in best_pdict.items():
            print(f"{k}: {v}")

        save_skopt_result(res, run_prefix)
        save_best_results(best_pdict, float(res.fun), trial_log, run_prefix)

        fig_live.savefig(
            os.path.join(save_dir, f"{run_prefix}_final_progress.png"),
            dpi=300,
            bbox_inches="tight"
        )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
