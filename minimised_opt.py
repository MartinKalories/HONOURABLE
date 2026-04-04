import copy
import json
import numpy as np

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from thirdopt import train_one_run, get_base_pdict


space = [
    Real(1e-5, 5e-3, prior='log-uniform', name='learningRate'),
    Real(0.0, 0.4, name='dropout_rate'),
    Real(0.0, 0.6, name='dropout_rate_dense'),
    Real(0.0, 0.6, name='dropout_rate_psf'),
    Integer(512, 4096, name='n_units_dense'),
]

trial_log = []


@use_named_args(space)
def objective(**params):
    result = train_one_run(
        pdict_override=None,
        do_predictions=False,
        do_plotting=False,
        save_model=False,
        save_preds=False,
        save_movie=False,
        verbose=0,
    )

    objective_val = float(result["objective_val_loss"])

    trial_log.append({
        "params": copy.deepcopy(result["pdict"]),
        "objective_val_loss": objective_val,
        "final_val_loss": float(result["final_val_loss"]),
        "history_val_loss": [float(x) for x in result["history_val_loss"]],
    })

    print(f"Trial objective (min val_loss): {objective_val:.8f}")
    print(f"Params: {params}")
    print("-" * 60)

    return objective_val


def main():
    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=15,
        n_initial_points=5,
        acq_func='EI',
        random_state=42
    )

    best_params = {dim.name: val for dim, val in zip(space, res.x)}
    best_pdict = get_base_pdict()
    best_pdict.update(best_params)

    print("\nBest result")
    print("Best objective (min val_loss):", res.fun)
    print("Best parameters:")
    for k, v in best_pdict.items():
        print(f"{k}: {v}")

    with open("bayes_opt_results.json", "w") as f:
        json.dump({
            "best_objective_val_loss": float(res.fun),
            "best_params": best_pdict,
            "all_trials": trial_log
        }, f, indent=2)


if __name__ == "__main__":
    main()
