#!/usr/bin/env python

import sys
import argparse
import numpy as np
from scipy.optimize import minimize

from dgx_suite.dgx_environment import DgxEnvironment
from dgx_suite.dgx_parameter_space import DgxActionSpace, DgxObservationSpace
from dgx_suite.dgx_observer import Observer
from dgx_suite.opnic_utils import patch_opnic_wrapper

nb_average = 200.0

# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pi-pulse optimization with DGX and OPX (bit-level reward)."
    )

    # Kept for compatibility with spawn(**parameters.model_dump())
    parser.add_argument('--total_timesteps', type=int, default=700,
                        help='(unused) Total number of timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='(unused) Learning rate for the old RL agent')
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2],
                        help='Verbosity level (0: no output, 1: info, 2: debug)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='(unused) Device to use for training (cpu or cuda)')
    parser.add_argument('--learning_starts', type=int, default=100,
                        help='(unused) Number of steps of exploration before learning starts')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='(unused) Batch size for RL')
    parser.add_argument('--train_freq', type=int, default=1,
                        help='(unused) Train frequency for RL')
    parser.add_argument('--sigma_end_as_scale', type=float, default=1.,
                        help='(unused) Action noise parameter for RL')

    # These ARE used
    parser.add_argument('--observer', type=str,
                        help='Serialized base64 message packed string of an Observer instance')
    parser.add_argument('--action_space', type=str,
                        help='Serialized base64 message packed string of a DgxActionSpace instance')
    parser.add_argument('--observation_space', type=str,
                        help='Serialized base64 message packed string of a DgxObservationSpace instance')
    parser.add_argument('--path_to_python_wrapper', type=str,
                        help='Path to python-wrapper source code.')
    parser.add_argument('--force_recompile_python_wrapper', action=argparse.BooleanOptionalAction,
                        help='Whether to force-recompile the python wrapper')

    # New: how many shots per amplitude evaluation
    parser.add_argument('--shots_per_eval', type=int, default=200,
                        help='Number of single-shot experiments per amplitude evaluation')

    return parser.parse_args()



# ---------------------------------------------------------------------
# Helpers for running evaluations
# ---------------------------------------------------------------------
def estimate_excited_population(additional_detuning_storage,relative_detuning_storage_readout,additional_detuning_qubit,qubit_amp,storage_amp,readout_amp, dgx_env):






    observation, reward, terminated, _, info = dgx_env.step([float(additional_detuning_storage),float(relative_detuning_storage_readout),float(additional_detuning_qubit),float(qubit_amp),float(storage_amp),float(readout_amp), 1.0])


    p_exc = reward/nb_average 

    return p_exc


def objective(params, dgx_env, verbosity=0):
    """
    Objective for the scalar optimizer.

    For pi-pulse calibration we usually want to MAXIMIZE excited-state population p_exc.
    We achieve this by minimizing:

        cost(amp) = 1 - p_exc(amp)
    """
    p_exc = estimate_excited_population(params[0],params[1],params[2],params[3],params[4],params[5], dgx_env)

    # Maximize excitation -> minimize (1 - p_exc)
    cost = 1.0 - p_exc

    if verbosity >= 2:
        print(f"additional_detuning_storage: {params[0]}")
        print(f"relative_detuning_storage_readout: {params[1]}")
        print(f"additional_detuning_qubit: {params[2]}")
        print(f"qubit_amp: {params[3]}")
        print(f"storage_amp: {params[4]}")
        print(f"readout_amp: {params[5]}")
        print(f"cost: {cost}")

    return cost


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()

    # Deserialize environment-shape objects
    observer = Observer.deserialize(args.observer)
    action_space = DgxActionSpace.deserialize(args.action_space)
    observation_space = DgxObservationSpace.deserialize(args.observation_space)

    # Re-compile the python OPNIC wrapper according to the shape of the environment
    patch_opnic_wrapper(
        action_space,
        observation_space,
        args.path_to_python_wrapper,
        args.force_recompile_python_wrapper,
    )

    # Make sure Python can import the compiled wrapper
    sys.path.append(f"{args.path_to_python_wrapper}/wrapper/build/python")
    import opnic_wrapper

    # Build DGX environment
    dgx_env = DgxEnvironment(
        observer=observer,
        action_space=action_space,
        observation_space=observation_space,
        opnic_wrapper=opnic_wrapper,
        verbosity=args.verbosity,
    )

    # -----------------------------------------------------------------
    # Scalar optimization over the pi-pulse amplitude.
    # Bounds taken from your OPX action_space: (0., 1.99)
    # -----------------------------------------------------------------

    print("Starting pi-pulse amplitude optimization...")
    print(f"Using {args.shots_per_eval} shots per amplitude evaluation.")

    """
    names = ["additional_detuning_storage","relative_detuning_storage_readout","additional_detuning_qubit","qubit_amp","storage_amp","readout_amp","status"]
    bounds = [(-8.00,3.00),(-1.00,0.5),(-6.00,3.0),(1.0,2.0),(0.1,2.0),(0.05,2.0),(0., 1.99)]
    """

    bounds = [(-8.00,3.00),(-1.00,0.5),(-6.00,3.0),(1.0,2.0),(0.1,2.0),(0.05,2.0)]
    x0 = np.array([-1.7925, -0.108, 0.1773, 1.9969, 1.8341, 0.4558], dtype=float)

    # result = minimize(
    #     objective,
    #     x0 = x0,
    #     bounds=bounds,
    #     method='Nelder-Mead',
    #     args=(dgx_env,args.verbosity),
    #     options=dict(xatol=1e-3),
    # )
    result = minimize(
        objective,
        x0=x0,
        args=(dgx_env, args.verbosity),
        method="Powell",
        bounds=bounds,
        options=dict(
            xtol=5e-4,   # tune these to your shot noise level
            ftol=1e-3,
            maxiter=200000,
            maxfev=2000000,
        ),
    )

    best_vals = np.array(result.x, dtype=float)
    best_cost = float(result.fun)
    best_p_exc = estimate_excited_population(
        best_vals[0],
        best_vals[1],
        best_vals[2],
        best_vals[3],
        best_vals[4],
        best_vals[5],
        dgx_env,
    )


    print("\n=== Optimization result ===")
    print(f"additional_detuning_storage: {best_vals[0]}")
    print(f"relative_detuning_storage_readout: {best_vals[1]}")
    print(f"additional_detuning_qubit: {best_vals[2]}")
    print(f"qubit_amp: {best_vals[3]}")
    print(f"storage_amp: {best_vals[4]}")
    print(f"readout_amp: {best_vals[5]}")
    print(f"best_cost  ≈ {best_cost:.6f}")
    print(f"best_p_exc ≈ {best_p_exc:.6f}")


    print("nfev =", result.nfev)
    print("nit  =", result.nit)
    print("status:", result.status)
    print("message:", result.message)
    # -----------------------------------------------------------------
    # Final step: send one last action with status=0.0 to tell
    # the QUA while_ loop to stop (status_running < 0.5).
    # -----------------------------------------------------------------
    rewards = []
    for i in range(9):
        action = list(best_vals) + [1.0]  # 6 params + status
        obs_final, reward_final, terminated_final, _, info_final = dgx_env.step(action)
        rewards.append(float(reward_final))

    action_stop = list(best_vals) + [0.0]
    obs_final, reward_final, terminated_final, _, info_final = dgx_env.step(action_stop)
    rewards.append(float(reward_final))

    mean_reward = np.mean(np.array(rewards)/nb_average)
    print(f"Final reward with optimal parameters averaged over 10000 shots: {mean_reward*100:.1f}%")

    # Close environment cleanly
    dgx_env.close()
    print("Pi-pulse optimization script finished.")
