#!/usr/bin/env python

import sys
import argparse
import numpy as np
from scipy.optimize import minimize_scalar

from dgx_suite.dgx_environment import DgxEnvironment
from dgx_suite.dgx_parameter_space import DgxActionSpace, DgxObservationSpace
from dgx_suite.dgx_observer import Observer
from dgx_suite.opnic_utils import patch_opnic_wrapper


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
def estimate_excited_population(amp, dgx_env, num_shots_per_eval, verbosity=0):
    """
    For a given amplitude `amp`, run `num_shots_per_eval` single-shot experiments
    and return estimated excited-state population p_exc = mean(bit).

    Here, the bit is returned as `reward` from the environment, because the
    OPX-side observation was defined with is_reward=True.
    """
    rewards = np.empty(num_shots_per_eval, dtype=float)

    for i in range(num_shots_per_eval):
        # Action = [pi_pulse_amp, status_running]
        # status_running = 1.0 means "keep running" on the QUA side
        observation, reward, terminated, _, info = dgx_env.step([float(amp), 1.0])
        rewards[i] = float(reward)  # reward is the measured bit (0 or 1)

    p_exc = rewards.mean()

    if verbosity >= 2:
        print(f"amp={amp:.6f}, p_exc={p_exc:.4f}")

    return p_exc


def objective(amp, dgx_env, num_shots_per_eval, verbosity=0):
    """
    Objective for the scalar optimizer.

    For pi-pulse calibration we usually want to MAXIMIZE excited-state population p_exc.
    We achieve this by minimizing:

        cost(amp) = 1 - p_exc(amp)
    """
    p_exc = estimate_excited_population(amp, dgx_env, num_shots_per_eval, verbosity)

    # Maximize excitation -> minimize (1 - p_exc)
    cost = 1.0 - p_exc

    # If instead you want to minimize excitation, use:
    # cost = p_exc

    if verbosity >= 1:
        print(f"[objective] amp={amp:.6f}, p_exc={p_exc:.4f}, cost={cost:.4f}")
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
    bounds = (0.0, 1.99)

    print("Starting pi-pulse amplitude optimization...")
    print(f"Using {args.shots_per_eval} shots per amplitude evaluation.")

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method='bounded',
        args=(dgx_env, args.shots_per_eval, args.verbosity),
        options=dict(xatol=1e-3),
    )

    best_amp = float(result.x)
    best_cost = float(result.fun)
    best_p_exc = estimate_excited_population(
        best_amp, dgx_env, args.shots_per_eval, args.verbosity
    )

    print("\n=== Optimization result ===")
    print(f"best_amp   ≈ {best_amp:.6f}")
    print(f"best_cost  ≈ {best_cost:.6f}")
    print(f"best_p_exc ≈ {best_p_exc:.6f}")

    # -----------------------------------------------------------------
    # Final step: send one last action with status=0.0 to tell
    # the QUA while_ loop to stop (status_running < 0.5).
    # -----------------------------------------------------------------
    obs_final, reward_final, terminated_final, _, info_final = dgx_env.step(
        [best_amp, 0.0]
    )
    print(f"Final single-shot reward at best_amp: {float(reward_final)}")

    # Close environment cleanly
    dgx_env.close()
    print("Pi-pulse optimization script finished.")
