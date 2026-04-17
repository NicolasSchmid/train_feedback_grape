import nvtx
import sys
import numpy as np
import torch
import argparse
import os

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from dgx_suite.dgx_environment import DgxEnvironment
from dgx_suite.dgx_parameter_space import DgxActionSpace, DgxObservationSpace
from dgx_suite.dgx_observer import Observer
from dgx_suite.opnic_utils import patch_opnic_wrapper

from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

dtype = torch.float64

# --- REVERTED TO EXACT ORIGINAL PPO PARSER TO KEEP SPAWN() HAPPY ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an agent in a DGX environment.")
    parser.add_argument('--total_timesteps', type=int, default=700)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--learning_starts', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--sigma_end_as_scale', type=float, default=1.)
    parser.add_argument('--observer', type=str)
    parser.add_argument('--action_space', type=str)
    parser.add_argument('--observation_space', type=str)
    parser.add_argument('--path_to_python_wrapper', type=str)
    parser.add_argument('--force_recompile_python_wrapper', action=argparse.BooleanOptionalAction)
    return parser.parse_args()

def evaluate_pulse(action_numpy, env, num_shots):
    total_reward = 0.0
    for s in range(num_shots):
        print(f"    -> [Shot {s+1}/{num_shots}] Waiting for OPX...", end="", flush=True)
        obs, reward, done, _, _ = env.step(action_numpy)
        total_reward += reward
        print(f" Done! Reward = {reward:.4f}", flush=True)
    return total_reward / num_shots

if __name__ == "__main__":
    args = parse_arguments()
    
    # Force CPU to avoid CUDA PyTorch missing package errors
    device = torch.device("cpu") 

    # =========================================================
    # SYNC BLOCK: MATCH THESE VALUES IN YOUR JUPYTER NOTEBOOK
    # =========================================================
    BO_INIT_SAMPLES = 15
    BO_ITERATIONS = 70
    NB_REP_END = 200
    # =========================================================

    # 1. Environment Setup
    observer = Observer.deserialize(args.observer)
    action_space = DgxActionSpace.deserialize(args.action_space)
    observation_space = DgxObservationSpace.deserialize(args.observation_space)

    patch_opnic_wrapper(action_space, observation_space, args.path_to_python_wrapper, args.force_recompile_python_wrapper)
    sys.path.append(f"{args.path_to_python_wrapper}/wrapper/build/python")
    import opnic_wrapper

    dgx_env = DgxEnvironment(
        observer=observer, action_space=action_space, observation_space=observation_space,
        opnic_wrapper=opnic_wrapper, verbosity=args.verbosity
    )

    # Extract bounds manually
    lows, highs = [], []
    for param in action_space.parameters:
        lows.append(param.bounds[0])
        highs.append(param.bounds[1])
    bounds = torch.tensor([lows, highs], dtype=dtype, device=device)
    dim = len(lows)

    print(f"\n--- 1. Warm-Start Initialization ({BO_INIT_SAMPLES} samples) ---", flush=True)
    
    # 1. Build the absolute path to the .npz file located in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bundle_path = os.path.join(script_dir, "bspline_bundle_fock1_fid_9938_Tns_1488_20260113_19_23.npz")

    try:
        # 2. Load the bundle directly
        bundle = np.load(bundle_path, allow_pickle=False)
        coeffs = bundle["coeffs"]  # Shape (4, 9)

        # 3. Apply the scaling logic so the parameters fit within the (-1.99, 1.99) action space bounds
        scaling_factor_qubit = np.max(np.abs(coeffs[:2, :])) / np.sqrt(2)
        scaling_factor_storage = np.max(np.abs(coeffs[2:, :])) / np.sqrt(2)

        grape_qubit_I = coeffs[0].astype(float) / scaling_factor_qubit
        grape_qubit_Q = -coeffs[1].astype(float) / scaling_factor_qubit
        grape_cav_I   = coeffs[2].astype(float) / scaling_factor_storage
        grape_cav_Q   = -coeffs[3].astype(float) / scaling_factor_storage

        # 4. Pack all 36 coefficients into a single flattened array
        initial_guess = np.concatenate([grape_qubit_I, grape_qubit_Q, grape_cav_I, grape_cav_Q])
        print(f"Successfully loaded and scaled 36-parameter GRAPE guess from {bundle_path}", flush=True)

    except Exception as e:
        print(f"Warning: Could not load .npz file. Error: {e}. Falling back to zeros.", flush=True)
        initial_guess = np.zeros(dim)
        
    initial_guess_tensor = torch.tensor(initial_guess, dtype=dtype, device=device)

    # Create the initial exploration dataset
    train_X = [initial_guess_tensor]
    for _ in range(BO_INIT_SAMPLES - 1):
        noise = torch.randn(dim, dtype=dtype, device=device) * 0.05
        perturbed = torch.clamp(initial_guess_tensor + noise, bounds[0], bounds[1])
        train_X.append(perturbed)
    
    train_X = torch.stack(train_X)
    train_Y = []
    init_shots = 5
    
    for i in range(BO_INIT_SAMPLES):
        print(f"\n[Init {i+1}/{BO_INIT_SAMPLES}] Evaluating pulse on OPX...", flush=True)
        y_val = evaluate_pulse(train_X[i].cpu().numpy(), dgx_env, init_shots)
        train_Y.append([y_val])
        
    train_Y = torch.tensor(train_Y, dtype=dtype, device=device)

    print("\n--- 2. Bayesian Optimization ---", flush=True)
    with nvtx.annotate("BoTorch Optimize"):
        for iteration in range(BO_ITERATIONS):
            print(f"\n>>>>> Iteration {iteration+1}/{BO_ITERATIONS} <<<<<", flush=True)
            
            # Dynamic shots logic
            if iteration < BO_ITERATIONS // 3: current_shots = 10
            elif iteration < 2 * (BO_ITERATIONS // 3): current_shots = 20
            else: current_shots = 50
            
            print("  [Math] Fitting Gaussian Process Model...", flush=True)
            # On indique à BoTorch les bornes réelles pour qu'il normalise en [0, 1] en interne
            # et on standardise les récompenses (moyenne 0, variance 1)
            gp_model = SingleTaskGP(
                train_X, 
                train_Y,
                input_transform=Normalize(d=dim, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            fit_gpytorch_mll(mll)

            print("  [Math] Optimizing Acquisition Function...", flush=True)
            UCB = UpperConfidenceBound(gp_model, beta=0.1)
            candidates, _ = optimize_acqf(
                acq_function=UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=50,
            )
            new_x = candidates[0]
            action_numpy = new_x.cpu().numpy()

            print(f"  [Hardware] Evaluating new pulse on OPX ({current_shots} shots)...", flush=True)
            new_y_val = evaluate_pulse(action_numpy, dgx_env, current_shots)
            new_y = torch.tensor([[new_y_val]], dtype=dtype, device=device)

            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_y])

            print(f"  --> Result: Reward {new_y_val:.4f}", flush=True)

    print("\n--- 3. Final Evaluation ---", flush=True)
    best_idx = train_Y.argmax()
    best_action = train_X[best_idx].cpu().numpy()
    
    print(f"Evaluating the best found action with {NB_REP_END} shots...", flush=True)
    final_reward = evaluate_pulse(best_action, dgx_env, NB_REP_END)
    print(f"Real reward of best action: {final_reward:.4f}", flush=True)

    dgx_env.close()