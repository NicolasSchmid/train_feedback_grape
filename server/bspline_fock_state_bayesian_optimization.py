import nvtx
import sys
import numpy as np
import torch
import argparse

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from dgx_suite.dgx_environment import DgxEnvironment
from dgx_suite.dgx_parameter_space import DgxActionSpace, DgxObservationSpace
from dgx_suite.dgx_observer import Observer
from dgx_suite.opnic_utils import patch_opnic_wrapper

dtype = torch.float64

def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimize B-Splines using BoTorch.")
    parser.add_argument('--bo_iterations', type=int, default=50)
    parser.add_argument('--init_samples', type=int, default=15)
    parser.add_argument('--nb_rep_end', type=int, default=200)
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--observer', type=str)
    parser.add_argument('--action_space', type=str)
    parser.add_argument('--observation_space', type=str)
    parser.add_argument('--initial_guess_csv', type=str, default="", help="Comma separated GRAPE coefficients")
    parser.add_argument('--path_to_python_wrapper', type=str)
    parser.add_argument('--force_recompile_python_wrapper', action=argparse.BooleanOptionalAction)


    args, unknown = parser.parse_known_args()
    return args



def evaluate_pulse(action_numpy, env, num_shots):
    total_reward = 0.0
    for _ in range(num_shots):
        obs, reward, done, _, _ = env.step(action_numpy)
        total_reward += reward
    return total_reward / num_shots



if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device(args.device)

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

    # Extract bounds from the DgxActionSpace parameters manually
    lows = []
    highs = []
    for param in action_space.parameters:
        lows.append(param.bounds[0])
        highs.append(param.bounds[1])
        
    bounds = torch.tensor([lows, highs], dtype=dtype, device=device)
    dim = len(lows)

    print(f"--- 1. Warm-Start Initialization ({args.init_samples} samples) ---")
    
    # Safely parse the GRAPE initial guess
    clean_csv = [x.strip() for x in args.initial_guess_csv.split(',') if x.strip()]
    
    if len(clean_csv) == dim:
        initial_guess = np.array([float(x) for x in clean_csv])
    else:
        print(f"Warning: Expected {dim} parameters but got {len(clean_csv)}. Falling back to zeros for initial guess.")
        initial_guess = np.zeros(dim)
        
    initial_guess_tensor = torch.tensor(initial_guess, dtype=dtype, device=device)

    # Create the initial exploration dataset
    # 1st sample = EXACT GRAPE pulse
    # Remaining samples = GRAPE pulse + small random Gaussian noise (std dev = 0.05) to explore locally
    train_X = [initial_guess_tensor]
    for _ in range(args.init_samples - 1):
        noise = torch.randn(dim, dtype=dtype, device=device) * 0.05
        perturbed = torch.clamp(initial_guess_tensor + noise, bounds[0], bounds[1])
        train_X.append(perturbed)
    
    train_X = torch.stack(train_X)
    
    train_Y = []
    init_shots = 5
    for i in range(args.init_samples):
        y_val = evaluate_pulse(train_X[i].cpu().numpy(), dgx_env, init_shots)
        train_Y.append([y_val])
        print(f"Init {i+1}/{args.init_samples} evaluated. Reward: {y_val:.4f}")
        
    train_Y = torch.tensor(train_Y, dtype=dtype, device=device)

    print("--- 2. Bayesian Optimization ---")
    with nvtx.annotate("BoTorch Optimize"):
        for iteration in range(args.bo_iterations):
            # Dynamic shots logic
            if iteration < args.bo_iterations // 3: current_shots = 10
            elif iteration < 2 * (args.bo_iterations // 3): current_shots = 20
            else: current_shots = 50
            
            # High-dimensional GP Model
            gp_model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            fit_gpytorch_mll(mll)

            # UCB with a slight beta for balanced exploration
            UCB = UpperConfidenceBound(gp_model, beta=0.1)

            candidates, _ = optimize_acqf(
                acq_function=UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=50,
            )
            new_x = candidates[0]
            action_numpy = new_x.cpu().numpy()

            new_y_val = evaluate_pulse(action_numpy, dgx_env, current_shots)
            new_y = torch.tensor([[new_y_val]], dtype=dtype, device=device)

            train_X = torch.cat([train_X, candidates])
            train_Y = torch.cat([train_Y, new_y])

            print(f"Iter {iteration+1:02d}/{args.bo_iterations} | Shots: {current_shots:02d} | Reward: {new_y_val:.4f}")

    print("\n--- 3. Final Evaluation ---")
    best_idx = train_Y.argmax()
    best_action = train_X[best_idx].cpu().numpy()
    
    final_reward = evaluate_pulse(best_action, dgx_env, args.nb_rep_end)
    print(f"Real reward of best action (over {args.nb_rep_end} shots): {final_reward:.4f}")

    dgx_env.close()