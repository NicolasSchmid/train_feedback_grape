import nvtx
import sys
import numpy as np

from stable_baselines3 import PPO

from dgx_suite.dgx_environment import DgxEnvironment
from dgx_suite.dgx_parameter_space import DgxActionSpace, DgxObservationSpace
from dgx_suite.dgx_observer import Observer
from dgx_suite.opnic_utils import patch_opnic_wrapper

import argparse

from stable_baselines3.common.callbacks import BaseCallback
SEED = 5

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an agent in a DGX environment.")
    parser.add_argument('--total_timesteps', type=int, default=700, help='Total number of timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the TD3 agent')
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2], help='Verbosity level (0: no output, 1: info, 2: debug)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use for training (cpu or cuda)')
    parser.add_argument('--learning_starts', type=int, default=100, help='Number of steps of exploration before learning starts')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of samples from the replay buffer used for each gradient update')
    parser.add_argument('--train_freq', type=int, default=1, help='Determines how often the model updates its parameters based on the number of steps in the environment')
    parser.add_argument('--sigma_end_as_scale', type=float, default=1., help='Ending action noise std. dev is sigma_start * sigma_end_as_scale')
    parser.add_argument('--observer', type=str, help='Serialized base64 message packed string of a Observer instance')
    parser.add_argument('--action_space', type=str, help='Serialized base64 message packed string of a DgxActionSpace instance')
    parser.add_argument('--observation_space', type=str, help='Serialized base64 message packed string of a DgxActionSpace instance')
    parser.add_argument('--path_to_python_wrapper', type=str, help='Path to python-wrapper source code.')
    parser.add_argument('--force_recompile_python_wrapper', action=argparse.BooleanOptionalAction, help='Whether to force-recompile the python wrapper')

    return parser.parse_args()



if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Deserialize environment-shape objects
    observer = Observer.deserialize(args.observer)
    action_space = DgxActionSpace.deserialize(args.action_space)
    observation_space = DgxObservationSpace.deserialize(args.observation_space)

    # Re-compile the python OPNIC wrapper according to the shape of the environment
    patch_opnic_wrapper(action_space, observation_space, args.path_to_python_wrapper, args.force_recompile_python_wrapper)

    sys.path.append(f"{args.path_to_python_wrapper}/wrapper/build/python")
    import opnic_wrapper

    # Create environment with ability to exchange information over DGX
    dgx_env = DgxEnvironment(
        observer=observer,
        action_space=action_space,
        observation_space=observation_space,
        opnic_wrapper=opnic_wrapper,
        verbosity=args.verbosity
    )
    
    class EntropyDecayCallback(BaseCallback):
        def __init__(self, initial_ent_coef=0.1, final_ent_coef=0.001, decay_rate=0.999, verbose=0):
            super().__init__(verbose)
            self.initial = initial_ent_coef
            self.final = final_ent_coef
            self.decay = decay_rate

        def _on_step(self) -> bool:
            # Update ent_coef
            new_ent = max(self.final, self.model.ent_coef * self.decay)
            self.model.ent_coef = new_ent
            return True
        
    # Modify the size of both the policy and critic agents. These are basically fully connected Neural Networks of size input->n->n->output
    # Hence they have roughly nxn + a few parameters 
    policy_kwargs = dict(
    net_arch=[2, 2]   # this is enough for a 2-parameter circuit
    )
    # Define the reinforcement learning agent
    model = PPO(
        policy='MlpPolicy',
        env=dgx_env,
        gamma=1,
        seed=SEED,
        policy_kwargs=policy_kwargs,
        # action_noise=action_noise,
        learning_rate=5e-4, # how intensely the model should update its network per reward
        batch_size=64, #should be a multiple of n_steps
        # train_freq=args.train_freq,
        n_steps=2, # how many rewards should be buffered before updating the network
        # learning_starts=args.learning_starts,
        verbose=args.verbosity,
        device=args.device,
        ent_coef=0.01
    )

    # Train the model. Here is the experiment actually executed
    with nvtx.annotate("Model Learn"):
        model.learn(total_timesteps=args.total_timesteps,callback=EntropyDecayCallback(
        initial_ent_coef=0.1,
        final_ent_coef=0.0001,
        decay_rate=0.999
    ))

    # Find the model prediction of the action which leads to the wished target state
    action, _ = model.predict(observation_space.target_state, deterministic=True)
    nb_rep_end = 200 #put the same on the client side !!!
    num_shots=1 # number of shots per observation,put the same on the client side!!!
    print(f"Evaluate action: {action}, on {num_shots *nb_rep_end} shots")
    avg_reward = 0
    for i in range(nb_rep_end):
        obs, reward, done, _, _ = dgx_env.step(action)
        avg_reward+=reward
    avg_reward = avg_reward/nb_rep_end

    print(f"Sampled final action: {action}, got an averaged reward of: {avg_reward}")
    dgx_env.close()