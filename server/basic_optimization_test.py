 
import nvtx
import sys
import numpy as np

from dgx_suite.dgx_environment import DgxEnvironment
from dgx_suite.dgx_parameter_space import DgxActionSpace, DgxObservationSpace
from dgx_suite.dgx_observer import Observer
from dgx_suite.opnic_utils import patch_opnic_wrapper
from scipy.optimize import minimize

import argparse

import numpy as np
import nvtx

from typing import Optional

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, ActionNoise
from stable_baselines3.common.type_aliases import TrainFreq, RolloutReturn
from stable_baselines3.common.vec_env import VecEnv

# Instantiate the agent
class TD3Profiled(TD3):
    @nvtx.annotate("TD3 Rollout")
    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        return super(TD3Profiled, self).collect_rollouts(env, callback, train_freq, replay_buffer, action_noise,
                                                         learning_starts, log_interval)

    @nvtx.annotate("TD3 Training")
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        return super(TD3Profiled, self).train(gradient_steps, batch_size)


class DecayingNormalActionNoise(NormalActionNoise):
    def __init__(self, sigma_start, sigma_end_as_scale, total_timesteps):
        super().__init__(mean=np.zeros_like(sigma_start), sigma=sigma_start)
        self.sigma_start = sigma_start
        self.sigma_end = sigma_start * sigma_end_as_scale
        self.total_timesteps = total_timesteps
        self.decay_rate = (sigma_start - self.sigma_end) / total_timesteps

    def __call__(self):
        # Linearly decay sigma over time
        self._sigma = np.maximum(self.sigma_end, self._sigma - self.decay_rate)
        return super().__call__()




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
    
    dgx_env = DgxEnvironment(
        observer=observer,
        action_space=action_space,
        observation_space=observation_space,
        opnic_wrapper=opnic_wrapper,
        verbosity=args.verbosity
    )

    def optimize_x_function(x):
        observation,reward, terminated, _, info =  dgx_env.step([float(x),1])
        return reward

    
    # some_starting_t1 = 10e-6
    # # wait_time = estimate_wait_time(some_starting_t1)  # Bayesian  estimation
    # # Option #1.
    # # -> Use DGXEnvironment by calling its methods manually
    # for i in range(args.total_timesteps+1):
    #     observation, reward, terminated, _, info = dgx_env.step([0.9])
        # wait_time = estimate_wait_time(new_t1_measurement)
    bounds = [(0, 2)]
    res = minimize(optimize_x_function,1,method = 'SLSQP',bounds=bounds)
    observation,reward, terminated, _, info =  dgx_env.step([float(res.x),0.2])
    print("res.x = ",res.x)
    dgx_env.close()
    # print(f"Sampled final action: {0.9}, got reward: {reward}")
    # Option #2.
    # -> Use the opnic_wrapper methods directly, as if you were writing the C++ code.
    # outgoing_stream = opnic_wrapper.configure_stream(1, opnic_wrapper.Direction_OUTGOING)
    # incoming_stream = opnic_wrapper.configure_stream(1, opnic_wrapper.Direction_INCOMING)
    # for i in range(num_steps):
    #     opnic_wrapper.send_packet(outgoing_stream, )