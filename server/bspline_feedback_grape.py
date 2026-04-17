import sys
import os
import argparse
import numpy as np
import traceback

# --- ADD THESE TWO LINES TO FORCE CPU ---
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ----------------------------------------

# 1. Silent Crash Logger
def crash_logger(exc_type, exc_value, exc_tb):
    crash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fatal_crash.txt")
    with open(crash_path, "w") as f:
        f.write("FATAL CRASH LOG:\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = crash_logger

# 2. Imports
from dgx_suite.dgx_environment import DgxEnvironment
from dgx_suite.dgx_parameter_space import DgxActionSpace, DgxObservationSpace
from dgx_suite.dgx_observer import Observer
from dgx_suite.opnic_utils import patch_opnic_wrapper

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint as ocp


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

if __name__ == "__main__":
    args = parse_arguments()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load B-Spline Initial Guess
    bundle_path = os.path.join(script_dir, "bspline_bundle_fock1_fid_9938_Tns_1488_20260113_19_23.npz")
    try:
        bundle = np.load(bundle_path, allow_pickle=False)
        coeffs = bundle["coeffs"]
        scaling_factor_qubit = np.max(np.abs(coeffs[:2, :])) / np.sqrt(2)
        scaling_factor_storage = np.max(np.abs(coeffs[2:, :])) / np.sqrt(2)

        grape_qubit_I = coeffs[0].astype(float) / scaling_factor_qubit
        grape_qubit_Q = -coeffs[1].astype(float) / scaling_factor_qubit
        grape_cav_I   = coeffs[2].astype(float) / scaling_factor_storage
        grape_cav_Q   = -coeffs[3].astype(float) / scaling_factor_storage

        initial_guess = np.concatenate([grape_qubit_I, grape_qubit_Q, grape_cav_I, grape_cav_Q])
    except Exception:
        initial_guess = np.zeros(36)

    # Setup Environment and Wrapper
    observer = Observer.deserialize(args.observer)
    action_space = DgxActionSpace.deserialize(args.action_space)
    observation_space = DgxObservationSpace.deserialize(args.observation_space)
    
    patch_opnic_wrapper(action_space, observation_space, args.path_to_python_wrapper, args.force_recompile_python_wrapper)

    wrapper_build_path = f"{args.path_to_python_wrapper}/wrapper/build/python"
    if wrapper_build_path not in sys.path:
        sys.path.append(wrapper_build_path)
        
    import opnic_wrapper 

    dgx_env = DgxEnvironment(
        observer=observer, action_space=action_space, observation_space=observation_space,
        opnic_wrapper=opnic_wrapper, verbosity=args.verbosity
    )
    
    # Handshake with OPX
    dgx_env.reset()



    # Neural Network Parameters
    idling_time_us=25
    Nb_feedback_GRAPE_steps = 30
    num_bspln = 9
    inp_shape = (1,)
    rec_features = 36
    dense_features = [54, 72, 54]
    load_fld = os.path.join(script_dir, "trained_nn25")

    def load_flax_state(fld, state): 
        abs_path = os.path.abspath(fld)
        orbax_path = ocp.test_utils.epath.gpath.PosixGPath(abs_path) / "1"
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        return ckptr.restore(orbax_path, args=ocp.args.StandardRestore(state))

    def init_carry(key, rec_features, inp_shape):
        return nn.GRUCell(features=rec_features).initialize_carry(key, inp_shape)

    def create_flax_state(key, rec_features, dense_features, inp_shape):
        class Model(nn.Module):
            @nn.compact
            def __call__(self, carry, x):
                carry, x = nn.GRUCell(features=rec_features)(carry, x)
                for ndf in dense_features:
                    x = jax.nn.relu(nn.Dense(ndf)(x))
                return carry, nn.Dense(4 * num_bspln)(x).reshape(4, num_bspln)

        model = Model()
        fake_key = jax.random.key(0)
        fake_carry = init_carry(fake_key, rec_features, inp_shape)
        fake_inp = jnp.ones(inp_shape)
        params = model.init(key, fake_carry, fake_inp)["params"]
        return TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.001))

    # Load weights
    flax_state = load_flax_state(load_fld, create_flax_state(jax.random.key(0), rec_features, dense_features, inp_shape))

    @jax.jit
    def interrogate_nn(carry, meas_outcome):
        inp = jnp.array(meas_outcome).reshape((1,))
        return flax_state.apply_fn({'params': flax_state.params}, carry, inp)
    






    # Warmup compilation & Initial evaluation
    carry = init_carry(jax.random.key(0), rec_features, inp_shape)
    print("carry dtype",carry.dtype)


    send_packet = dgx_env.opnic_wrapper.send_packet
    wait_packets = dgx_env.opnic_wrapper.wait_for_packets
    read_packet = dgx_env.opnic_wrapper.read_packet

    out_handle = dgx_env.OUTGOING_PACKET_STREAM_HANDLE
    in_handle = dgx_env.INCOMING_PACKET_STREAM_HANDLE
    OutgoingPacket = dgx_env.opnic_wrapper.OutgoingPacket






    for i in range(3): #warm up communication and neural network calls
        dummy_carry, _ = interrogate_nn(carry, 1.0)
        action_numpy = np.asarray(-initial_guess, dtype=np.float64).ravel()
        packet = OutgoingPacket(*action_numpy)
        send_packet(out_handle, packet)
        wait_packets(in_handle, 1)
        packet = read_packet(in_handle, 0)
        current_meas = packet.reward[0]

        dummy_carry, _ = interrogate_nn(carry, -1.0)
        action_numpy = np.asarray(initial_guess, dtype=np.float64).ravel()
        packet = OutgoingPacket(*action_numpy)
        send_packet(out_handle, packet)
        wait_packets(in_handle, 1)
        packet = read_packet(in_handle, 0)
        current_meas = packet.reward[0]


    dummy_action = np.asarray(initial_guess, dtype=np.float64).ravel()
    packet = OutgoingPacket(*dummy_action)
    send_packet(out_handle, packet)
    wait_packets(in_handle, 1)
    packet = read_packet(in_handle, 0) #first measurement after state preparation
    current_meas = packet.reward[0]


    # --- MAIN FEEDBACK GRAPE LOOP (Active NN Mode) ---
    for iteration in range(Nb_feedback_GRAPE_steps-1):
        
        # 1. NN inference
        carry, ctrl = interrogate_nn(carry, current_meas)
        # print("ctrl dtype",ctrl.dtype)
        action_numpy_reshaped = np.asarray(ctrl, dtype=np.float64).ravel()
        packet = OutgoingPacket(*action_numpy_reshaped)
        send_packet(out_handle, packet)
        wait_packets(in_handle, 1)
        packet = read_packet(in_handle, 0)
        current_meas = packet.reward[0]

        
    carry, ctrl = interrogate_nn(carry, current_meas)
    action_numpy = np.asarray(ctrl, dtype=np.float64).ravel()
    packet = OutgoingPacket(*action_numpy)
    send_packet(out_handle, packet)
    # Cleanup
    dgx_env.close()
    print("DGX Server Loop Complete.")