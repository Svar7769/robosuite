import h5py
import numpy as np
from scipy.interpolate import interp1d


def load_kmp_data(demo_file="keyboard_demo.h5", desired_length=100):
    """Loads demonstrations from an HDF5 file and prepares them for KMP training."""

    with h5py.File(demo_file, "r") as f:
        demo_keys = list(f["data"].keys())  # Get all demo keys
        num_demos = len(demo_keys)  # Count total demonstrations
        print(f"Total demonstrations in file: {num_demos}")

    all_positions = []

    for demo_idx in range(num_demos):
        with h5py.File(demo_file, "r") as f:
            # Load state data
            states = np.array(f[f"data/demo_{demo_idx}/states"])

        # Extract end-effector positions (assuming x, y, z are the first 3 elements)
        ee_positions = states[:, :3]

        # Normalize time to [0, 1]
        timesteps = np.linspace(0, 1, len(ee_positions))

        # Interpolate trajectory to a fixed length
        interp_func = interp1d(timesteps, ee_positions,
                               axis=0, kind="linear", fill_value="extrapolate")
        new_timesteps = np.linspace(0, 1, desired_length)
        interpolated_positions = interp_func(new_timesteps)

        # Store trajectory (time, x, y, z)
        trajectory = np.hstack(
            (new_timesteps.reshape(-1, 1), interpolated_positions))
        all_positions.append(trajectory)

    # Convert to NumPy array and save for KMP training
    all_positions = np.array(all_positions)
    np.save("kmp_training_data.npy", all_positions)
    print("Saved KMP training data as kmp_training_data.npy")

    return all_positions
