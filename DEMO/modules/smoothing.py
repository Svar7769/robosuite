import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



def smooth_trajectory(data, num_points=1000):
    timesteps = np.linspace(0, 1, len(data))
    interp_func = interp1d(timesteps, data, axis=0,
                           kind="linear", fill_value="extrapolate")
    new_timesteps = np.linspace(0, 1, num_points)
    return interp_func(new_timesteps)


def process_and_save_smoothed_data(h5_file_path, save_directory):

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    smoothed_file_path = os.path.join(
        save_directory, "recorded_demo_smoothed.h5")

    with h5py.File(h5_file_path, "r") as f, h5py.File(smoothed_file_path, "w") as new_h5:
        for demo_name in f.keys(): 
            print(f"Processing {demo_name}...")

            demo_group = f[demo_name]
            smoothed_group = new_h5.create_group(f"smoothed_{demo_name}")

            for dataset_name in demo_group.keys():
                dataset = np.array(demo_group[dataset_name])

                if dataset.ndim > 2:
                    dataset = dataset.reshape(dataset.shape[0], -1)

                smoothed_dataset = smooth_trajectory(dataset)
                smoothed_group.create_dataset(
                    dataset_name, data=smoothed_dataset)

    print(f"Smoothed data saved to: {smoothed_file_path}")


h5_directory = r"robosuite/DEMO/recorded_demo/recorded_demo.h5"
save_directory = r"robosuite/DEMO/smoothed_demo"

process_and_save_smoothed_data(h5_directory, save_directory)