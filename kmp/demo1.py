#!/usr/bin/env python3

import logging
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle
from tqdm import tqdm
from scipy.interpolate import interp1d
from kmp import utils
from kmp.mixture import GaussianMixtureModel
from kmp.model import KMP
from matplotlib.widgets import Button, TextBox


class demo1:
    """KMP with HDF5-based 7D end-effector trajectory data, preserving GUI interaction."""

    def __init__(self) -> None:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s - %(levelname)s - %(message)s")
        self.__logger = logging.getLogger(__name__)

        # Set up the GUI
        self.fig, self.axs = plt.subplots(1, 4, figsize=(12, 6))
        self.fig.subplots_adjust(bottom=0.2)

        # GUI Buttons and TextBoxes
        axbtn = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        axC = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.btn = Button(axbtn, "Update")
        self.btn.on_clicked(self.update)
        self.Ctxt = TextBox(axC, "C", "5")  # Default GMM components

        # Enable clicking for waypoint adaptation
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.waypoints = None  # Stores added waypoints

        # Load and process the HDF5 dataset
        self.load_kmp_data()
        self.update(None)  # Initialize KMP

        # Keep GUI active
        plt.show(block=True)

    def onclick(self, event):
        """Handles user clicks to add waypoints dynamically for adaptation."""
        if event.inaxes is None:
            return  # Ignore clicks outside the plot

        x, y = event.xdata, event.ydata
        logging.info(f"Clicked at: x={x}, y={y}")

        if self.waypoints is None:
            self.waypoints = np.array([[x, y]])
        else:
            self.waypoints = np.vstack((self.waypoints, [x, y]))

        logging.debug(f"Updated waypoints: {self.waypoints.shape}")

        # Update the KMP model with the new waypoint
        self.apply_waypoints()
        self.plot_trajectories()

    def load_kmp_data(self, demo_file="keyboard_demo.h5", desired_length=1000):
        """Loads demonstrations from an HDF5 file and prepares 7D data for KMP training."""
        logging.info(f"Loading HDF5 data from {demo_file}...")

        with h5py.File(demo_file, "r") as f:
            demo_keys = list(f["data"].keys())  # Get all demo keys
            self.H = len(demo_keys)  # Number of demonstrations
            logging.info(f"Total demonstrations found: {self.H}")

        all_positions = []

        for demo_idx in tqdm(range(self.H), desc="Processing demonstrations"):
            with h5py.File(demo_file, "r") as f:
                states = np.array(f[f"data/demo_{demo_idx}/states"])  # (T, 7)

            # Extract full 7D state: (x, y, z, roll, pitch, yaw, gripper)
            ee_positions = states[:, :7]  # Now using full 7D states

            # Normalize time to [0, 1]
            timesteps = np.linspace(0, 1, len(ee_positions))

            # Interpolate trajectory to a fixed length
            interp_func = interp1d(
                timesteps, ee_positions, axis=0, kind="linear", fill_value="extrapolate"
            )
            new_timesteps = np.linspace(0, 1, desired_length)
            interpolated_positions = interp_func(new_timesteps)

            # Store trajectory (time, x, y, z, roll, pitch, yaw, gripper)
            trajectory = np.hstack(
                (new_timesteps.reshape(-1, 1), interpolated_positions))
            all_positions.append(trajectory)

        # Convert to NumPy array for KMP training
        self.data = np.array(all_positions)  # Shape (H, T, 8) [Time + 7D]
        np.save("kmp_training_data.npy", self.data)
        logging.info(
            f"Saved KMP training data as 'kmp_training_data.npy' with shape {self.data.shape}")

    def update(self, event):
        """Train KMP model using extracted HDF5 trajectory data (7D state)."""
        logging.info("Updating KMP model with 7D state...")

        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()
        self.axs[3].cla()
        self.waypoints = None

        # Extract time and full 7D state data
        self.N = self.data.shape[1]  # Number of time steps
        logging.debug(
            f"Time steps (N): {self.N}, Demonstrations (H): {self.H}")

        time = self.data[:, :, 0].T  # (N, H) → Transpose for correct shape
        pos = self.data[:, :, 1:].transpose(2, 0, 1)  # (H, N, 7) → (7, N, H)

        # Set time step
        dt = 0.01
        time_gmr = dt * np.arange(self.N).reshape(1, -1)

        # Train GMM and predict using GMR
        C = int(self.Ctxt.text)  # Number of Gaussian components
        self.gmm = GaussianMixtureModel(n_components=C, diag_reg_factor=1e-6)
        X = np.vstack((time.flatten(), pos.reshape(7, -1))).T

        logging.info("Training Gaussian Mixture Model (GMM)...")
        for _ in tqdm(range(1), desc="Fitting GMM"):
            self.gmm.fit(X)
        logging.info("GMM fit done.")

        self.mu, self.sigma = self.gmm.predict(time_gmr)
        logging.info("GMR done.")

        # Train KMP on extracted data
        self.kmp = KMP(l=0.5, alpha=40, sigma_f=200, time_driven_kernel=False)
        logging.info("Training Kernelized Movement Primitives (KMP)...")
        for _ in tqdm(range(1), desc="Fitting KMP"):
            self.kmp.fit(time_gmr, self.mu, self.sigma)
        logging.info("KMP fit done.")

        for _ in tqdm(range(1), desc="Predicting KMP trajectory"):
            self.mu_kmp, self.sigma_kmp = self.kmp.predict(time_gmr)
        logging.info("KMP predict done.")

        # Save trained KMP model
        self.save_kmp_model()
        self.save_trajectory_to_h5()
        plt.pause(0.01)  # Allow GUI refresh

    def save_kmp_model(self, filename="kmp_model.pkl"):
        """Save the trained KMP model."""
        logging.info(f"Saving KMP model to {filename}...")
        kmp_data = {"mu_kmp": self.mu_kmp, "sigma_kmp": self.sigma_kmp}
        with open(filename, "wb") as f:
            pickle.dump(kmp_data, f)
        logging.info(f"KMP model saved successfully!")

    def save_trajectory_to_h5(self, filename="keyboard_demo_train.h5"):
        """Save generated KMP trajectory (7D) to HDF5 for Robosuite playback."""
        logging.info(f"Saving 7D trajectory to {filename}...")

        with h5py.File(filename, "w") as f:
            demo_group = f.create_group("data/demo_0")
            actions = np.diff(self.mu_kmp.T, axis=0)  # (T-1, 7)
            demo_group.create_dataset("actions", data=actions)
            states = self.mu_kmp.T  # (T, 7)
            demo_group.create_dataset("states", data=states)

        logging.info(f"7D Trajectory saved successfully to {filename}!")


if __name__ == "__main__":
    demo1()
