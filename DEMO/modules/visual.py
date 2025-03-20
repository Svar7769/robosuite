import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_trajectories(h5_file_path, use_smoothed=False):

    if not os.path.exists(h5_file_path):
        print(f"Error: {h5_file_path} does not exist.")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ["red", "blue",]
    colors1 = ["green","purple"]

    with h5py.File(h5_file_path, "r") as f:
        demo_names = [name for name in f.keys() if name.startswith(
            "demo") or name.startswith("smoothed_demo")]

        for idx, demo_name in enumerate(demo_names):
            print(f"Processing {demo_name}...")

            data_group = f[demo_name]

            # Load datasets
            robot_states = np.array(data_group["robot_states"])
            cube_states = np.array(data_group["cube_states"])

            if robot_states.ndim > 2:
                robot_states = robot_states.reshape(-1, 3)
            if cube_states.ndim > 2:
                cube_states = cube_states.reshape(-1, 3)

            color = colors[idx % len(colors)]
            color1 = colors1[idx % len(colors1)]

            # Plot end-effector trajectory
            ax.scatter(robot_states[:, 0], robot_states[:, 1], robot_states[:, 2],
                       color=color1, s=15, label=f"EE Trajectory ({demo_name})")

            # Mark start and end positions
            ax.scatter(robot_states[0, 0], robot_states[0, 1], robot_states[0, 2],
                       color="black", s=30, label=f"Start EE ({demo_name})")
            ax.scatter(robot_states[-1, 0], robot_states[-1, 1], robot_states[-1, 2],
                       color="red", s=30, label=f"End EE ({demo_name})")

            # Plot cube positions
            ax.scatter(cube_states[:, 0], cube_states[:, 1], cube_states[:, 2],
                       color=color, s=15, label=f"Cube Positions ({demo_name})")

    # Labels & Equal Axis Scaling
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(
        "3D Scatter Trajectories of Robot End-Effector and Cube Positions")

    # Ensure equal aspect ratio for all axes
    all_x, all_y, all_z = np.array([]), np.array([]), np.array([])

    with h5py.File(h5_file_path, "r") as f:
        for demo_name in demo_names:
            robot_states = np.array(f[demo_name]["robot_states"])
            if robot_states.ndim > 2:
                robot_states = robot_states.reshape(-1, 3)

            all_x = np.append(all_x, robot_states[:, 0])
            all_y = np.append(all_y, robot_states[:, 1])
            all_z = np.append(all_z, robot_states[:, 2])

    max_range = np.array([all_x.max() - all_x.min(),
                          all_y.max() - all_y.min(),
                          all_z.max() - all_z.min()]).max() / 2.0

    mid_x = (all_x.max() + all_x.min()) * 0.5
    mid_y = (all_y.max() + all_y.min()) * 0.5
    mid_z = (all_z.max() + all_z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.show()


if __name__ == "__main__":
    h5_file_path = r"robosuite/DEMO/recorded_demo/recorded_demo.h5"
    smoothed_h5_file_path = r"robosuite/DEMO/smoothed_demo/recorded_demo_smoothed.h5"

    print("\nPlotting Raw Data...\n")
    plot_trajectories(h5_file_path, use_smoothed=False)

    print("\nPlotting Smoothed Data...\n")
    plot_trajectories(smoothed_h5_file_path, use_smoothed=True)
