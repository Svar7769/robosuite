import h5py
import robosuite as suite
import numpy as np
import time
import pygame
import os
import cv2
import sys


def replay_demo(use_smoothed=False, save_video=True):
    replay_choice = input(
        "Do you want to replay the demonstration? (yes/no): ").strip().lower()
    if replay_choice != "yes":
        print("Replay canceled. Exiting...")
        sys.exit(0)

    pygame.init()
    screen = pygame.display.set_mode((300, 300))

    # Set the HDF5 file path
    h5_file_path = (
        r"robosuite/DEMO/smoothed_demo/recorded_demo_smoothed.h5"
        if use_smoothed else
        r"robosuite/DEMO/recorded_demo/recorded_demo.h5"
    )

    if not os.path.exists(h5_file_path):
        print(f"Error: {h5_file_path} not found.")
        return

    dataset_group = "smoothed_demo_" if use_smoothed else "demo_"

    # Load the latest demo
    with h5py.File(h5_file_path, "r") as f:
        demo_names = sorted(
            [name for name in f.keys() if name.startswith(dataset_group)])
        if not demo_names:
            print("No demonstration data found in the file.")
            return

        latest_demo = demo_names[-1]  # Use the latest recorded demo
        print(f"Using demonstration: {latest_demo}")

        demo_data = f[latest_demo]
        actions = np.array(demo_data["actions"])
        joint_angles = np.array(demo_data["joint_angles"])
        cube_states = np.array(demo_data["cube_states"])

        num_steps = min(len(actions), len(joint_angles), len(cube_states))
        print(f"Loaded demonstration with {num_steps} steps.")

    # Initialize environment
    env = suite.make(
        env_name="Lift",
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=20,
        hard_reset=False,
        camera_names="frontview",
        camera_heights=512,
        camera_widths=512
    )

    # Set up video recording
    if save_video:
        video_filename = "simulation_replay.mp4"
        video_fps = 20
        video_size = (512, 512)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_out = cv2.VideoWriter(
            video_filename, fourcc, video_fps, video_size)

    env.reset()

    expected_qpos = env.sim.model.nq
    expected_qvel = env.sim.model.nv

    # Set initial state
    env.sim.data.qpos[:] = joint_angles[0][:expected_qpos]
    env.sim.data.qvel[:] = joint_angles[0][expected_qpos:]
    env.sim.forward()

    print("Press ESC to stop replay.")

    for step in range(num_steps):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print("Exiting replay...")
                env.close()
                pygame.quit()
                if save_video:
                    video_out.release()
                return

        env.sim.data.qpos[:] = joint_angles[step][:expected_qpos]
        env.sim.data.qvel[:] = joint_angles[step][expected_qpos:]
        env.sim.forward()

        obs, _, _, _ = env.step(actions[step])

        # Capture rendered frame
        if save_video:
            frame = obs["frontview_image"]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_out.write(frame)

        env.render()
        time.sleep(1 / video_fps)

    if save_video:
        video_out.release()
        print(f"Replay complete! Video saved as {video_filename}")

    env.close()
    pygame.quit()


if __name__ == "__main__":
    use_smoothed = input(
        "Replay smoothed data? (yes/no): ").strip().lower() == "yes"
    replay_demo(use_smoothed)
