import robosuite as suite
import numpy as np
import h5py
import pygame
import time
import os
from datetime import datetime
import sys


class DemoRecorder:
    def __init__(self, num_demos):
        if num_demos <= 0:
            print("No demonstrations to record. Exiting...")
            sys.exit(0)

        pygame.init()
        self.screen = pygame.display.set_mode((300, 300))
        self.num_demos = num_demos
        self.saved_demos = 0  

        # Set up environment
        self.env = suite.make(
            env_name="Lift",
            robots="UR5e",
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            ignore_done=True,
        )

        # Set up the save path
        self.save_path = "robosuite/DEMO/recorded_demo/recorded_demo.h5"
        self.temp_data = []  

        # Key mappings
        self.key_action = {
            pygame.K_w: [0.5, 0, 0, 0, 0, 0, 0],   # Move forward
            pygame.K_s: [-0.5, 0, 0, 0, 0, 0, 0],  # Move backward
            pygame.K_a: [0, -0.5, 0, 0, 0, 0, 0],  # Move left
            pygame.K_d: [0, 0.5, 0, 0, 0, 0, 0],   # Move right
            pygame.K_q: [0, 0, 0, -0.5, 0, 0, 0],  # Rotate left
            pygame.K_e: [0, 0, 0, 0.5, 0, 0, 0],   # Rotate right
            pygame.K_r: [0, 0, 0.5, 0, 0, 0, 0],   # Move up
            pygame.K_f: [0, 0, -0.5, 0, 0, 0, 0],  # Move down
            pygame.K_g: [0, 0, 0, 0, 0, 0, -1],    # Close gripper
            pygame.K_h: [0, 0, 0, 0, 0, 0, 1],     # Open gripper
        }

    def randomize_robot_and_object(self):
        qpos = self.env.sim.data.qpos.copy()
        qpos[:6] += np.random.uniform(-0.05, 0.05, size=6)
        self.env.sim.data.qpos[:] = qpos

        cube_qpos = self.env.sim.data.get_joint_qpos("cube_joint0").copy()
        cube_qpos[:2] += np.random.uniform(-0.2, 0.2, size=2)
        self.env.sim.data.set_joint_qpos("cube_joint0", cube_qpos)
        self.env.sim.forward()

    def record_demo(self):
        for demo_idx in range(self.num_demos):
            self.env.reset()
            self.randomize_robot_and_object()

            demo_data = {
                "actions": [],
                "robot_states": [],
                "joint_angles": [],
                "cube_states": []
            }

            print(f"Recording demo {demo_idx + 1}. Press ESC to stop.")

            running = True
            while running:
                action = np.zeros(7)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        print("Stopping recording...")
                        running = False

                keys = pygame.key.get_pressed()
                for key, movement in self.key_action.items():
                    if keys[key]:
                        action += np.array(movement)

                obs, reward, done, _ = self.env.step(action)

                # Capture State Data
                joint_qpos = self.env.sim.data.qpos.copy()
                joint_qvel = self.env.sim.data.qvel.copy()
                joint_state = np.concatenate([joint_qpos, joint_qvel])

                ee_index = self.env.sim.model.site_name2id(
                    "gripper0_right_grip_site")
                ee_position = self.env.sim.data.site_xpos[ee_index].copy()

                cube_position = self.env.sim.data.get_body_xpos("cube_main")
                cube_orientation = self.env.sim.data.get_body_xquat(
                    "cube_main")
                cube_state = np.concatenate([cube_position, cube_orientation])

                demo_data["actions"].append(action.tolist())
                demo_data["robot_states"].append(ee_position.tolist())
                demo_data["joint_angles"].append(joint_state.tolist())
                demo_data["cube_states"].append(cube_state.tolist())

                self.env.render()
                time.sleep(0.05)

            save_demo = input(
                f"Save demo {demo_idx + 1}? (yes/no): ").strip().lower()
            if save_demo == "yes":
                self.temp_data.append(demo_data)
                self.saved_demos += 1
                print(f"Demo {demo_idx + 1} will be saved.")
            else:
                print(f"Demo {demo_idx + 1} discarded.")

        if self.saved_demos > 0:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with h5py.File(self.save_path, "w") as h5f:
                for i, demo_data in enumerate(self.temp_data):
                    demo_group = h5f.create_group(f"demo_{i+1}")
                    for key, value in demo_data.items():
                        demo_group.create_dataset(key, data=np.array(
                            value, dtype=np.float32), compression="gzip")
            print(f"\n{self.saved_demos} demonstrations saved in {self.save_path}.")
        else:
            print("\nNo demonstrations were saved. No file was created.")

        self.env.close()
        pygame.quit()


if __name__ == "__main__":
    num_demos = int(input("Enter the number of demonstrations: "))

    if num_demos <= 0:
        print("No demonstrations to record. Exiting program.")
        sys.exit(0)

    recorder = DemoRecorder(num_demos)
    recorder.record_demo()
