from modules.demo_record import DemoRecorder
from modules.smoothing import process_and_save_smoothed_data
from modules.visual import plot_trajectories
from modules.replay import replay_demo
import os
import sys
import os

if __name__ == "__main__":
    print("Step 1: Recording Demonstrations")
    num_demos = int(input("Enter the number of demonstrations to record: "))

    if num_demos > 0:
        recorder = DemoRecorder(num_demos)
        recorder.record_demo()

    # file paths
    h5_file_path = r"robosuite/DEMO/recorded_demo/recorded_demo.h5"
    save_directory = r"robosuite/DEMO/smoothed_demo"
    smoothed_h5_file_path = os.path.join(
        save_directory, "recorded_demo_smoothed.h5")

    print("\nStep 2: Smoothing Recorded Data ")
    if os.path.exists(h5_file_path):
        process_and_save_smoothed_data(h5_file_path, save_directory)
        print("Smoothing completed successfully!")
    else:
        print("Error: Recorded data file not found.")

    print("\nStep 3: Visualizing Trajectories ")
    print("Plotting raw trajectory data...")
    plot_trajectories(h5_file_path, use_smoothed=False)

    print("\nPlotting smoothed trajectory data...")
    plot_trajectories(smoothed_h5_file_path, use_smoothed=True)

    # Step 4: Replay the latest recorded demonstration
    print("\nStep 4: Replaying the Latest Demonstration ")
    use_smoothed = input(
        "Replay smoothed data? (yes/no): ").strip().lower() == "yes"
    replay_demo(use_smoothed)

    print("\nAll steps completed successfully!")
