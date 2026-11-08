import time
import cv2
import numpy as np
import psutil
from stream_simulator import RGBTinyStreamSimulator

# Direct path to sequence folder
SEQUENCE_PATH = "/home/group6/Documents/group_6_RGBIR_2026/images/DJI_0024_2"

def run_test():
    simulator = RGBTinyStreamSimulator(SEQUENCE_PATH, target_fps=15.0)

    frame_count = 0
    start_time = time.perf_counter()

    # Lists to store resource usage metrics per frame
    cpu_percent_list = []
    ram_percent_list = []
    ram_gb_list = []

    print("[Test] Starting 15 FPS dual-camera stream simulator (Press 'q' to stop)...")

    while True:
        loop_start = time.perf_counter()
   
        # Sample CPU without blocking (interval=None) and RAM metrics
        cpu_percent_list.append(psutil.cpu_percent(interval=None))
        ram = psutil.virtual_memory()
        ram_percent_list.append(ram.percent)
        ram_gb_list.append(ram.used / 1e9)
        
        # Read 4-channel tensor (640, 640, 4) and raw RGB frame
        fused_tensor, raw_rgb, success = simulator.read_fused_frame(target_size=(640, 640))

        if not success:
            print("[Test] End of sequence reached.")
            break

        frame_count += 1

        # Verify array shape and type
        assert fused_tensor.shape == (640, 640, 4), f"Unexpected tensor shape: {fused_tensor.shape}"
        assert fused_tensor.dtype == np.uint8

        # Enforce 15 FPS playback timing delay
        elapsed = time.perf_counter() - loop_start
        sleep_time = simulator.frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        current_fps = frame_count / (time.perf_counter() - start_time)

        # Draw frame information overlay
        cv2.putText(raw_rgb, f"Rate: {current_fps:.1f} FPS (Frame {frame_count}/{simulator.num_frames})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("RGBT Stream Simulator", raw_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"[Test] Successfully streamed {frame_count} frames at {current_fps:.2f} FPS.")

    # Calculate and print averages
    if frame_count > 0:
        avg_cpu = sum(cpu_percent_list) / len(cpu_percent_list)
        avg_ram_percent = sum(ram_percent_list) / len(ram_percent_list)
        avg_ram_gb = sum(ram_gb_list) / len(ram_gb_list)

        print("\n--- Average Resource Usage ---")
        print(f"Average CPU usage (%): {avg_cpu:.2f}%")
        print(f"Average RAM usage (%): {avg_ram_percent:.2f}%")
        print(f"Average RAM used (GB): {avg_ram_gb:.2f} GB")

if __name__ == "__main__":
    run_test()