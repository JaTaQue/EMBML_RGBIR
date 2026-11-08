"""
WBS 3.2 + WBS 3.6 + WBS 3.3 Integrated Pipeline Test
Simulates dual video stream, updates ByteTrack tracker, logs execution latency & FPS.
"""

import time
import cv2
import numpy as np
from stream_simulator import RGBTinyStreamSimulator
from byte_tracker import ByteTracker

SEQUENCE_PATH = "/home/group6/Documents/group_6_RGBIR_2026/images/DJI_0022_1"

def run_pipeline():
    # 1. Initialize Stream Simulator (15 FPS)
    simulator = RGBTinyStreamSimulator(SEQUENCE_PATH, target_fps=15.0)
    
    # 2. Initialize ByteTrack Tracker
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.8)

    frame_count = 0
    total_tracking_time = 0.0
    pipeline_start = time.perf_counter()

    print("[Pipeline] Running Integrated Stream + Tracker Pipeline...")

    while True:
        loop_start = time.perf_counter()

        # Step A: Get 4-channel tensor from stream simulator (WBS 3.2)
        fused_tensor, raw_rgb, success = simulator.read_fused_frame(target_size=(640, 640))
        if not success:
            print("[Pipeline] Stream finished.")
            break

        frame_count += 1

        # Step B: Simulated Drone Detections (Simulates YOLO output [x1, y1, x2, y2, score])
        # In a real run, this array comes directly from Hailo-8 NPU inference
        t_time = (frame_count * 0.5)
        simulated_detections = np.array([
            [100 + t_time, 150 + t_time, 114 + t_time, 162 + t_time, 0.88],  # Moving 14x12 px tiny drone
            [500 - t_time * 0.8, 100 + t_time * 0.3, 512 - t_time * 0.8, 110 + t_time * 0.3, 0.79],
            [300 + t_time * 0.6, 200 - t_time * 0.4, 312 + t_time * 0.6, 210 - t_time * 0.4, 0.85]
        ], dtype=np.float32)

        # Step C: Update Tracker (WBS 3.6)
        track_start = time.perf_counter()
        online_targets = tracker.update(simulated_detections)
        track_latency_ms = (time.perf_counter() - track_start) * 1000.0
        total_tracking_time += track_latency_ms

        # Step D: Enforce 15 FPS stream delay
        elapsed = time.perf_counter() - loop_start
        sleep_time = simulator.frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        current_fps = frame_count / (time.perf_counter() - pipeline_start)

        # Step E: Visualize Bounding Boxes and Track IDs
        for target in online_targets:
            tlbr = target.tlbr.astype(int)
            tid = target.track_id
            cv2.rectangle(raw_rgb, (tlbr[0], tlbr[1]), (tlbr[2], tlbr[3]), (0, 255, 0), 2)
            cv2.putText(raw_rgb, f"Drone ID: {tid}", (tlbr[0], max(15, tlbr[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Overlay System Metrics (WBS 3.3)
        cv2.putText(raw_rgb, f"Pipeline FPS: {current_fps:.1f} | Tracker Latency: {track_latency_ms:.2f} ms",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("RGBT Drone Tracker (ByteTrack)", raw_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    avg_track_ms = total_tracking_time / max(1, frame_count)
    print(f"\n[Summary Metric]")
    print(f" - Total Frames Processed: {frame_count}")
    print(f" - Average Tracker Latency: {avg_track_ms:.3f} ms per frame")
    print(f" - End-to-End Pipeline FPS: {current_fps:.2f} FPS")

if __name__ == "__main__":
    run_pipeline()