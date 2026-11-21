import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import psutil
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 1. SYSTEM TELEMETRY MONITOR (Pi 5 CPU, RAM, Temp + Hailo-8 Die Temp)
# ---------------------------------------------------------------------------
def get_pi_temp() -> float:
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        return float(out.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        return -1.0


class ResourceProfiler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.proc = psutil.Process()
        self.cpu_samples = []
        self.ram_samples = []
        self.temp_samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=self.interval))
            self.ram_samples.append(self.proc.memory_info().rss / (1024 * 1024))
            t = get_pi_temp()
            if t > 0:
                self.temp_samples.append(t)

    def start(self):
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        return {
            "avg_cpu_pct": round(float(np.mean(self.cpu_samples)), 2) if self.cpu_samples else 0.0,
            "peak_cpu_pct": round(float(np.max(self.cpu_samples)), 2) if self.cpu_samples else 0.0,
            "avg_ram_mb": round(float(np.mean(self.ram_samples)), 2) if self.ram_samples else 0.0,
            "peak_ram_mb": round(float(np.max(self.ram_samples)), 2) if self.ram_samples else 0.0,
            "peak_soc_temp_c": round(float(np.max(self.temp_samples)), 1) if self.temp_samples else -1.0,
        }


# ---------------------------------------------------------------------------
# 2. CAMERA SIMULATOR (Pre-recorded RGBT-Tiny Sequence or Dummy Fallback)
# ---------------------------------------------------------------------------
class CameraSimulator:
    """Simulates a 15 FPS camera stream from an image sequence, video file, or dummy generator."""

    def __init__(self, source: str | None, dummy_frames: int = 300, width: int = 640, height: int = 512):
        self.mode = "dummy"
        self.image_paths = []
        self.cap = None
        self.dummy_frames = dummy_frames
        self.width = width
        self.height = height
        self.idx = 0

        if source:
            p = Path(source)
            if p.is_dir():
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                self.image_paths = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
                if self.image_paths:
                    self.mode = "image_seq"
            elif p.is_file():
                self.cap = cv2.VideoCapture(str(p))
                if self.cap.isOpened():
                    self.mode = "video"

    def __len__(self) -> int:
        if self.mode == "image_seq":
            return len(self.image_paths)
        if self.mode == "video":
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.dummy_frames
        return self.dummy_frames

    def read(self):
        if self.mode == "image_seq":
            if self.idx >= len(self.image_paths):
                return False, None
            frame = cv2.imread(str(self.image_paths[self.idx]))
            self.idx += 1
            return frame is not None, frame

        if self.mode == "video":
            ret, frame = self.cap.read()
            self.idx += 1
            return ret, frame

        # Dummy 640x512 BGR frames (matches RGBT-Tiny resolution)
        if self.idx >= self.dummy_frames:
            return False, None
        self.idx += 1
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------------
# 3. OPTIONAL ZERO-FLASK MJPEG STREAM SERVER (http://<PI_IP>:5000)
# ---------------------------------------------------------------------------
class LiveFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg_bytes = None

    def update(self, bgr_frame: np.ndarray):
        ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.jpeg_bytes = buf.tobytes()


FRAME_BUFFER = LiveFrameBuffer()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with FRAME_BUFFER.lock:
                    data = FRAME_BUFFER.jpeg_bytes
                if data is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        return  # Silence HTTP access logs


# ---------------------------------------------------------------------------
# 4. MAIN EVALUATION & TRACKING PIPELINE
# ---------------------------------------------------------------------------
def auto_locate_hailo_model_dir() -> str:
    # Looks for folder containing both *.hef and metadata.yaml
    search_root = Path(__file__).resolve().parents[3]
    for hef in search_root.rglob("*.hef"):
        if (hef.parent / "metadata.yaml").is_file():
            return str(hef.parent)
    raise FileNotFoundError(
        f"Could not find a folder with both *.hef and metadata.yaml under {search_root}"
    )


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Hailo-8 Detection & Tracking Evaluator")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to *_hailo_model folder")
    parser.add_argument("--source", type=str, default=None, help="Path to test sequence folder or .mp4 (omit for dummy)")
    parser.add_argument("--data-yaml", type=str, default=None, help="Path to dataset.yaml for mAP/Precision/Recall")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference input size (640, 512, 416)")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (0.20 tuned for INT8)")
    parser.add_argument("--drone-class-id", type=int, default=0, help="Class ID for drones in your dataset")
    parser.add_argument("--stream", action="store_true", help="Enable live MJPEG browser preview on port 5000")
    parser.add_argument("--out-dir", type=str, default="eval_results", help="Output directory for logs & MOT file")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = args.model_dir or auto_locate_hailo_model_dir()
    hef_file = next(Path(model_dir).glob("*.hef"))
    model_size_mb = round(hef_file.stat().st_size / (1024 * 1024), 2)

    print(f"[INFO] Loading Hailo-8 NPU Model: {model_dir} ({model_size_mb} MB)")
    model = YOLO(model_dir, task="detect")

    if args.stream:
        srv = ThreadingHTTPServer(("0.0.0.0", 5000), MJPEGHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print("[INFO] Live stream enabled at http://<RASPBERRY_PI_IP>:5000/")

    # --- STEP A: STREAM + NPU DETECTION + BYTETRACK + SYSTEM LATENCY ---
    cam = CameraSimulator(args.source, dummy_frames=300)
    print(f"[INFO] Camera Simulator Mode: {cam.mode} ({len(cam)} frames)")

    # Warmup NPU (15 frames)
   # Warmup NPU using model.track() so only ONE VDevice is created
    warmup_img = np.zeros((512, 640, 3), dtype=np.uint8)
    for _ in range(15):
        model.track(warmup_img, imgsz=args.imgsz, conf=args.conf, persist=False, tracker=args.tracker, verbose=False)
    
    profiler = ResourceProfiler(interval=0.1)
    profiler.start()

    mot_lines = []
    e2e_latencies_ms = []
    pre_ms_list, infer_ms_list, post_ms_list = [], [], []
    frame_idx = 0

    t_run_start = time.perf_counter()
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        results = model.track(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            persist=True,
            tracker=args.tracker,
            verbose=False,
        )
        e2e_ms = (time.perf_counter() - t0) * 1000.0
        e2e_latencies_ms.append(e2e_ms)

        r = results[0]
        if hasattr(r, "speed") and r.speed:
            pre_ms_list.append(r.speed.get("preprocess", 0.0))
            infer_ms_list.append(r.speed.get("inference", 0.0))
            post_ms_list.append(r.speed.get("postprocess", 0.0))

        # Export MOTChallenge format for TrackEval (HOTA, MOTA, IDF1, ID switches)
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else np.full(len(xyxy), -1)
            for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids):
                w, h = x2 - x1, y2 - y1
                mot_lines.append(f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")

        if args.stream:
            annotated = r.plot()
            fps_now = 1000.0 / max(e2e_ms, 1e-3)
            cv2.putText(
                annotated,
                f"Hailo-8 + ByteTrack | {fps_now:.1f} FPS ({e2e_ms:.1f} ms)",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            FRAME_BUFFER.update(annotated)

    total_elapsed_s = time.perf_counter() - t_run_start
    cam.release()
    sys_metrics = profiler.stop()

    mot_file = out_dir / "tracks_mot.txt"
    mot_file.write_text("".join(mot_lines))

    e2e_fps = frame_idx / max(total_elapsed_s, 1e-6)
    summary = {
        "model_dir": model_dir,
        "hef_size_mb": model_size_mb,
        "source_mode": cam.mode,
        "frames_processed": frame_idx,
        "target_source_fps": 15.0,
        "meets_15fps_realtime": bool(e2e_fps >= 15.0),
        "e2e_fps": round(e2e_fps, 2),
        "e2e_mean_ms": round(float(np.mean(e2e_latencies_ms)), 2),
        "e2e_p95_ms": round(float(np.percentile(e2e_latencies_ms, 95)), 2),
        "stage_preprocess_ms": round(float(np.mean(pre_ms_list)), 2) if pre_ms_list else 0.0,
        "stage_npu_infer_ms": round(float(np.mean(infer_ms_list)), 2) if infer_ms_list else 0.0,
        "stage_post_track_ms": round(float(np.mean(post_ms_list)), 2) if post_ms_list else 0.0,
        **sys_metrics,
        "mot_tracks_file": str(mot_file),
    }

    # --- STEP B: DETECTION ACCURACY EVALUATION (mAP, Precision, Recall) ---
    if args.data_yaml:
        print(f"\n[INFO] Running NPU Accuracy Validation against: {args.data_yaml}")
        val_model = YOLO(model_dir, task="detect")
        metrics = val_model.val(data=args.data_yaml, imgsz=args.imgsz, conf=args.conf, batch=1, plots=False)

        summary["all_classes_mAP50"] = round(float(metrics.box.map50), 4)
        summary["all_classes_mAP50_95"] = round(float(metrics.box.map), 4)
        summary["all_classes_precision"] = round(float(metrics.box.mp), 4)
        summary["all_classes_recall"] = round(float(metrics.box.mr), 4)

        # Drone-specific metrics (per your SOTA paper: report drones separately)
        if args.drone_class_id < len(metrics.box.ap50):
            summary["drone_mAP50"] = round(float(metrics.box.ap50[args.drone_class_id]), 4)
            summary["drone_mAP50_95"] = round(float(metrics.box.ap[args.drone_class_id]), 4)
            summary["drone_precision"] = round(float(metrics.box.p[args.drone_class_id]), 4)
            summary["drone_recall"] = round(float(metrics.box.r[args.drone_class_id]), 4)

    report_path = out_dir / "pi5_benchmark_report.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 64)
    print("RASPBERRY PI 5 (8GB) + HAILO-8 (26 TOPS) EVALUATION SUMMARY")
    print("=" * 64)
    for k, v in summary.items():
        print(f"  {k:<24}: {v}")
    print("=" * 64)


if __name__ == "__main__":
    main()