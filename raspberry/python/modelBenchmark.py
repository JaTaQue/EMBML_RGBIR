import subprocess
import threading
import time
import numpy as np
import psutil
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType
from pathlib import Path

# Searches /home/group6/Documents/group_6_RGBIR_2026/ recursively for your .hef file
PROJECT_ROOT = Path(__file__).resolve().parents[3]
hef_matches = list(PROJECT_ROOT.rglob("*.hef"))

if not hef_matches:
    raise FileNotFoundError(f"No .hef file found anywhere under: {PROJECT_ROOT}")

HEF_PATH = str(hef_matches[0])
print(f"Found HEF model at: {HEF_PATH}")
WARMUP_FRAMES = 30
BENCHMARK_FRAMES = 1500


def get_pi_cpu_temp() -> float:
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True)
        return float(out.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        return -1.0


def get_hailo_temp(vdevice) -> float:
    try:
        devices = vdevice.get_physical_devices()
        if devices:
            t = devices[0].control.get_chip_temperature()
            return round((t.ts0_temperature + t.ts1_temperature) / 2.0, 1)
    except Exception:
        pass
    return -1.0


class HostTelemetry:
    """Monitors Pi 5 host CPU %, RAM (MB), and SoC Temp while the NPU runs."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.proc = psutil.Process()
        self.cpu_samples = []
        self.ram_samples = []
        self.temp_samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=self.interval))
            self.ram_samples.append(self.proc.memory_info().rss / (1024 * 1024))
            temp = get_pi_cpu_temp()
            if temp > 0:
                self.temp_samples.append(temp)

    def start(self):
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        return {
            "avg_cpu_pct": float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0,
            "peak_cpu_pct": float(np.max(self.cpu_samples)) if self.cpu_samples else 0.0,
            "avg_ram_mb": float(np.mean(self.ram_samples)) if self.ram_samples else 0.0,
            "peak_ram_mb": float(np.max(self.ram_samples)) if self.ram_samples else 0.0,
            "peak_pi_temp_c": float(np.max(self.temp_samples)) if self.temp_samples else -1.0,
        }


def run_npu_benchmark():
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as vdevice:
        npu_temp_start = get_hailo_temp(vdevice)
        infer_model = vdevice.create_infer_model(HEF_PATH)
        infer_model.set_batch_size(10)
        infer_model.input().set_format_type(FormatType.UINT8)
        for out in infer_model.outputs:
            out.set_format_type(FormatType.FLOAT32)

        with infer_model.configure() as configured_model:
            bindings = configured_model.create_bindings()

            in_shape = infer_model.input().shape
            input_buf = np.random.randint(0, 255, size=in_shape, dtype=np.uint8)
            bindings.input().set_buffer(input_buf)

            out_buffers = {}
            for out in infer_model.outputs:
                buf = np.empty(out.shape, dtype=np.float32)
                bindings.output(out.name).set_buffer(buf)
                out_buffers[out.name] = buf

            # Warmup NPU
            for _ in range(WARMUP_FRAMES):
                configured_model.run([bindings], 10000)

            # Timed pure-NPU inference loop
            telemetry = HostTelemetry(interval=0.1)
            telemetry.start()

            latencies_ms = []
            t_start = time.perf_counter()
            for _ in range(BENCHMARK_FRAMES):
                t0 = time.perf_counter()
                configured_model.run([bindings], 10000)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            total_s = time.perf_counter() - t_start

            host_stats = telemetry.stop()
            npu_temp_end = get_hailo_temp(vdevice)

    fps = BENCHMARK_FRAMES / total_s
    print("\n" + "=" * 56)
    print("PURE HAILO-8 (26 TOPS) NPU BENCHMARK (.HEF ONLY)")
    print("=" * 56)
    print(f"HEF File:           {HEF_PATH}")
    print(f"Input Shape:        {in_shape}")
    print(f"NPU FPS:            {fps:.2f} FPS")
    print(f"Mean NPU Latency:   {np.mean(latencies_ms):.2f} ms")
    print(f"P95 NPU Latency:    {np.percentile(latencies_ms, 95):.2f} ms")
    if npu_temp_end > 0:
        print(f"Hailo-8 Die Temp:   {npu_temp_start}°C -> {npu_temp_end}°C")
    print("-" * 56)
    print(f"Host CPU Load:      {host_stats['avg_cpu_pct']:.1f}% avg | {host_stats['peak_cpu_pct']:.1f}% peak")
    print(f"Host RAM (RSS):     {host_stats['avg_ram_mb']:.1f} MB avg | {host_stats['peak_ram_mb']:.1f} MB peak")
    if host_stats["peak_pi_temp_c"] > 0:
        print(f"Pi 5 Peak SoC Temp: {host_stats['peak_pi_temp_c']:.1f}°C")
    print("=" * 56)


if __name__ == "__main__":
    run_npu_benchmark()