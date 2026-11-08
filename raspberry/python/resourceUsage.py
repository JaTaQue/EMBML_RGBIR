import psutil

class ResourceUsage:
    def __init__(self):
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.disk_usage = 0.0

    def update_usage(self, cpu: float = None, memory: float = None, disk: float = None) -> None:
        """Manually update usage values or automatically fetch current usage if parameters are omitted."""
        if cpu is None or memory is None or disk ==None:
            self.monitor_resources()
        if cpu is not None:
            self.cpu_usage = cpu
        if memory is not None:
            self.memory_usage = memory
        if disk is not None:
            self.disk_usage = disk

    def monitor_resources(self) -> None:
        """Fetch real-time usage metrics using psutil."""
        self.cpu_usage = psutil.cpu_percent(interval=1)
        self.memory_usage = psutil.virtual_memory().percent
        self.disk_usage = psutil.disk_usage('/').percent

    def get_usage(self, refresh: bool = True) -> dict[str, float]:
        """Return the resource usage as a dictionary. Refresh data by default."""
        if refresh:
            self.monitor_resources()
        return {
            "cpu": self.cpu_usage,
            "memory": self.memory_usage,
            "disk": self.disk_usage
        }


import psutil
print("CPU usage (%):", psutil.cpu_percent(interval=1))

ram = psutil.virtual_memory()
print("RAM usage (%):", ram.percent)
print("RAM used (GB):", round(ram.used / 1e9, 2))