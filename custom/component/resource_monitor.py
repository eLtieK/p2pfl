import time
import threading
import psutil

from p2pfl.utils.node_component import NodeComponent

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


class ResourceMonitor(NodeComponent):
    def __init__(self, interval: float = 1.0):
        super().__init__()

        self.interval = interval

        self._running = False
        self._thread = None

        # logs (time series)
        self.timestamps = []

        self.cpu_usage = []          # process CPU (% per core normalized)
        self.cpu_system = []         # system CPU %

        self.ram_usage = []          # process RAM (MB)
        self.ram_system = []         # system RAM %

        self.gpu_usage = []          # GPU %
        self.gpu_mem = []            # GPU mem (MB)
        self.gpu_mem_pct = []        # GPU mem %

    # ======================
    # CONTROL
    # ======================
    def start(self):
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    # ======================
    # MONITOR LOOP
    # ======================
    def _run(self):
        process = psutil.Process()

        while self._running:
            now = time.time()

            # CPU
            cpu_proc = process.cpu_percent(interval=None) / psutil.cpu_count()
            cpu_sys = psutil.cpu_percent(interval=None)

            # RAM
            mem_proc = process.memory_info().rss / (1024**2)  # MB
            mem_sys = psutil.virtual_memory().percent

            # GPU
            gpu = 0
            gpu_mem = 0
            gpu_mem_pct = 0

            if GPU_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    gpu = util.gpu
                    gpu_mem = meminfo.used / (1024**2)  # MB
                    gpu_mem_pct = (meminfo.used / meminfo.total) * 100
                except:
                    pass

            # store
            self.timestamps.append(now)

            self.cpu_usage.append(cpu_proc)
            self.cpu_system.append(cpu_sys)

            self.ram_usage.append(mem_proc)
            self.ram_system.append(mem_sys)

            self.gpu_usage.append(gpu)
            self.gpu_mem.append(gpu_mem)
            self.gpu_mem_pct.append(gpu_mem_pct)

            time.sleep(self.interval)

    # ======================
    # STATS
    # ======================
    def _avg(self, arr):
        return sum(arr) / len(arr) if arr else 0

    def _max(self, arr):
        return max(arr) if arr else 0

    def _min(self, arr):
        return min(arr) if arr else 0

    def get_stats(self):
        return {
            "cpu_avg": self._avg(self.cpu_usage),
            "cpu_max": self._max(self.cpu_usage),

            "cpu_system_avg": self._avg(self.cpu_system),

            "ram_avg": self._avg(self.ram_usage),
            "ram_max": self._max(self.ram_usage),

            "ram_system_avg": self._avg(self.ram_system),

            "gpu_avg": self._avg(self.gpu_usage),
            "gpu_max": self._max(self.gpu_usage),

            "gpu_mem_avg": self._avg(self.gpu_mem),
            "gpu_mem_max": self._max(self.gpu_mem),

            "gpu_mem_pct_avg": self._avg(self.gpu_mem_pct),

            "samples": len(self.cpu_usage),
        }

    def get_timeseries(self):
        return {
            "timestamps": self.timestamps,
            "cpu": self.cpu_usage,
            "cpu_system": self.cpu_system,
            "ram": self.ram_usage,
            "ram_system": self.ram_system,
            "gpu": self.gpu_usage,
            "gpu_mem": self.gpu_mem,
            "gpu_mem_pct": self.gpu_mem_pct,
        }

    def reset(self):
        self.timestamps.clear()

        self.cpu_usage.clear()
        self.cpu_system.clear()

        self.ram_usage.clear()
        self.ram_system.clear()

        self.gpu_usage.clear()
        self.gpu_mem.clear()
        self.gpu_mem_pct.clear()