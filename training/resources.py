"""CPU / RAM / GPU sampling used by the benchmark and training scripts.

CPU percentages follow psutil's convention: 100 % == one fully busy core, so on an
8-core machine a saturated run reports 800 %.  Child processes are included, which
matters because the subprocess environment backend puts the simulation in workers.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is optional
    psutil = None  # type: ignore[assignment]


def _torch_info() -> Dict[str, object]:
    info: Dict[str, object] = {"torch": None, "cuda_available": False, "device_name": None}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["cuda_total_memory_mb"] = round(props.total_memory / 1024 ** 2, 1)
    except Exception:  # pragma: no cover
        pass
    return info


def _cuda_memory_mb() -> Optional[Dict[str, float]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 ** 2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024 ** 2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        }
    except Exception:  # pragma: no cover
        return None


def _gpu_utilization() -> Optional[float]:
    try:
        import torch

        if hasattr(torch.cuda, "utilization") and torch.cuda.is_available():
            return float(torch.cuda.utilization())
    except Exception:  # pragma: no cover
        return None
    return None


class ResourceMonitor:
    """Samples CPU %, RSS and CUDA memory in a background thread."""

    def __init__(self, interval: float = 0.2, include_children: bool = True) -> None:
        self.interval = interval
        self.include_children = include_children
        self.cpu_samples: List[float] = []
        self.rss_samples: List[float] = []
        self.gpu_util_samples: List[float] = []
        self.cuda_alloc_samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = None
        if psutil is not None:
            try:
                self._process = psutil.Process(os.getpid())
                self._process.cpu_percent(None)  # prime the first delta
            except Exception:  # pragma: no cover
                self._process = None

    # ------------------------------------------------------------------ control
    def start(self) -> "ResourceMonitor":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.summary()

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ sampling
    def _sample_cpu(self) -> float:
        if self._process is None:
            return 0.0
        total = 0.0
        try:
            total += self._process.cpu_percent(None)
            if self.include_children:
                for child in self._process.children(recursive=True):
                    try:
                        total += child.cpu_percent(None)
                    except Exception:
                        continue
        except Exception:  # pragma: no cover - process vanished
            return 0.0
        return total

    def _sample_rss_mb(self) -> float:
        if self._process is None:
            return 0.0
        total = 0.0
        try:
            total += self._process.memory_info().rss
            if self.include_children:
                for child in self._process.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except Exception:
                        continue
        except Exception:  # pragma: no cover
            return 0.0
        return total / 1024 ** 2

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.cpu_samples.append(self._sample_cpu())
            self.rss_samples.append(self._sample_rss_mb())
            util = _gpu_utilization()
            if util is not None:
                self.gpu_util_samples.append(util)
            memory = _cuda_memory_mb()
            if memory is not None:
                self.cuda_alloc_samples.append(memory["allocated_mb"])

    # ------------------------------------------------------------------ summary
    def summary(self) -> Dict[str, object]:
        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "max": 0.0, "count": 0}
            return {"mean": sum(values) / len(values), "max": max(values), "count": len(values)}

        summary: Dict[str, object] = {
            "cpu_percent": _stats(self.cpu_samples),
            "rss_mb": _stats(self.rss_samples),
            "psutil_available": psutil is not None,
        }
        if self.gpu_util_samples:
            summary["gpu_utilization_percent"] = _stats(self.gpu_util_samples)
        if self.cuda_alloc_samples:
            summary["cuda_allocated_mb"] = _stats(self.cuda_alloc_samples)
        memory = _cuda_memory_mb()
        if memory is not None:
            summary["cuda_memory"] = memory
        return summary


def machine_info() -> Dict[str, object]:
    """Static description of the machine (CPU count, RAM, torch/CUDA)."""
    info: Dict[str, object] = {
        "cpu_count": os.cpu_count(),
        "platform": os.name,
        "pid": os.getpid(),
    }
    if psutil is not None:
        info["cpu_logical"] = psutil.cpu_count(logical=True)
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_mb"] = round(psutil.virtual_memory().total / 1024 ** 2, 1)
        info["ram_available_mb"] = round(psutil.virtual_memory().available / 1024 ** 2, 1)
    info.update(_torch_info())
    return info


def format_resource_summary(summary: Dict[str, object]) -> str:
    cpu = summary.get("cpu_percent", {})
    rss = summary.get("rss_mb", {})
    parts = [
        f"CPU {cpu.get('mean', 0):.0f}% (peak {cpu.get('max', 0):.0f}%)",
        f"RAM {rss.get('mean', 0):.0f} MB (peak {rss.get('max', 0):.0f} MB)",
    ]
    cuda = summary.get("cuda_memory")
    if cuda:
        parts.append(f"CUDA {cuda['allocated_mb']:.0f} MB (peak {cuda['max_allocated_mb']:.0f} MB)")
    gpu = summary.get("gpu_utilization_percent")
    if gpu:
        parts.append(f"GPU util {gpu.get('mean', 0):.0f}%")
    return " | ".join(parts)


def wait_for_memory(threshold_percent: float = 90.0, timeout: float = 5.0) -> bool:
    """Best-effort helper used before launching many worker processes."""
    if psutil is None:
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        if psutil.virtual_memory().percent < threshold_percent:
            return True
        time.sleep(0.2)
    return False
