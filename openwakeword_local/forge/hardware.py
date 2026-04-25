"""
forge/hardware.py
Detect GPU, CPU, RAM. Return a HardwareProfile with safe training limits.
"""
from __future__ import annotations
import os, platform, subprocess
from dataclasses import dataclass
from .common import log_ok, log_warn, log_info, C


@dataclass
class HardwareProfile:
    has_cuda: bool      = False
    gpu_name: str       = ""
    vram_gb:  float     = 0.0
    cuda_ver: str       = ""
    cpu_cores: int      = 1
    ram_gb:   float     = 4.0
    os_name:  str       = ""
    # computed safe limits
    aug_workers:  int   = 2
    val_rows:     int   = 5_000
    batch_size:   int   = 512

    def device(self) -> str:
        return "cuda" if self.has_cuda else "cpu"

    def summary_lines(self) -> list:
        lines = [
            f"Platform : {self.os_name}",
            f"CPU cores: {self.cpu_cores}",
            f"RAM      : {self.ram_gb:.1f} GB",
        ]
        if self.has_cuda:
            lines += [
                f"GPU      : {self.gpu_name}",
                f"VRAM     : {self.vram_gb:.1f} GB",
                f"CUDA     : {self.cuda_ver}",
            ]
        else:
            lines.append("GPU      : not available (CPU mode)")
        lines += [
            f"Workers  : {self.aug_workers}",
            f"Val rows : {self.val_rows:,}",
        ]
        return lines


def detect() -> HardwareProfile:
    hw = HardwareProfile()
    hw.os_name   = f"{platform.system()} {platform.machine()}"
    hw.cpu_cores = os.cpu_count() or 1
    hw.ram_gb    = _ram_gb()

    # Try GPU detection (torch may not be installed yet in step1)
    try:
        import torch
        if torch.cuda.is_available():
            hw.has_cuda = True
            hw.gpu_name = torch.cuda.get_device_name(0)
            hw.vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            hw.cuda_ver = torch.version.cuda or ""
    except Exception:
        pass

    # Safe limits
    hw.aug_workers = max(1, min(hw.cpu_cores - 1, 8))
    if hw.has_cuda:
        if hw.vram_gb >= 16:
            hw.val_rows, hw.batch_size = 100_000, 2048
        elif hw.vram_gb >= 8:
            hw.val_rows, hw.batch_size = 50_000, 1024
        elif hw.vram_gb >= 6:
            hw.val_rows, hw.batch_size = 10_000, 512
        elif hw.vram_gb >= 4:
            hw.val_rows, hw.batch_size =  5_000, 256
        else:
            hw.val_rows, hw.batch_size =  2_000, 128
    else:
        hw.val_rows  = 10_000 if hw.ram_gb >= 16 else 5_000 if hw.ram_gb >= 8 else 2_000
        hw.batch_size = 256

    return hw


def print_summary(hw: HardwareProfile):
    from .common import log_box
    log_box(hw.summary_lines())
    if hw.has_cuda:
        log_ok(f"GPU ready: {hw.gpu_name} ({hw.vram_gb:.1f} GB VRAM)")
    else:
        log_warn("No GPU — training will be slow. Install CUDA PyTorch if possible.")


def _ram_gb() -> float:
    try:
        sys = platform.system()
        if sys == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024 / 1024
        elif sys == "Darwin":
            r = subprocess.run(["sysctl","hw.memsize"],capture_output=True,text=True)
            return int(r.stdout.split(":")[1]) / 1024**3
        elif sys == "Windows":
            import ctypes
            class MEM(ctypes.Structure):
                _fields_ = [("dwLength",ctypes.c_ulong),
                            ("dwMemoryLoad",ctypes.c_ulong),
                            ("ullTotalPhys",ctypes.c_ulonglong)] + \
                           [(f"x{i}",ctypes.c_ulonglong) for i in range(7)]
            m = MEM(); m.dwLength = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1024**3
    except Exception:
        pass
    return 4.0
