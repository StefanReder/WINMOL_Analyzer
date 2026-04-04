from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import List


@dataclass
class HardwareInfo:
    cpu_count: int
    total_ram_gb: float
    gpu_count: int
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_gb: List[float] = field(default_factory=list)

    @classmethod
    def detect(cls) -> "HardwareInfo":
        cpu_count = os.cpu_count() or 1
        total_ram_gb = cls._detect_total_ram_gb()
        gpu_names = cls._detect_gpu_names_tf()
        gpu_memory_gb = cls._detect_gpu_memory_gb_nvidia_smi()
        gpu_count = len(gpu_names)
        if gpu_memory_gb and gpu_count != len(gpu_memory_gb):
            if len(gpu_memory_gb) > gpu_count:
                gpu_memory_gb = gpu_memory_gb[:gpu_count]
            else:
                gpu_memory_gb.extend([0.0] * (gpu_count - len(gpu_memory_gb)))
        return cls(
            cpu_count=cpu_count,
            total_ram_gb=total_ram_gb,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_memory_gb=gpu_memory_gb,
        )

    @staticmethod
    def _detect_total_ram_gb() -> float:
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            return 0.0

    @staticmethod
    def _detect_gpu_names_tf() -> List[str]:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            names = []
            for gpu in gpus:
                names.append(getattr(gpu, 'name', str(gpu)))
            return names
        except Exception:
            return []

    @staticmethod
    def _detect_gpu_memory_gb_nvidia_smi() -> List[float]:
        try:
            result = \
                subprocess.run(['nvidia-smi', '--query-gpu=memory.total',
                                '--format=csv,noheader,nounits'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               check=False, )
            if result.returncode != 0:
                return []
            values = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(round(float(line) / 1024.0, 2))
                except Exception:
                    continue
            return values
        except Exception:
            return []
