import os
import torch
import torch.nn as nn
import numpy as np
import einops
import matplotlib.pyplot as plt
from collections.abc import Callable, Iterable
from typing import Optional
import math
import tqdm

import psutil

total_gb = psutil.virtual_memory().total / (1024 ** 3)
print(f"Total RAM: {total_gb:.2f} GB")

available_gb = psutil.virtual_memory().available / (1024 ** 3)
print(f"Available RAM: {available_gb:.2f} GB")






data_dir = "/scratch/shared/beegfs/piyush/datasets/text_data"
filepath = f"{data_dir}/TinyStoriesV2-GPT4-train-tokenized.npy"
assert os.path.exists(filepath)
data = np.load(filepath, mmap_mode='r')
type(data), len(data)

import os
import psutil

_proc = psutil.Process(os.getpid())

def get_rss_mb() -> float:
    """Current process RSS in MB."""
    return _proc.memory_info().rss / (1024 * 1024)


class MemoryTracker:
    """Tracks current and peak RSS."""
    def __init__(self):
        self.peak = get_rss_mb()

    def update(self) -> tuple[float, float]:
        current = get_rss_mb()
        self.peak = max(self.peak, current)
        return current, self.peak


import numpy as np

def iterate_with_memmap(path, chunk_size=100_000):
    arr = np.load(path, mmap_mode='r')
    tracker = MemoryTracker()
    
    ram = []
    itr = []

    pbar = tqdm.tqdm(range(0, len(arr), chunk_size), bar_format="{l_bar}{bar:20}{r_bar}")

    for i in pbar:
        chunk = arr[i:i + chunk_size]

        # ---- your processing here ----
        # Check: if any token is beyond vocab size
        assert not (chunk >= 10257).sum().astype(bool)

        # update memory stats
        current, peak = tracker.update()
        ram.append(current)
        itr.append(i)

        # add to tqdm bar
        pbar.set_postfix({
            "RAM": f"{current:.1f} MB",
            "Peak": f"{peak:.1f} MB"
        })

        del chunk  # optional (helps clarity, not always required)
    
    # Plot the ram and itr
    fig, ax = plt.subplots(1, 1)
    ax.plot(itr, ram)
    ax.grid(alpha=0.5)
    plt.savefig("ram.png")

    return tracker.peak


iterate_with_memmap(filepath)