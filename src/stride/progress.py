from __future__ import annotations

import logging
import os
import time
import warnings
from contextlib import contextmanager
from collections.abc import Iterator

from tqdm import tqdm


def configure_library_output(show_warnings: bool = False) -> None:
    """Hide optional library chatter while retaining errors and ETA bars."""
    if show_warnings:
        return
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore")
    for name in (
        "transformers",
        "huggingface_hub",
        "tokenizers",
        "bitsandbytes",
        "accelerate",
        "datasets",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except ImportError:
        pass


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


@contextmanager
def timed_stage(label: str) -> Iterator[None]:
    """Print an explicit start/completion marker around non-iterative work."""
    start = time.perf_counter()
    tqdm.write(f"[start] {label}")
    try:
        yield
    except Exception:
        tqdm.write(f"[fail]  {label} ({format_duration(time.perf_counter() - start)})")
        raise
    else:
        tqdm.write(f"[done]  {label} ({format_duration(time.perf_counter() - start)})")
