"""
pipeline_utils — Shared helpers for PREVIA and zero-shot pipelines.
===================================================================
GPU memory management, config loading, I/O, and output formatting.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from LLMModel import LLMModel


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def free_gpu_memory() -> None:
    """Release cached GPU memory and log current usage."""
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.empty_cache()
    props = torch.cuda.get_device_properties(0)
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    print(
        f"GPU Memory — Total: {props.total_memory / 1024**2:.0f} MB, "
        f"Reserved: {reserved / 1024**2:.0f} MB, "
        f"Allocated: {allocated / 1024**2:.0f} MB, "
        f"Free: {(reserved - allocated) / 1024**2:.0f} MB"
    )


# ---------------------------------------------------------------------------
# Config / I/O
# ---------------------------------------------------------------------------

def load_config(config_dir: Path, deployment: str = "hpc") -> tuple[dict, str]:
    """Load ``config.json`` and ``tokens.json`` from *config_dir*."""
    with open(config_dir / "config.json") as f:
        config_dict = json.load(f)[deployment]
    with open(config_dir / "tokens.json") as f:
        hf_token = json.load(f)["HF_TOKEN"]
    return config_dict, hf_token


def load_transcripts(base_path: Path, file_end_name: str) -> pd.DataFrame:
    csv_path = base_path / "data" / f"{file_end_name}.csv"
    return pd.read_csv(csv_path, index_col="record_id")


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path)
    print(f"Saved → {path}")


def result_path(results_dir: Path, prefix: str, *parts: str) -> Path:
    return results_dir / (prefix + "_".join(parts) + ".csv")


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def batch_size_for(llm_model: str, default: int) -> int:
    return 1 if "deepseek" in llm_model else default


def max_tokens_for(llm_model: str, default: int) -> int:
    return 2048 if "deepseek" in llm_model else default


# ---------------------------------------------------------------------------
# Output reshaping
# ---------------------------------------------------------------------------

def get_participant_ids(df: pd.DataFrame) -> pd.Index:
    if isinstance(df.index, pd.MultiIndex):
        return df.index.get_level_values("record_id").unique()
    return df.index.unique()


def flatten_agent_outputs(multi_agents_results: list[dict]) -> pd.DataFrame:
    """Convert ``[{record_id_agentName: output}, ...]`` to a pivoted DataFrame."""
    rows = []
    for agent_dict in multi_agents_results:
        key, output = next(iter(agent_dict.items()))
        record_id, agent_name = key.split("_", 1)
        rows.append((int(record_id), agent_name, output))
    df = pd.DataFrame(rows, columns=["record_id", "agent", "output"])
    return df.pivot(index="record_id", columns="agent", values="output")
