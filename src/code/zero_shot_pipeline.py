"""
Zero-Shot Pipeline
==================
Standalone single-pass PTSD risk assessment from ED trauma narratives.
Demographics and prevalence information are injected directly into the prompt.
No multi-layer ensemble — this is the baseline comparator for PREVIA.

Usage:
  python zero_shot_pipeline.py --config-dir ./configs --outcome clinical --llm mistral_large
  python zero_shot_pipeline.py --config-dir ./configs --outcome subclinical --llm llama33_70B
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from LLMModel import LLMModel
from PromptBuilder import PromptBuilder
from pipeline_utils import (
    batch_size_for,
    flatten_agent_outputs,
    free_gpu_memory,
    get_participant_ids,
    load_config,
    load_transcripts,
    max_tokens_for,
    result_path,
    save,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZeroShotConfig:
    """Immutable configuration for the zero-shot baseline pipeline."""

    # Paths
    base_path: Path
    results_path: Path
    config_dict: dict
    hf_token: str

    # Data
    file_end_name: str = "llm_test"
    transcript_col: str = "text_whisper"

    # Outcome
    outcome: str = "clinical"

    # LLM
    llm_model: str = "mistral_large"

    # Generation
    default_max_new_tokens: int = 512
    default_batch_size: int = 16

    # Demographics / prevalence (always on for zero-shot)
    include_demographics: bool = True
    include_prevalence: bool = True

    @property
    def prompt_id(self) -> str:
        return (
            "zero_shot_clinical"
            if self.outcome == "clinical"
            else "zero_shot_subclinical"
        )

    def _result_path(self, *parts: str) -> Path:
        return result_path(self.results_path, "ZeroShot_", *parts)

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ZeroShotConfig":
        config_dict, hf_token = load_config(Path(args.config_dir))
        base_path = Path(config_dict["base_path"])
        return ZeroShotConfig(
            base_path=base_path,
            results_path=base_path / "results",
            config_dict=config_dict,
            hf_token=hf_token,
            file_end_name=args.file_end_name,
            outcome=args.outcome,
            llm_model=args.llm,
        )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompts(
    df: pd.DataFrame,
    prompt_id: str,
    transcript_col: str = "text_whisper",
) -> list[dict]:
    prompts_path = Path(__file__).parent / "configs" / "zero_shot_prompt.json"
    builder = PromptBuilder(
        df,
        prompts_path=prompts_path,
        prompt_id=prompt_id,
        include_demographics=True,
        include_prevalence=True,
    )
    ids = get_participant_ids(df)
    return builder.prepare_prompts_zero_shot(ids, transcript_col=transcript_col)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_zero_shot(cfg: ZeroShotConfig) -> pd.DataFrame:
    """Run a single-pass zero-shot assessment and save results."""
    free_gpu_memory()

    transcripts = load_transcripts(cfg.base_path, cfg.file_end_name)

    bs = batch_size_for(cfg.llm_model, cfg.default_batch_size)
    tokens = max_tokens_for(cfg.llm_model, cfg.default_max_new_tokens)

    model_path = Path(cfg.config_dict[f"{cfg.llm_model}_path"])
    llm = LLMModel(model_path, max_new_tokens=tokens, hf_token=cfg.hf_token)

    prompts = _build_prompts(transcripts, cfg.prompt_id, transcript_col=cfg.transcript_col)
    print(f"[{cfg.llm_model}] {len(prompts)} prompts, batch_size={bs}")

    raw_output = llm.process_all_batches(prompts=prompts, batch_size=bs)
    output_df = flatten_agent_outputs(raw_output)

    output_df["model"] = cfg.llm_model
    output_df = output_df.set_index("model", append=True)

    save(
        output_df,
        cfg._result_path(cfg.prompt_id, cfg.transcript_col, cfg.llm_model),
    )
    return output_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot PTSD risk assessment")
    p.add_argument("--config-dir", type=str, 
                default=str(Path(__file__).parent.parent),
                help="Directory with config.json and tokens.json")
    p.add_argument("--outcome", choices=["clinical", "subclinical"], default="clinical")
    p.add_argument("--llm", type=str, default="mistral_large", help="e.g. llama33_70B, mistral_large")
    p.add_argument("--file-end-name", default="llm_CAPS_PCL1m_n145_allAvailData", help="Suffix identifying the input CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ZeroShotConfig.from_args(args)
    run_zero_shot(cfg)


if __name__ == "__main__":
    main()
