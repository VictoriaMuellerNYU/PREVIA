"""
PREVIA Pipeline
===============
Three-layer LLM ensemble for early PTSD risk prediction from ED trauma narratives.

Architecture:
  Layer 1: Parallel risk assessments via two LLMs
  Layer 2: Summarizer that reconciles Layer 1 outputs (scores replaced by true means)
  Layer 3: Final clinical/subclinical assessment with demographics

Usage:
  python previa_pipeline.py --config-dir ./configs --outcome clinical
  python previa_pipeline.py --config-dir ./configs --outcome clinical --skip-layer1
  python previa_pipeline.py --config-dir ./configs --outcome clinical --skip-layer1 --skip-layer2
"""

from __future__ import annotations

import argparse
import re
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
class PreviaConfig:
    """Immutable configuration for the three-layer PREVIA pipeline."""

    # Paths
    base_path: Path
    results_path: Path
    config_dict: dict
    hf_token: str

    # Data
    file_end_name: str = "llm_test"
    transcript_col: str = "text_whisper"
    demo_columns: tuple[str, ...] = ("age", "sex_at_birth", "trauma_type")

    # Outcome
    outcome: str = "clinical"

    # Prompt IDs
    layer1_prompt_id: str = "layer1_prompts"
    layer2_prompt_id: str = "layer2_prompts"

    # LLMs
    layer1_llms: tuple[str, str] = ("llama33_70B", "mistral_large")
    layer2_llm: str = "deepseek_70B"
    layer3_llm: str = "mistral_large"

    # Generation
    default_max_new_tokens: int = 512
    default_batch_size: int = 16

    # Layer 3
    include_demographics: bool = True
    include_prevalence: bool = True

    @property
    def layer3_prompt_id(self) -> str:
        return (
            "layer3_prompts_clinical"
            if self.outcome == "clinical"
            else "layer3_prompts_subclinical"
        )

    def _result_path(self, *parts: str) -> Path:
        return result_path(self.results_path, "PREVIA_", *parts)

    @staticmethod
    def from_args(args: argparse.Namespace) -> "PreviaConfig":
        config_dict, hf_token = load_config(Path(args.config_dir))
        base_path = Path(config_dict["base_path"])
        return PreviaConfig(
            base_path=base_path,
            results_path=base_path / "results",
            config_dict=config_dict,
            hf_token=hf_token,
            file_end_name=args.file_end_name,
            outcome=args.outcome,
            include_demographics=args.include_demographics,
            include_prevalence=args.include_prevalence,
        )


# ---------------------------------------------------------------------------
# Score extraction / replacement (pure)
# ---------------------------------------------------------------------------

def _extract_risk_score(text: str) -> Optional[int]:
    pattern = r'(?:\*\*)?PTSD risk score:(?:\*\*)?\s*["\\[]?(\d+)[\\]"]?'
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _replace_risk_score(text: str, new_score: float) -> str:
    pattern = r'((?:\*\*)?PTSD risk score:(?:\*\*)?\s*)["\\[]?\d+[\\]"]?'
    return re.sub(pattern, rf'\1"{new_score}"', text)


def compute_mean_risk_scores(
    layer1_output: pd.DataFrame,
    summarizer_df: pd.DataFrame,
    model_names: tuple[str, str],
    score_columns: tuple[str, ...] = (
        "ED-PTSD-symptoms",
        "peri-trauma-symptoms",
        "protective-factors",
        "risk-factors",
    ),
) -> pd.DataFrame:
    """
    Replace the summarizer's LLM-generated risk scores with the arithmetic
    mean of the two Layer-1 models' scores.
    """
    model_a, model_b = model_names

    score_series: dict[str, pd.Series] = {}
    for col in score_columns:
        scores_a = (
            layer1_output.loc[layer1_output.index.get_level_values("model") == model_a, col]
            .apply(_extract_risk_score)
            .reset_index(drop=True)
        )
        scores_b = (
            layer1_output.loc[layer1_output.index.get_level_values("model") == model_b, col]
            .apply(_extract_risk_score)
            .reset_index(drop=True)
        )
        score_series[col] = ((scores_a + scores_b) / 2).round(1)

    updated = summarizer_df.copy()
    for col, means in score_series.items():
        updated[col] = [
            _replace_risk_score(text, score)
            for text, score in zip(updated[col], means)
        ]

    combined = pd.concat([updated, layer1_output.reset_index()])
    combined = combined.set_index(["record_id", "model"]).sort_index()
    return combined


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompts(
    df: pd.DataFrame,
    prompt_id: str,
    *,
    include_demographics: bool = False,
    include_prevalence: bool = False,
    transcript_col: str = "text_whisper",
    layer2_summary: bool = False,
    layer3_assessment: bool = False,
    layer1_llms: Optional[tuple[str, ...]] = None,
) -> list:
    prompts_path = Path(__file__).parent / "configs" / "previa_prompts.json"
    builder = PromptBuilder(
        df,
        prompts_path=prompts_path,
        prompt_id=prompt_id,
        include_demographics=include_demographics,
        include_prevalence=include_prevalence,
    )
    ids = get_participant_ids(df)

    if layer2_summary:
        return builder.prepare_prompts_layer2(
            ids, layer1_assessor_names=df.columns, layer1_llms=layer1_llms
        )
    if layer3_assessment:
        return builder.prepare_prompts_layer3(ids)
    return builder.prepare_prompts_layer1(ids, transcript_col=transcript_col)


# ---------------------------------------------------------------------------
# LLM execution (single layer)
# ---------------------------------------------------------------------------

def _run_llm(
    cfg: PreviaConfig,
    df_input: pd.DataFrame,
    llm_model: str,
    prompt_id: str,
    *,
    include_demographics: bool = False,
    include_prevalence: bool = False,
    transcript_col: Optional[str] = None,
    layer2_summary: bool = False,
    layer3_assessment: bool = False,
    layer1_llms: Optional[tuple[str, ...]] = None,
    max_new_tokens: Optional[int] = None,
) -> pd.DataFrame:
    """Run a single LLM pass and return a tidy DataFrame of outputs."""
    free_gpu_memory()

    bs = batch_size_for(llm_model, cfg.default_batch_size)
    tokens = max_tokens_for(llm_model, max_new_tokens or cfg.default_max_new_tokens)

    model_path = Path(cfg.config_dict[f"{llm_model}_path"])
    llm = LLMModel(model_path, max_new_tokens=tokens, hf_token=cfg.hf_token)

    prompts = _build_prompts(
        df_input,
        prompt_id,
        include_demographics=include_demographics,
        include_prevalence=include_prevalence,
        transcript_col=transcript_col or cfg.transcript_col,
        layer2_summary=layer2_summary,
        layer3_assessment=layer3_assessment,
        layer1_llms=layer1_llms,
    )
    print(f"[{llm_model}] {len(prompts)} prompts, batch_size={bs}")

    raw_output = llm.process_all_batches(prompts=prompts, batch_size=bs)
    output_df = flatten_agent_outputs(raw_output)

    if not layer3_assessment:
        output_df["model"] = "summarizer" if layer2_summary else llm_model
        output_df = output_df.set_index("model", append=True)

    return output_df


# ---------------------------------------------------------------------------
# Pipeline layers
# ---------------------------------------------------------------------------

def run_layer1(cfg: PreviaConfig, transcripts: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for llm_name in cfg.layer1_llms:
        out = _run_llm(cfg, transcripts, llm_name, cfg.layer1_prompt_id, transcript_col=cfg.transcript_col)
        save(out, cfg._result_path(cfg.layer1_prompt_id, cfg.transcript_col, llm_name, cfg.file_end_name))
        outputs.append(out)
    return pd.concat(outputs).sort_index()


def load_layer1(cfg: PreviaConfig) -> pd.DataFrame:
    frames = []
    for llm_name in cfg.layer1_llms:
        path = cfg._result_path(cfg.layer1_prompt_id, cfg.transcript_col, llm_name, cfg.file_end_name)
        frames.append(pd.read_csv(path, index_col=["record_id", "model"]))
    return pd.concat(frames).sort_index()


def run_layer2(cfg: PreviaConfig, layer1_output: pd.DataFrame, transcripts: pd.DataFrame) -> pd.DataFrame:
    summarizer_output = _run_llm(
        cfg, layer1_output, cfg.layer2_llm, cfg.layer2_prompt_id,
        layer2_summary=True, layer1_llms=cfg.layer1_llms,
    )

    combined = compute_mean_risk_scores(layer1_output, summarizer_output.reset_index(), model_names=cfg.layer1_llms)

    save(
        combined,
        cfg._result_path(
            cfg.layer1_prompt_id, *cfg.layer1_llms, cfg.transcript_col,
            "summary", cfg.layer2_prompt_id, cfg.layer2_llm, cfg.file_end_name,
        ),
    )
    return combined


def load_layer2(cfg: PreviaConfig) -> pd.DataFrame:
    path = cfg._result_path(
        cfg.layer1_prompt_id, *cfg.layer1_llms, cfg.transcript_col,
        "summary", cfg.layer2_prompt_id, cfg.layer2_llm, cfg.file_end_name,
    )
    score_cols = ["ED-PTSD-symptoms", "peri-trauma-symptoms", "risk-factors", "protective-factors"]
    return pd.read_csv(path, index_col=["record_id", "model"])[score_cols]


def run_layer3(cfg: PreviaConfig, layer2_output: pd.DataFrame, transcripts: pd.DataFrame) -> pd.DataFrame:
    layer2_output_slice = layer2_output.xs("summarizer", level="model")
    summarizer_slice = layer2_output_slice.join(
        transcripts[list(cfg.demo_columns)], on="record_id", how="right"
    )

    output = _run_llm(
        cfg, summarizer_slice, cfg.layer3_llm, cfg.layer3_prompt_id,
        layer3_assessment=True,
        include_demographics=cfg.include_demographics,
        include_prevalence=cfg.include_prevalence,
    )
    output = output.join(transcripts, on="record_id", how="right")

    save(
        output,
        cfg._result_path(
            cfg.transcript_col, cfg.file_end_name,
            cfg.layer1_prompt_id, *cfg.layer1_llms,
            cfg.layer2_prompt_id, cfg.layer2_llm,
            cfg.layer3_prompt_id, cfg.layer3_llm,
            f"demo-{cfg.include_demographics}", f"prev-{cfg.include_prevalence}",
            "mean", cfg.file_end_name,
        ),
    )
    return output


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_previa(cfg: PreviaConfig) -> pd.DataFrame:
    transcripts = load_transcripts(cfg.base_path, cfg.file_end_name)
    layer1_output = run_layer1(cfg, transcripts)
    layer2_output = run_layer2(cfg, layer1_output, transcripts)
    return run_layer3(cfg, layer2_output, transcripts)


def run_previa_from_cache(
    cfg: PreviaConfig, *, skip_layer1: bool = False, skip_layer2: bool = False
) -> pd.DataFrame:
    transcripts = load_transcripts(cfg.base_path, cfg.file_end_name)

    if skip_layer2:
        layer2_output = load_layer2(cfg)
    else:
        layer1_output = load_layer1(cfg) if skip_layer1 else run_layer1(cfg, transcripts)
        layer2_output = run_layer2(cfg, layer1_output, transcripts)

    return run_layer3(cfg, layer2_output, transcripts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PREVIA three-layer LLM pipeline")
    p.add_argument("--config-dir", type=str, default="..", help="Directory with config.json and tokens.json")
    p.add_argument("--outcome", choices=["clinical", "subclinical"], default="clinical")
    p.add_argument("--file-end-name", default="llm_test", help="Suffix identifying the input CSV")
    p.add_argument("--skip-layer1", action="store_true", help="Load cached Layer 1 results")
    p.add_argument("--skip-layer2", action="store_true", help="Load cached Layer 1+2 results")
    p.add_argument("--no-demographics", dest="include_demographics", action="store_false")
    p.add_argument("--no-prevalence", dest="include_prevalence", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreviaConfig.from_args(args)

    if args.skip_layer1 or args.skip_layer2:
        run_previa_from_cache(cfg, skip_layer1=args.skip_layer1, skip_layer2=args.skip_layer2)
    else:
        run_previa(cfg)


if __name__ == "__main__":
    main()
