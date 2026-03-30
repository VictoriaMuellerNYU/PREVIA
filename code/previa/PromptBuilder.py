"""
PromptBuilder
===========================
Constructs structured prompts for each PREVIA pipeline layer.

Key changes from original:
- Layer-specific logic is in dedicated methods, not gated by mutable boolean flags.
- No instance-level mutation of `self.summary`, `self.final_assessment`, etc.
- Demographics/prevalence injection is explicit and parameterised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers (pure)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_text(path: Path) -> str:
    with open(path) as f:
        return f.read()


def _format_demographics(df: pd.DataFrame, participant_id) -> str:
    row = df.loc[participant_id]
    sex = str(row["sex_at_birth"]).lower()
    age = row["age"]
    trauma = str(row["trauma_type"]).lower()
    return f"The patient is {sex}, {age} years old and experienced a {trauma}."


def _build_chat_prompt(
    participant_id,
    agent_type: str,
    system: str,
    user_content: str,
) -> dict:
    """Return the standard ``{id, prompt}`` dict consumed by ``LLMModel``."""
    system_text = system.replace("{{agent_type}}", agent_type)
    return {
        "id": f"{participant_id}_{agent_type}",
        "prompt": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ],
    }


SCORE_AGENT_COLUMNS = (
    "ED-PTSD-symptoms",
    "peri-trauma-symptoms",
    "risk-factors",
    "protective-factors",
)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Builds structured prompts for each PREVIA layer."""

    def __init__(
        self,
        df: pd.DataFrame,
        prompts_path: Path,
        prompt_id: str,
        *,
        include_demographics: bool = False,
        include_prevalence: bool = False,
    ) -> None:
        prompt_data = _load_json(prompts_path)
        self._agents = prompt_data[prompt_id]
        self._df = df
        self._include_demographics = include_demographics
        self._include_prevalence = include_prevalence
        self._prevalence_info: Optional[str] = (
            _load_text(prompts_path.parent / "PTSD_prevalence.txt")
            if include_prevalence
            else None
        )

    # ------------------------------------------------------------------
    # Additional-information block (shared across layers)
    # ------------------------------------------------------------------

    def _additional_info(self, df: pd.DataFrame, participant_id) -> str:
        parts: list[str] = []
        if self._include_demographics:
            parts.append(f"DEMOGRAPHICS\n{_format_demographics(df, participant_id)}")
        if self._include_prevalence and self._prevalence_info:
            parts.append(f"PTSD PREVALENCE\n{self._prevalence_info}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Layer 1: Individual risk-factor assessments from transcripts
    # ------------------------------------------------------------------

    def _user_prompt_layer1(
        self, participant_id, task: str, transcript_col: str
    ) -> str:
        transcript = self._df.loc[participant_id, transcript_col]
        return f"TASK\n{task}\n---\nPATIENT's NARRATIVE\n{transcript}\n"

    def prepare_prompts_layer1(
        self, participant_ids, transcript_col: str = "text_whisper"
    ) -> list[dict]:
        results: list[dict] = []
        for agent in self._agents:
            system, task, agent_type = agent["system"], agent["user"], agent["type"]
            for pid in participant_ids:
                user = self._user_prompt_layer1(pid, task, transcript_col)
                results.append(_build_chat_prompt(pid, agent_type, system, user))
        return results

    # ------------------------------------------------------------------
    # Zero-shot: standalone single-pass assessment with demographics
    # ------------------------------------------------------------------

    def _user_prompt_zero_shot(
        self, participant_id, task: str, transcript_col: str
    ) -> str:
        transcript = self._df.loc[participant_id, transcript_col]
        prompt = f"TASK\n{task}\n---\nPATIENT's NARRATIVE\n{transcript}\n"
        extra = self._additional_info(self._df, participant_id)
        if extra:
            prompt += extra
        return prompt

    def prepare_prompts_zero_shot(
        self, participant_ids, transcript_col: str = "text_whisper"
    ) -> list[dict]:
        results: list[dict] = []
        for agent in self._agents:
            system, task, agent_type = agent["system"], agent["user"], agent["type"]
            for pid in participant_ids:
                user = self._user_prompt_zero_shot(pid, task, transcript_col)
                results.append(_build_chat_prompt(pid, agent_type, system, user))
        return results

    # ------------------------------------------------------------------
    # Layer 2: Summarise two Layer-1 assessments per agent
    # ------------------------------------------------------------------

    def _user_prompt_layer2(
        self,
        participant_id,
        task: str,
        agent_type: str,
        model_names: tuple[str, str],
    ) -> str:
        task_filled = task.replace("{{agent_type}}", agent_type)
        output_a = self._df.xs(model_names[0], level="model").loc[participant_id, agent_type]
        output_b = self._df.xs(model_names[1], level="model").loc[participant_id, agent_type]
        return (
            f"TASK\n{task_filled}\n---\n"
            f"Agent A:\n{output_a}\n\nAgent B:\n{output_b}\n"
        )

    def prepare_prompts_layer2(
        self,
        participant_ids,
        layer1_assessor_names,
        layer1_llms: tuple[str, str],
    ) -> list[dict]:
        agent = self._agents[0]
        system, task = agent["system"], agent["user"]
        results: list[dict] = []
        for agent_name in layer1_assessor_names:
            for pid in participant_ids:
                user = self._user_prompt_layer2(pid, task, agent_name, layer1_llms)
                results.append(_build_chat_prompt(pid, agent_name, system, user))
        return results

    # ------------------------------------------------------------------
    # Layer 3: Final clinical assessment from summaries + demographics
    # ------------------------------------------------------------------

    def _user_prompt_layer3(self, participant_id, task: str) -> str:
        summary_block = ""
        for col in SCORE_AGENT_COLUMNS:
            text = self._df.loc[participant_id, col]
            summary_block += f"Agent: {col}\n{text}\n\n"

        prompt = f"TASK\n{task}\n---\nSUMMARY AGENTS\n{summary_block}"
        extra = self._additional_info(self._df, participant_id)
        if extra:
            prompt += extra
        return prompt

    def prepare_prompts_layer3(self, participant_ids) -> list[dict]:
        agent = self._agents[0]
        system, task = agent["system"], agent["user"]
        return [
            _build_chat_prompt(pid, "final-assessment", system, self._user_prompt_layer3(pid, task))
            for pid in participant_ids
        ]
