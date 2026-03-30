"""
LLMModel
======================
Wrapper around HuggingFace causal LM with 4-bit quantization,
batched inference, and model-specific chat template handling.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Assistant-marker registry (pure lookup, no instance needed)
# ---------------------------------------------------------------------------

_ASSISTANT_MARKERS: dict[str, str] = {
    "llama": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "deepseek": "</think>\n\n",
    "mistral": "[/INST] "
}


def _resolve_assistant_marker(model_name: str) -> str:
    """Return the assistant-turn marker for a known model family."""
    name = model_name.lower()
    for key, marker in _ASSISTANT_MARKERS.items():
        if key in name:
            return marker
    raise ValueError(
        f"Unknown model family for assistant marker: {model_name}. "
        f"Known families: {list(_ASSISTANT_MARKERS)}"
    )


# ---------------------------------------------------------------------------
# Prompt formatting (pure)
# ---------------------------------------------------------------------------

def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    """DeepSeek has no system role — fold system content into the user message."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "").strip()
    user = next((m["content"] for m in messages if m["role"] == "user"), "").strip()
    combined = f"{system}\n\n{user}" if system else user
    return [{"role": "user", "content": combined}]


def format_prompt(messages_or_text, tokenizer) -> str:
    """Convert raw text or chat-style messages into a tokenizer-ready string."""
    if isinstance(messages_or_text, str):
        return messages_or_text

    if not (
        isinstance(messages_or_text, list)
        and all("role" in m and "content" in m for m in messages_or_text)
    ):
        raise ValueError("Prompt must be a string or a list of {role, content} dicts.")

    messages = messages_or_text
    if "deepseek" in tokenizer.name_or_path.lower():
        messages = _merge_system_into_user(messages)

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        raise ValueError("Tokenizer does not support chat templates.") from e


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def log_gpu_memory() -> None:
    """Log per-device GPU memory stats."""
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        res = torch.cuda.memory_reserved(i) / 1e9
        logger.info(f"GPU {i}: allocated={alloc:.2f} GB, reserved={res:.2f} GB")


def free_gpu_memory() -> None:
    """Aggressively release GPU memory."""
    if not torch.cuda.is_available():
        return
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    reserved = torch.cuda.memory_reserved(0) / 1024**2
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    total = torch.cuda.get_device_properties(0).total_mem / 1024**2
    logger.info(
        f"GPU Memory — Total: {total:.0f} MB, Reserved: {reserved:.0f} MB, "
        f"Allocated: {allocated:.0f} MB, Free: {reserved - allocated:.0f} MB"
    )


# ---------------------------------------------------------------------------
# LLMModel
# ---------------------------------------------------------------------------

class LLMModel:
    """4-bit quantised causal LM with batched generation."""

    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 512,
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.max_new_tokens = max_new_tokens
        self.hf_token = hf_token

        self._tokenizer = None
        self._model = None
        self._eos_token_id: int | None = None
        self._assistant_marker: str = ""

        self._load_model()

    # ---- model lifecycle ---------------------------------------------------

    def _load_model(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, token=self.hf_token
        )

        self._eos_token_id = self._tokenizer.eos_token_id

        # Ensure a neutral pad token
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            else:
                self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self._eos_token_id = self._tokenizer.convert_tokens_to_ids("[PAD]")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=None,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=self.hf_token,
        )

        accelerator = Accelerator(mixed_precision="fp16")
        self._model = accelerator.prepare(self._model)

        self._assistant_marker = _resolve_assistant_marker(self.model_path)

    def unload(self) -> None:
        """Release model from GPU."""
        del self._model
        self._model = None
        torch.cuda.empty_cache()
        gc.collect()

    def reload(self) -> None:
        """Full unload → reload cycle (useful for long DeepSeek runs)."""
        self.unload()
        self._load_model()

    # ---- generation --------------------------------------------------------

    def _generate(
        self,
        prompts: list,
        *,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 0.0,
    ) -> list[str]:
        """Run generation on a list of prompts and return decoded assistant text."""
        formatted = [format_prompt(p, self._tokenizer) for p in prompts]

        padding = len(formatted) > 1
        inputs = self._tokenizer(
            formatted,
            return_tensors="pt",
            padding_side="left" if padding else "right",
            padding=padding,
            truncation=True,
        ).to(self._model.device)

        logger.info(f"Input token length: {inputs['input_ids'].shape[-1]}")

        gen_kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self._eos_token_id,
            "pad_token_id": self._eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        decoded = self._tokenizer.batch_decode(output_ids)
        return [self._extract_assistant_response(seq) for seq in decoded]

    def _extract_assistant_response(self, sequence: str) -> str:
        """Split on the assistant marker and clean up special tokens."""
        parts = sequence.split(self._assistant_marker)
        text = parts[1] if len(parts) > 1 else parts[0]
        return (
            text.replace(self._tokenizer.pad_token, "")
            .replace("<|eot_id|>", "")
            .strip()
        )

    # ---- batch processing --------------------------------------------------

    def _process_batch(self, batch: list[dict]) -> list[dict]:
        prompts = [item["prompt"] for item in batch]
        ids = [item["id"] for item in batch]
        try:
            responses = self._generate(prompts)
            return [{id_: resp} for id_, resp in zip(ids, responses)]
        except Exception as e:
            logger.error(f"Batch error: {e}")
            return [{id_: None} for id_ in ids]

    def process_all_batches(
        self, prompts: list[dict], batch_size: int
    ) -> list[dict]:
        """Run inference over all prompts in batches, returning [{id: response}]."""
        results: list[dict] = []
        is_deepseek = "deepseek" in self._tokenizer.name_or_path.lower()

        for batch_idx, start in enumerate(
            tqdm(range(0, len(prompts), batch_size), desc="Batches")
        ):
            batch = prompts[start : start + batch_size]
            results.extend(self._process_batch(batch))

            # DeepSeek memory-leak workaround
            if is_deepseek and (batch_idx + 1) % 50 == 0:
                self.reload()

        return results
