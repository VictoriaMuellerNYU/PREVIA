# PREVIA

**Clinically informed LLM ensemble for early PTSD risk prediction from emergency department trauma narratives.**

PREVIA processes transcribed patient narratives through a three-layer architecture that combines multiple LLMs into a structured risk assessment. A zero-shot baseline pipeline is included for comparison.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PREVIA Pipeline                               │
│                                                                      │
│  ┌───────────────────── Layer 1 ───────────────────────┐             │
│  │                                                     │             │
│  │   Transcript ──┬──► LLM A (e.g. Llama 3.3 70B) ───┐ │             │
│  │                │                                  ├─►  Agent      │
│  │                └──► LLM B (e.g. Mistral Large) ───┘ │  outputs    │
│  │                                                     │  per risk   │
│  │   Agents: ED-PTSD-symptoms, peri-trauma-symptoms,   │  domain     │
│  │           risk-factors, protective-factors          │             │
│  └─────────────────────────┬───────────────────────────┘             │
│                            │                                         │
│  ┌──────────── Layer 2 ────▼───────────────┐                         │
│  │                                         │                         │
│  │   Summarizer LLM (e.g. DeepSeek 70B)    │                         │
│  │   Reconciles LLM A + LLM B outputs      │                         │
│  │   Risk scores replaced with true means  │                         │
│  └─────────────────────┬───────────────────┘                         │
│                        │                                             │
│  ┌──────── Layer 3 ────▼───────────────────┐                         │
│  │                                         │                         │
│  │   Final assessor LLM                    │                         │
│  │   + Demographics (age, sex, trauma)     │                         │
│  │   + PTSD prevalence data                │                         │
│  │   ──► Clinical / subclinical decision   │                         │
│  └─────────────────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     Zero-Shot Baseline                               │
│                                                                      │
│   Transcript + Demographics + Prevalence ──► Single LLM ──► Output   │
└──────────────────────────────────────────────────────────────────────┘
```

**Layer 1** runs each transcript through two LLMs independently. Each LLM evaluates four clinical domains and assigns a risk score per domain.

**Layer 2** uses a third LLM to reconcile the two Layer-1 outputs into a single summary per domain. The LLM-generated risk scores are then replaced with the arithmetic mean of the Layer-1 scores for reliability.

**Layer 3** takes the summarized assessments, adds patient demographics and PTSD base-rate information, and produces a final clinical or subclinical risk prediction.

The **zero-shot baseline** collapses this into a single LLM pass with all available information in the prompt.

---

## Models

PREVIA uses open-weight LLMs from Hugging Face, run locally with 4-bit quantisation (bitsandbytes). No data is sent to external APIs.

| Pipeline role | Model | Parameters | License | Hugging Face |
|---|---|---|---|---|
| Layer 1 — Risk assessor A | Llama 3.3 70B Instruct | 70B | [Llama 3.3 Community](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE) | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| Layer 1 — Risk assessor B | Mistral Large Instruct 2411 | 123B | [Mistral Research License](https://mistral.ai/licenses/MRL-0.1.md) | [mistralai/Mistral-Large-Instruct-2411](https://huggingface.co/mistralai/Mistral-Large-Instruct-2411) |
| Layer 2 — Summarizer | DeepSeek-R1-Distill-Llama-70B | 70B | [MIT](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/blob/main/LICENSE) | [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| Layer 3 — Final assessor | Mistral Large Instruct 2411 | 123B | [Mistral Research License](https://mistral.ai/licenses/MRL-0.1.md) | [mistralai/Mistral-Large-Instruct-2411](https://huggingface.co/mistralai/Mistral-Large-Instruct-2411) |
| Zero-shot baseline | Configurable (default: Mistral Large) | — | — | See above |

---

## Project structure

```
├── previa_pipeline.py      # Three-layer PREVIA pipeline (CLI entrypoint)
├── zero_shot_pipeline.py   # Zero-shot baseline pipeline (CLI entrypoint)
├── pipeline_utils.py       # Shared helpers (GPU, I/O, config, output formatting)
├── LLMModel.py             # HuggingFace causal LM wrapper (4-bit quantised)
├── PromptBuilder.py        # Prompt construction for all layers + zero-shot
├── requirements.txt        # Pinned Python dependencies
└── configs/
    ├── config.json          # Model paths and base data path (per deployment)
    ├── tokens.json          # HuggingFace API token
    ├── previa_prompts.json  # System/user prompts for Layers 1-3
    ├── zero_shot_prompt.json# System/user prompts for zero-shot
    └── PTSD_prevalence.txt  # Base-rate prevalence text injected into prompts
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU (tested with A100 80 GB)

### Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins all core dependencies (PyTorch 2.3 with CUDA 11.8, Transformers 4.51, Accelerate 1.6, bitsandbytes 0.45). PyTorch packages are pulled from the CUDA 11.8 wheel index automatically via `--extra-index-url`.

Typical install time is approximately 5-10 minutes on a standard internet connection.

> **Note**: `bitsandbytes` requires a Linux environment with CUDA. See the [bitsandbytes installation guide](https://github.com/TimDettmers/bitsandbytes) for platform-specific instructions.

---

## Configuration

### `configs/config.json`

Keyed by deployment environment (default: `"hpc"`). Each entry specifies the base data path and model paths:

```json
{
  "hpc": {
    "base_path": "/path/to/project",
    "llama33_70B_path": "/path/to/models/Llama-3.3-70B-Instruct",
    "mistral_large_path": "/path/to/models/Mistral-Large-Instruct",
    "deepseek_70B_path": "/path/to/models/DeepSeek-R1-Distill-Llama-70B"
  }
}
```

The `base_path` directory is expected to contain:

```
base_path/
├── data/
│   └── <file_end_name>.csv    # Transcripts with record_id index
└── results/                   # Pipeline outputs saved here
```

### `configs/tokens.json`

```json
{
  "HF_TOKEN": "hf_..."
}
```

### Prompt JSON files

Both `previa_prompts.json` and `zero_shot_prompt.json` follow the same structure. Each top-level key is a prompt set ID (e.g., `"layer1_prompts"`, `"zero_shot_clinical"`), mapping to a list of agent definitions:

```json
{
  "layer1_prompts": [
    {
      "type": "ED-PTSD-symptoms",
      "system": "You are a clinical psychologist specialising in ...",
      "user": "TASK\nEvaluate the following narrative for ..."
    }
  ]
}
```

The `{{agent_type}}` placeholder in system/user prompts is replaced at runtime with the agent's `type` value.

---

## Usage

### PREVIA pipeline (three-layer ensemble)

```bash
# Full pipeline
python previa_pipeline.py --config-dir ./configs --outcome clinical

# Resume from cached Layer 1 results
python previa_pipeline.py --config-dir ./configs --outcome clinical --skip-layer1

# Resume from cached Layer 1 + 2 results
python previa_pipeline.py --config-dir ./configs --outcome clinical --skip-layer1 --skip-layer2

# Subclinical outcome, no prevalence injection
python previa_pipeline.py --config-dir ./configs --outcome subclinical --no-prevalence
```

**PREVIA CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--config-dir` | `..` | Directory containing `config.json` and `tokens.json` |
| `--outcome` | `clinical` | `clinical` or `subclinical` |
| `--file-end-name` | `llm_test` | Suffix of the input CSV in `data/` |
| `--skip-layer1` | off | Load cached Layer 1 results instead of rerunning |
| `--skip-layer2` | off | Load cached Layer 1+2 results |
| `--no-demographics` | off | Omit demographics from Layer 3 prompt |
| `--no-prevalence` | off | Omit PTSD prevalence from Layer 3 prompt |

### Zero-shot baseline

```bash
# Clinical outcome with Mistral Large
python zero_shot_pipeline.py --config-dir ./configs --outcome clinical --llm mistral_large

# Subclinical outcome with Llama
python zero_shot_pipeline.py --config-dir ./configs --outcome subclinical --llm llama33_70B
```

**Zero-shot CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--config-dir` | `..` | Directory containing `config.json` and `tokens.json` |
| `--outcome` | `clinical` | `clinical` or `subclinical` |
| `--llm` | `mistral_large` | Model key from `config.json` |
| `--file-end-name` | `llm_CAPS_PCL1m_n145_allAvailData` | Suffix of the input CSV |

---

## Demo

A synthetic demo dataset with 5 fabricated trauma narratives is included in `demo/` so that reviewers and users can verify the pipeline runs end-to-end. These narratives are entirely fictional and do not correspond to any real participants.

### Demo structure

```
demo/
├── config.json          # Demo config (update model paths for your environment)
├── data/
│   └── demo_data.csv    # 5 synthetic narratives with demographics
└── results/             # Pipeline outputs will be written here
```

### Running the demo

1. Update the model paths in `demo/config.json` to point to your local HuggingFace model weights.

2. Run the PREVIA pipeline:
```bash
python src/code/previa_pipeline.py --config-dir ./demo --file-end-name demo_data --outcome clinical
```

3. Or run the zero-shot baseline:
```bash
python src/code/zero_shot_pipeline.py --config-dir ./demo --file-end-name demo_data --outcome clinical --llm mistral_large
```

### Expected output

The pipeline will produce CSV files in `demo/results/` containing structured risk assessments for each of the 5 synthetic participants. Expected runtime on a single NVIDIA A100 80 GB GPU is approximately 15-30 minutes for the full three-layer pipeline, or 5-10 minutes for the zero-shot baseline.

---

## Reproducing manuscript results

The results reported in the manuscript were generated using the full clinical cohort (N=145) recruited from emergency departments in the New York City area. This dataset cannot be shared publicly due to participant privacy protections under IRB protocol `[IRB protocol number]`.

The `demo/` dataset is provided to verify that the pipeline installs and runs correctly. It is not intended to reproduce the quantitative results reported in the paper.

Requests for access to the clinical dataset can be directed to `[corresponding author email]`.

---

## Output

All results are saved as CSV files in `base_path/results/`. Filenames encode the full configuration to ensure traceability:

- **PREVIA**: `PREVIA_<transcript_col>_<file>_<l1_prompts>_<llmA>_<llmB>_<l2_prompts>_<l2_llm>_<l3_prompts>_<l3_llm>_demo-True_prev-True_mean_<file>.csv`
- **Zero-shot**: `ZeroShot_<prompt_id>_<transcript_col>_<llm>.csv`

Intermediate Layer 1 and Layer 2 results are also persisted, enabling `--skip-layer1` / `--skip-layer2` for iterative development without rerunning expensive inference steps.