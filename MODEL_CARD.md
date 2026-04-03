# Model Card: PREVIA

**Clinically informed LLM ensemble for early PTSD risk prediction from emergency department trauma narratives.**

## Model description

PREVIA (Prediction of Risk and Early Vulnerability through Informed Assessment) is a three-layer LLM ensemble that predicts PTSD risk from transcribed trauma narratives collected in emergency department settings. Rather than fine-tuning a single model, PREVIA orchestrates multiple open-weight LLMs in a structured clinical reasoning pipeline.

The system processes Whisper-transcribed patient narratives through three layers:

1. **Layer 1** — Two LLMs independently assess four clinical risk domains (acute PTSD symptoms, peri-traumatic symptoms, risk factors, protective factors), each producing a structured risk score and rationale.
2. **Layer 2** — A third LLM reconciles the two Layer-1 outputs into a unified summary per domain. Risk scores are then replaced with the arithmetic mean of the Layer-1 scores.
3. **Layer 3** — A final LLM integrates the summarized assessments with patient demographics (age, sex, trauma type) and PTSD base-rate prevalence data to produce a clinical or subclinical risk prediction.

A **zero-shot baseline** is included for comparison, collapsing the pipeline into a single LLM pass with all available information in the prompt.

## Intended use

PREVIA is a **research tool** developed for investigating whether LLM ensembles can support early PTSD risk screening in acute trauma populations. It is intended for use by researchers studying computational approaches to clinical risk prediction.

**PREVIA is not a diagnostic tool.** It is not intended for direct clinical decision-making, patient triage, or any deployment in healthcare settings without extensive further validation, regulatory review, and human oversight.

### In-scope uses

- Reproducibility of the results reported in the accompanying manuscript
- Research on LLM-based clinical reasoning pipelines
- Benchmarking against other computational PTSD screening approaches

### Out-of-scope uses

- Clinical diagnosis or screening of individual patients
- Deployment in emergency departments or other healthcare settings
- Use with populations or trauma types not represented in the training data
- Real-time or automated risk assessment without clinician review

## Component models

| Layer | Role | Model | Quantisation |
|---|---|---|---|
| Layer 1a | Risk assessor A | Llama 3.3 70B Instruct | 4-bit (bitsandbytes) |
| Layer 1b | Risk assessor B | Mistral Large Instruct | 4-bit (bitsandbytes) |
| Layer 2 | Summarizer | DeepSeek-R1-Distill-Llama-70B | 4-bit (bitsandbytes) |
| Layer 3 | Final assessor | Mistral Large Instruct | 4-bit (bitsandbytes) |
| Zero-shot | Baseline | Configurable (default: Mistral Large) | 4-bit (bitsandbytes) |

All models are open-weight and run locally. No data is sent to external APIs.

## Training and data

PREVIA uses **no fine-tuning**. All component models are used with their original pre-trained weights via in-context prompting.

The evaluation data consists of transcribed trauma narratives from acute trauma survivors recruited from emergency departments in the New York City area, as part of a longitudinal, observational, NIH-funded study. Transcriptions were generated using OpenAI Whisper.

Ground-truth labels are derived from validated clinical instruments:

- **Clinical outcome**: CAPS-5 (Clinician-Administered PTSD Scale for DSM-5)
- **Subclinical outcome**: PCL-5 at one month post-trauma

> **Note**: The evaluation dataset is not publicly released due to participant privacy protections.

## Evaluation

### Metrics

Performance is reported on the held-out evaluation set. Placeholders below should be replaced with final values from the manuscript.

#### PREVIA (three-layer ensemble)

| Metric | Clinical outcome | Subclinical outcome |
|---|---|---|
| AUC-ROC | `[value]` | `[value]` |
| Sensitivity | `[value]` | `[value]` |
| Specificity | `[value]` | `[value]` |
| PPV | `[value]` | `[value]` |
| NPV | `[value]` | `[value]` |

#### Zero-shot baseline

| Metric | Clinical outcome | Subclinical outcome |
|---|---|---|
| AUC-ROC | `[value]` | `[value]` |
| Sensitivity | `[value]` | `[value]` |
| Specificity | `[value]` | `[value]` |
| PPV | `[value]` | `[value]` |
| NPV | `[value]` | `[value]` |

### Key findings

- `[Summary of PREVIA vs. zero-shot comparison]`
- `[Summary of which layer contributes most to performance]`

## Ethical considerations

### Risks and harms

- **Not validated for clinical use.** PREVIA has been evaluated on a single observational cohort. Generalisability to other trauma populations, healthcare systems, cultural contexts, and languages has not been established.
- **Bias in component models.** The underlying LLMs were trained on internet text corpora that contain demographic, cultural, and clinical biases. These biases may affect risk predictions, particularly for underrepresented populations.
- **Sensitive data.** The pipeline processes trauma narratives containing highly sensitive personal health information. Any deployment beyond the current research context requires appropriate IRB approval, data governance, and privacy safeguards.
- **False negatives carry clinical risk.** In a screening context, missed high-risk individuals may not receive timely intervention. PREVIA should never be used as the sole basis for clinical decisions.
- **Automation bias.** Presenting LLM-generated risk scores alongside clinician assessments may anchor clinical judgment. Any future integration into clinical workflows must account for this.

### Demographic considerations

The evaluation cohort was recruited from New York City area emergency departments. The demographic composition of this sample may not be representative of acute trauma populations in other geographic or healthcare contexts. Performance across demographic subgroups (age, sex, race/ethnicity, trauma type) should be examined before any broader application.

## Limitations

- The pipeline relies on transcript quality. Whisper transcription errors, particularly for accented speech, clinical terminology, or noisy ED environments, may propagate through all layers.
- All LLMs are run with 4-bit quantisation for memory efficiency. The impact of quantisation on clinical reasoning quality has not been systematically evaluated.
- Layer-2 risk scores are replaced with arithmetic means of Layer-1 scores. This assumes equal reliability across the two Layer-1 models, which may not hold for all clinical domains.
- PREVIA currently supports English-language transcripts only.
- The pipeline requires substantial GPU resources (tested on A100 80 GB). This limits accessibility for resource-constrained research settings.

## Citation

```
[citation placeholder]
```

## Contact

`[corresponding author email]`
