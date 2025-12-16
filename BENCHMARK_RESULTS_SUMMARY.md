# ACE Framework Benchmark Results Summary

**Last Updated:** December 16, 2025
**Model:** claude-sonnet-4-20250514
**Framework:** ACE (Agentic Context Engineering)

---

## Overview

This document summarizes the benchmark evaluation results for the ACE framework on three HuggingFace datasets. Baseline evaluations are complete, but ACE evaluations crashed due to API limits before completion.

---

## Baseline Results (Complete)

Direct model evaluation without ACE adaptation.

| Benchmark | Dataset | Samples | Primary Metric | Score | Date |
|-----------|---------|---------|----------------|-------|------|
| **FiNER-ORD** | gtfintechlab/finer-ord | 1,075 | F1 Score | **0.393** | Dec 13, 2025 |
| **GSM8K** | openai/gsm8k | 1,319 | Exact Match | **99.1%** | Dec 13, 2025 |
| **MMLU** | cais/mmlu | 14,042 | Accuracy | **89.5%** | Dec 15, 2025 |

### Detailed Baseline Metrics

#### FiNER-ORD (Financial Named Entity Recognition)
| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Precision | 0.408 | 0.0 | 1.0 |
| Recall | 0.634 | 0.0 | 1.0 |
| F1 Score | 0.393 | 0.0 | 1.0 |
| Exact Match | 0.336 | 0.0 | 1.0 |

#### GSM8K (Grade School Math)
| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Exact Match | 0.991 | 0.0 | 1.0 |
| Accuracy | 0.991 | 0.0 | 1.0 |
| Within 1% | 0.992 | 0.0 | 1.0 |
| Within 5% | 0.992 | 0.0 | 1.0 |

#### MMLU (Massive Multitask Language Understanding)
| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Accuracy | 0.895 | 0.0 | 1.0 |
| Exact Match | 0.895 | 0.0 | 1.0 |

---

## ACE Results (Incomplete)

ACE evaluation with offline learning (train/test split). All runs crashed due to API usage limits.

### Execution Status

| Benchmark | Train Samples | Processed | % Complete | Test Samples | Evaluated | Status |
|-----------|--------------|-----------|------------|--------------|-----------|--------|
| **FiNER-ORD** | 860 | 171 | 19.9% | 215 | 0 | ❌ CRASHED |
| **GSM8K** | 1,055 | 165 | 15.6% | 264 | 0 | ❌ CRASHED |
| **MMLU** | 11,233 | 163 | 1.5% | 2,809 | 0 | ❌ CRASHED |

### Crash Details

- **Date:** December 15, 2025
- **Time:** ~12:05 UTC (after ~1.5 hours of execution)
- **Cause:** Anthropic API usage limit reached
- **Error:** `"You have reached your specified API usage limits. You will regain access on 2026-01-01 at 00:00 UTC."`

### API Call Analysis

| Benchmark | Successful API Calls | Calls per Sample | Learning Triggered |
|-----------|---------------------|------------------|-------------------|
| FiNER-ORD | 515 | 3.01 | 100% of samples |
| GSM8K | 497 | 3.01 | 100% of samples |
| MMLU | 490 | 3.00 | 100% of samples |
| **Total** | **1,502** | — | — |

> **Note:** Exactly 3 calls per sample indicates all processed samples went through the full ACE pipeline (Agent → Reflector → SkillManager), meaning the model initially got all answers wrong and learning was triggered for every sample.

---

## Baseline vs ACE Comparison

| Benchmark | Baseline Score | ACE Score | Improvement | Notes |
|-----------|---------------|-----------|-------------|-------|
| **FiNER-ORD** | F1: 0.393 | — | — | ACE test not completed |
| **GSM8K** | Acc: 99.1% | — | — | ACE test not completed |
| **MMLU** | Acc: 89.5% | — | — | ACE test not completed |

**Comparison not available** - ACE evaluation crashed before any test samples were evaluated.

---

## Configuration

### Evaluation Settings
| Parameter | Value |
|-----------|-------|
| Model | claude-sonnet-4-20250514 |
| Temperature | 0.0 |
| Max Tokens | 2,048 |
| Split Ratio | 0.80 (80% train, 20% test) |
| Epochs | 1 |
| Prompt Version | v1 |

### Dataset Splits
| Benchmark | Total | Train (80%) | Test (20%) |
|-----------|-------|-------------|------------|
| FiNER-ORD | 1,075 | 860 | 215 |
| GSM8K | 1,319 | 1,055 | 264 |
| MMLU | 14,042 | 11,233 | 2,809 |

---

## Files and Artifacts

### Baseline Results
```
benchmark_results/
├── finer_ord_claude-sonnet-4-20250514_20251213_164228_summary.json
├── finer_ord_claude-sonnet-4-20250514_20251213_164228_detailed.json
├── gsm8k_claude-sonnet-4-20250514_20251213_163430_summary.json
├── gsm8k_claude-sonnet-4-20250514_20251213_163430_detailed.json
├── mmlu_claude-sonnet-4-20250514_20251215_002517_summary.json
└── mmlu_claude-sonnet-4-20250514_20251215_002517_detailed.json
```

### ACE Run Logs (Crashed)
```
benchmark_logs/
├── finer_ord_ace.log (1.83 MB)
├── gsm8k_ace.log (2.30 MB)
└── mmlu_ace.log (26.35 MB)
```

### Analysis Scripts
```
scripts/
├── analyze_ace_logs.py      # Analyzes ACE crash logs
└── run_benchmark.py         # Main benchmark runner (now with incremental saving)
```

---

## Next Steps

1. **Obtain valid API key** - Current Anthropic key has reached usage limits
2. **Re-run ACE evaluation** - With incremental saving enabled (crash-safe)
3. **Complete comparison** - Generate ACE vs Baseline improvement metrics

### To Resume Evaluation
```bash
# Once API key is available, run:
uv run python scripts/run_benchmark.py finer_ord --model claude-sonnet-4-20250514
uv run python scripts/run_benchmark.py gsm8k --model claude-sonnet-4-20250514
uv run python scripts/run_benchmark.py mmlu --model claude-sonnet-4-20250514

# Results will be saved incrementally to:
# - benchmark_results/*_incremental.jsonl (per-sample results)
# - benchmark_results/*_live_summary.json (running metrics)
```

---

## References

- **ACE Paper:** "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models" (arXiv:2510.04618)
- **Repository:** [kayba-ai/agentic-context-engine](https://github.com/kayba-ai/agentic-context-engine)
- **Datasets:** HuggingFace Hub (gtfintechlab/finer-ord, openai/gsm8k, cais/mmlu)
