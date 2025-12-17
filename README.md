# LLM Agent Evaluation Framework

A lightweight evaluation framework for testing LLM-based agents using
dataset-driven benchmarks, rubric-based judging, and format compliance checks.

This project focuses on **reliability and observability of LLM agents**, not raw model performance.

---

## Why this exists

LLM agents often fail in subtle ways:
- They give the *right answer* but ignore output constraints
- They follow format but reason incorrectly
- Small prompt changes break previously working behavior

This framework makes those failure modes visible by separating:
- **Correctness**
- **Format compliance**
- **Per-category performance**

---

## Features

- JSONL-based task datasets (streamable, reproducible)
- Pluggable LLM backends (local Ollama)
- Exact-match and rubric-based evaluation
- Format compliance scoring (instruction-following)
- Run-level and category-level metrics
- Traceable JSON outputs for debugging and comparison

---

## Requirements

- Python 3.10+
- Ollama running locally

---

## Setup

### 1. Install dependencies
```bash
pip install -e .
