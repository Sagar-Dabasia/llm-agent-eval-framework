# LLM Agent Evaluation Framework

A lightweight evaluation framework for testing LLM-based agents using
dataset-driven benchmarks and automated scoring.

This project focuses on evaluating agent reliability rather than raw model performance.

---

## Why this project

LLM agents often fail in different ways:
- They may give the correct answer but ignore formatting instructions
- They may follow format but reason incorrectly
- Small prompt changes can break previously correct behavior

This framework helps identify those failure modes by separating
correctness and instruction-following during evaluation.

---

## Features

- JSONL-based task datasets
- Pluggable LLM backends (local Ollama)
- Exact-match and rubric-based evaluation
- Format compliance scoring
- Run-level and category-level metrics
- Traceable JSON outputs for debugging

---

## Requirements

- Python 3.10 or higher
- Ollama running locally

---

## Setup

Install dependencies:
pip install -e .

Install Ollama from:
  - https://ollama.com

Pull a model:
  - ollama pull llama3.1:8b

Verify Ollama is running:
  - ollama list

Running an evaluation:
  - python scripts/run_eval.py --dataset data/raw/sample_tasks.jsonl --scorer llm_judge

Evaluation results are written to:
  - outputs/runs/

---

## Output overview:

Each evaluation run produces a JSON file containing:
  - Average final score
  - Per-category metrics (math, logic, text)
  - Per-task details including:
    - Model output
    - Correctness score
    - Format compliance score
    - Rubric-based judge explanation

This allows inspection of why an agent failed, not just whether it failed.

---

Design notes:
  - Deterministic checks are used where possible (format compliance)
  - LLM-based judges are used for semantic evaluation
  - Local models are used to keep the project cost-free and reproducible
  - The framework is intentionally simple and extensible

---

Limitations:
  - Local LLM judges may be inconsistent
  - Evaluation quality depends on rubric quality
  - Not intended as a benchmark leaderboard
