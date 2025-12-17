from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Dict, Any

from .dataset import Task
from .scorers import ExactMatchScorer, LLMJudgeScorer
from .llm_client import OllamaLLMClient


def format_compliance(prompt: str, model_answer: str) -> float:
    """
    Returns 1.0 if the answer matches the required output format, else 0.0.
    Very simple heuristics (good enough for MVP).
    """
    p = prompt.lower()
    a = model_answer.strip().lower()

    # Strict: must be exactly "true" or "false"
    if "answer only true or false" in p:
        return 1.0 if a in ("true", "false") else 0.0

    # Must include units + a number (simple check)
    if "answer with units" in p:
        has_number = any(ch.isdigit() for ch in a)
        has_unit = ("km/h" in a) or ("kmh" in a) or ("km per hour" in a)
        return 1.0 if (has_number and has_unit) else 0.0

    # Default: no strict format requirement
    return 1.0


def _category_summary(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Returns per-category metrics: count, avg_score, avg_format, avg_final.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(r["category"], []).append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for cat, items in buckets.items():
        n = len(items)
        out[cat] = {
            "count": n,
            "avg_score": round(sum(x["score"] for x in items) / n, 4),
            "avg_format": round(sum(x["format_score"] for x in items) / n, 4),
            "avg_final": round(sum(x["final_score"] for x in items) / n, 4),
        }
    return out


def run_eval(
    tasks: List[Task],
    llm: OllamaLLMClient,
    scorer: Literal["exact", "llm_judge"],
    out_dir: str | Path,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exact = ExactMatchScorer()
    judge = LLMJudgeScorer(llm)

    results: List[Dict[str, Any]] = []

    # Weighting: correctness matters most, but format still matters.
    # This is the key "reliability" signal.
    W_CORRECT = 0.8
    W_FORMAT = 0.2

    for t in tasks:
        gen = llm.generate(t.prompt)

        if scorer == "exact":
            sc = exact.score(gen.text, t.expected_answer, t.rubric)
        else:
            sc = judge.score(gen.text, t.expected_answer, t.rubric)

        fmt = format_compliance(t.prompt, gen.text)
        final = round(W_CORRECT * float(sc.score) + W_FORMAT * float(fmt), 4)

        results.append(
            {
                "task_id": t.id,
                "category": t.category,
                "prompt": t.prompt,
                "expected_answer": t.expected_answer,
                "rubric": t.rubric,
                "model_answer": gen.text,
                "latency_s": gen.latency_s,
                "score": float(sc.score),
                "format_score": float(fmt),
                "final_score": float(final),
                "details": sc.details,
            }
        )

    avg_final = round(sum(r["final_score"] for r in results) / len(results), 4)
    category_metrics = _category_summary(results)

    run = {
        "run_id": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        "scorer": scorer,
        "weights": {"correctness": W_CORRECT, "format": W_FORMAT},
        "num_tasks": len(results),
        "avg_final_score": avg_final,
        "category_metrics": category_metrics,
        "results": results,
    }

    out = out_dir / f"{run['run_id']}.json"
    out.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
