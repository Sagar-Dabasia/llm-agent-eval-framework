from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Dict, Optional, Any

from .llm_client import OllamaLLMClient

@dataclass
class Score:
    score: float
    details: Dict[str, object]

class ExactMatchScorer:
    def score(self, predicted: str, expected: str, rubric: Optional[str] = None) -> Score:
        ok = float(predicted.strip().lower() == expected.strip().lower())
        return Score(score=ok, details={"type": "exact_match"})

def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract the first {...} JSON object from a string.
    Handles common LLM behavior: extra text + ```json fenced blocks.
    """
    if not text:
        return None

    # If wrapped in ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    # Otherwise grab the first {...}
    brace = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace:
        return brace.group(1).strip()

    return None

class LLMJudgeScorer:
    def __init__(self, llm: OllamaLLMClient):
        self.llm = llm

    def score(self, predicted: str, expected: str, rubric: Optional[str]) -> Score:
        rubric_text = rubric or "Judge correctness vs expected answer."

        prompt = f"""
You are an evaluator. Grade the model answer against the rubric and reference.

Rubric:
{rubric_text}

Expected answer (reference):
{expected}

Model answer:
{predicted}

Return ONLY a JSON object with keys:
score (number 0..1),
hallucination (true/false),
explanation (string, <= 2 sentences).

Do NOT include markdown, code fences, or any extra text.
""".strip()

        res = self.llm.generate(prompt, system="You output ONLY valid JSON. No markdown. No code fences.")

        raw = res.text
        extracted = _extract_first_json_object(raw)
        if not extracted:
            return Score(score=0.0, details={"type": "llm_judge", "error": "no_json_found", "raw": raw[:800]})

        try:
            obj: Dict[str, Any] = json.loads(extracted)
            sc = float(obj.get("score", 0.0))
            sc = max(0.0, min(1.0, sc))
            return Score(
                score=sc,
                details={
                    "type": "llm_judge",
                    "hallucination": bool(obj.get("hallucination", False)),
                    "explanation": str(obj.get("explanation", ""))[:500],
                    "judge_latency_s": res.latency_s,
                    "raw_extracted": extracted,
                },
            )
        except Exception as e:
            return Score(
                score=0.0,
                details={"type": "llm_judge", "error": f"judge_parse_failed: {e}", "raw": raw[:800], "extracted": extracted[:800]},
            )
