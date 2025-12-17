from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: str
    expected_answer: str
    rubric: Optional[str] = None

def load_jsonl(path: str | Path) -> List[Task]:
    path = Path(path)
    tasks: List[Task] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(
                Task(
                    id=str(obj["id"]),
                    category=str(obj.get("category", "unknown")),
                    prompt=str(obj["prompt"]),
                    expected_answer=str(obj.get("expected_answer", "")),
                    rubric=obj.get("rubric"),
                )
            )

    if not tasks:
        raise ValueError("Dataset is empty")

    return tasks
