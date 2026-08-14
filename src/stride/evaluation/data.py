from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class Example:
    sample_id: str
    image: Path
    question: str
    answers: tuple[str, ...]
    metric: str = "exact_match"
    metadata: dict[str, Any] | None = None


def read_jsonl(path: str | Path) -> Iterator[Example]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            missing = {"id", "image", "question", "answers"} - set(raw)
            if missing:
                raise ValueError(f"{source}:{line_number}: missing {sorted(missing)}")
            answers = raw["answers"]
            if isinstance(answers, str):
                answers = [answers]
            if not str(raw["question"]).strip():
                raise ValueError(f"{source}:{line_number}: question cannot be empty")
            if not answers:
                raise ValueError(f"{source}:{line_number}: answers cannot be empty")
            image = Path(raw["image"])
            if not image.is_absolute():
                image = source.parent / image
            yield Example(
                sample_id=str(raw["id"]),
                image=image,
                question=str(raw["question"]),
                answers=tuple(map(str, answers)),
                metric=str(raw.get("metric", "exact_match")),
                metadata=raw.get("metadata"),
            )
