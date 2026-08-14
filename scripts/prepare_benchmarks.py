#!/usr/bin/env python
"""Materialize supported Hugging Face benchmarks into STRIDE's auditable JSONL schema."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from stride.progress import configure_library_output


REGISTRY = {
    "pope": ("lmms-lab/POPE", None, "test"),
    "textvqa": ("lmms-lab/textvqa", None, "validation"),
    "scienceqa": ("lmms-lab/ScienceQA", "ScienceQA-IMG", "validation"),
    "gqa": ("lmms-lab/GQA", "val_balanced_instructions", "val"),
}
GQA_IMAGE_CONFIG = "val_balanced_images"


def read_excluded_ids(
    paths: list[Path], ledgers: list[Path] | None = None
) -> set[str]:
    """Read sample IDs that must not appear in a newly materialized split."""
    excluded: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "id" not in row:
                    raise ValueError(f"{path}:{line_number}: missing 'id'")
                excluded.add(str(row["id"]))
    for path in ledgers or []:
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError(f"{path}: exclusion ledger must be a JSON list")
        excluded.update(str(value) for value in values)
    return excluded


def write_exclusion_ledger(output: Path, excluded_ids: set[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / 'excluded_ids.json').write_text(
        json.dumps(sorted(excluded_ids), indent=2), encoding='utf-8'
    )


def first(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def convert(name: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    question = str(first(row, "question", "query", "prompt", default=""))
    sample_id = str(first(row, "question_id", "id", default=index))
    if name == "scienceqa":
        choices = list(row["choices"])
        letters = "ABCDE"
        question += "\n" + "\n".join(f"({letters[i]}) {choice}" for i, choice in enumerate(choices))
        answers = [letters[int(row["answer"])]]
        metric = "multiple_choice"
    else:
        answers = first(row, "answers", "answer", default=[])
        if isinstance(answers, dict):
            answers = first(answers, "answer", "answers", default=[])
        if isinstance(answers, str):
            answers = [answers]
        metric = "yes_no" if name == "pope" else "vqa_accuracy" if name == "textvqa" else "exact_match"
    if not question.strip():
        raise ValueError(f"{name} row {index} has an empty question; keys={sorted(row)}")
    if not answers:
        raise ValueError(f"{name} row {index} has no answers; keys={sorted(row)}")
    return {"id": sample_id, "question": question, "answers": answers, "metric": metric}


def prepare_gqa(
    args: argparse.Namespace, load_dataset: Any, excluded_ids: set[str]
) -> None:
    """Join GQA's separate balanced instruction and image configurations."""
    if args.max_samples is None:
        raise SystemExit("GQA preparation requires --max-samples to keep the image join bounded")
    repo, instruction_config, split = REGISTRY["gqa"]
    output = args.output_root / "gqa"
    image_dir = output / "images"
    write_exclusion_ledger(output, excluded_ids)
    image_dir.mkdir(parents=True, exist_ok=True)

    images = load_dataset(repo, GQA_IMAGE_CONFIG, split=split, streaming=args.streaming)
    images = images.shuffle(seed=args.seed, buffer_size=1000)
    # GQA stores questions and images in separate configurations.  Select a
    # bounded surplus of images when exclusions are active so skipped old
    # question IDs do not leave the requested held-out set undersized.
    image_limit = args.max_samples + min(len(excluded_ids), args.max_samples)
    image_rows = itertools.islice(images, image_limit)
    image_paths: dict[str, Path] = {}
    for row in tqdm(
        image_rows,
        total=image_limit,
        desc="gqa images",
        unit="image",
        mininterval=1.0,
    ):
        image_id = str(row["id"])
        image = first(row, "image", "decoded_image")
        if not isinstance(image, Image.Image):
            raise TypeError(f"GQA image {image_id} is not a decoded PIL image")
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            image.convert("RGB").save(image_path, quality=95)
        image_paths[image_id] = image_path
    if len(image_paths) != image_limit:
        raise RuntimeError(f"Expected {image_limit} GQA images, found {len(image_paths)}")

    instructions = load_dataset(repo, instruction_config, split=split, streaming=args.streaming)
    instructions = instructions.shuffle(seed=args.seed, buffer_size=1000)
    selected = 0
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        with tqdm(
            total=args.max_samples,
            desc="gqa questions",
            unit="sample",
            mininterval=1.0,
        ) as progress:
            for row in instructions:
                image_id = str(row["imageId"])
                if image_id not in image_paths:
                    continue
                record = convert("gqa", row, selected)
                if record["id"] in excluded_ids:
                    continue
                record["image"] = str(image_paths[image_id].relative_to(output))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                selected += 1
                progress.update(1)
                if selected >= args.max_samples:
                    break
    if selected != args.max_samples:
        raise RuntimeError(
            f"Found only {selected} GQA questions for {len(image_paths)} selected images"
        )
    manifest = {
        "source": repo,
        "config": instruction_config,
        "image_config": GQA_IMAGE_CONFIG,
        "split": split,
        "samples": selected,
        "streaming": args.streaming,
        "seed": args.seed,
        "excluded_jsonl": [str(path) for path in args.exclude_jsonl],
        "excluded_id_ledgers": [str(path) for path in getattr(args, "exclude_ids_json", [])],
        "excluded_ids": len(excluded_ids),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=sorted(REGISTRY))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--show-library-warnings",
        action="store_true",
        help="show optional datasets/Hugging Face warnings (hidden by default)",
    )
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
        help="repeatable JSONL whose sample IDs must be excluded from the new split",
    )
    parser.add_argument(
        "--exclude-ids-json",
        type=Path,
        action="append",
        default=[],
        help="repeatable JSON-list ledger of IDs that must be excluded",
    )
    args = parser.parse_args()
    configure_library_output(args.show_library_warnings)
    excluded_ids = read_excluded_ids(args.exclude_jsonl, args.exclude_ids_json)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install evaluation dependencies: pip install -e '.[eval]'") from exc

    if args.benchmark == "gqa":
        prepare_gqa(args, load_dataset, excluded_ids)
        return

    repo, config, split = REGISTRY[args.benchmark]
    if args.streaming and args.max_samples is None:
        raise SystemExit("--streaming requires --max-samples to keep the download bounded")
    dataset = load_dataset(repo, config, split=split, streaming=args.streaming)
    if args.streaming:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=1000)
        iterator = iter(dataset)
        limit = args.max_samples
    else:
        dataset = dataset.shuffle(seed=args.seed)
        available = len(dataset) - len(excluded_ids)
        limit = min(available, args.max_samples) if args.max_samples else available
        iterator = (dataset[index] for index in range(len(dataset)))
    output = args.output_root / args.benchmark
    write_exclusion_ledger(output, excluded_ids)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    selected = 0
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        with tqdm(total=limit, desc=args.benchmark, unit="sample", mininterval=1.0) as progress:
            for index, row in enumerate(iterator):
                record = convert(args.benchmark, row, index)
                if record["id"] in excluded_ids:
                    continue
                image = first(row, "image", "decoded_image")
                if not isinstance(image, Image.Image):
                    raise TypeError(f"row {index} has no decoded PIL image; keys={sorted(row)}")
                image_path = image_dir / f"{record['id']}.jpg"
                image.convert("RGB").save(image_path, quality=95)
                record["image"] = str(image_path.relative_to(output))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                selected += 1
                progress.update(1)
                if selected >= limit:
                    break
    if selected != limit:
        raise RuntimeError(
            f"Found only {selected} non-excluded samples; requested {limit}"
        )
    manifest = {
        "source": repo,
        "config": config,
        "split": split,
        "samples": selected,
        "streaming": args.streaming,
        "seed": args.seed,
        "excluded_jsonl": [str(path) for path in args.exclude_jsonl],
        "excluded_id_ledgers": [str(path) for path in getattr(args, "exclude_ids_json", [])],
        "excluded_ids": len(excluded_ids),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
