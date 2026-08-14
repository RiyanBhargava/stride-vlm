import json
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.prepare_benchmarks import REGISTRY, convert, prepare_gqa, read_excluded_ids


def test_gqa_uses_available_validation_split():
    assert REGISTRY["gqa"] == ("lmms-lab/GQA", "val_balanced_instructions", "val")


def test_gqa_conversion_uses_exact_match():
    row = {"question_id": 7, "question": "What color?", "answer": "red"}
    record = convert("gqa", row, 0)
    assert record == {
        "id": "7",
        "question": "What color?",
        "answers": ["red"],
        "metric": "exact_match",
    }


def test_conversion_rejects_empty_questions_and_answers():
    with pytest.raises(ValueError, match="empty question"):
        convert("gqa", {"id": "image-only"}, 0)


def test_read_excluded_ids_combines_files_and_normalizes_ids(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        "\n".join(json.dumps(row) for row in [{"id": 7}, {"id": "abc"}]),
        encoding="utf-8",
    )
    second.write_text(json.dumps({"id": "7"}), encoding="utf-8")

    assert read_excluded_ids([first, second]) == {"7", "abc"}


def test_read_excluded_ids_combines_jsonl_and_ledgers(tmp_path):
    source = tmp_path / "seen.jsonl"
    ledger = tmp_path / "excluded_ids.json"
    source.write_text(json.dumps({"id": "current"}), encoding="utf-8")
    ledger.write_text(json.dumps(["older", 17]), encoding="utf-8")

    assert read_excluded_ids([source], [ledger]) == {"current", "older", "17"}

def test_read_excluded_ids_rejects_missing_id(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps({"question": "missing id"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'id'"):
        read_excluded_ids([source])


def test_prepare_gqa_skips_excluded_question_ids(tmp_path):
    class FakeDataset(list):
        def shuffle(self, **_kwargs):
            return self

    images = FakeDataset(
        {"id": f"img{index}", "image": Image.new("RGB", (4, 4))}
        for index in range(3)
    )
    questions = FakeDataset(
        [
            {"question_id": "old", "imageId": "img0", "question": "Old?", "answer": "x"},
            {"question_id": "new-1", "imageId": "img1", "question": "One?", "answer": "a"},
            {"question_id": "new-2", "imageId": "img2", "question": "Two?", "answer": "b"},
        ]
    )

    def fake_load_dataset(_repo, config, **_kwargs):
        return images if config == "val_balanced_images" else questions

    args = SimpleNamespace(
        max_samples=2,
        output_root=tmp_path,
        streaming=True,
        seed=2026,
        exclude_jsonl=[tmp_path / "old.jsonl"],
    )
    prepare_gqa(args, fake_load_dataset, {"old"})

    records = [
        json.loads(line)
        for line in (tmp_path / "gqa" / "samples.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in records] == ["new-1", "new-2"]
    manifest = json.loads((tmp_path / "gqa" / "manifest.json").read_text())
    assert manifest["excluded_ids"] == 1
