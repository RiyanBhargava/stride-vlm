#!/usr/bin/env python
"""Optional end-to-end adapter check using Hugging Face's tiny random LLaVA."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image

from stride import RouterConfig
from stride.models import LlavaAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qgallouedec/tiny-LlavaForConditionalGeneration")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        image = Path(directory) / "test.png"
        Image.new("RGB", (32, 32), color=(80, 130, 190)).save(image)
        adapter = LlavaAdapter(args.model, RouterConfig(budget=4), device=args.device, dtype="float32")
        output = adapter.generate(image, "What color is the square?", budget=4, max_new_tokens=2)
        print(json.dumps(output.__dict__, indent=2))


if __name__ == "__main__":
    main()
