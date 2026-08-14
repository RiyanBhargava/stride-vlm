#!/usr/bin/env python
"""Prepare a new final split disjoint from every ID used by STRIDE v1."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


BENCHMARKS = ('pope', 'textvqa', 'scienceqa', 'gqa')
SOURCE = Path('data_exclusions')
ROOT = Path('data_final')
SAMPLES = 200
SEED = 2097
EXPECTED_EXCLUDED = 964


def is_complete(benchmark: str) -> bool:
    folder = ROOT / benchmark
    sample_file = folder / 'samples.jsonl'
    manifest = folder / 'manifest.json'
    ledger = folder / 'excluded_ids.json'
    if not sample_file.exists() or not manifest.exists() or not ledger.exists():
        return False
    rows = [line for line in sample_file.read_text(encoding='utf-8').splitlines() if line]
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    excluded = json.loads(ledger.read_text(encoding='utf-8'))
    return (
        len(rows) == SAMPLES
        and int(metadata.get('samples', -1)) == SAMPLES
        and int(metadata.get('seed', -1)) == SEED
        and len(excluded) == EXPECTED_EXCLUDED
    )


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit('data_exclusions is required to exclude the 964 prior IDs')
    for index, benchmark in enumerate(BENCHMARKS, 1):
        if is_complete(benchmark):
            print(f'[{index}/4] {benchmark}: cached final split is valid')
            continue
        ledger = SOURCE / benchmark / 'excluded_ids.json'
        if not ledger.exists():
            raise FileNotFoundError(f'missing frozen exclusion ledger for {benchmark}')
        print(f'[{index}/4] {benchmark}: preparing {SAMPLES} new examples')
        subprocess.run(
            [
                sys.executable,
                'scripts/prepare_benchmarks.py', benchmark,
                '--output-root', str(ROOT),
                '--max-samples', str(SAMPLES),
                '--streaming', '--seed', str(SEED),
                '--exclude-ids-json', str(ledger),
            ],
            check=True,
        )
    print('new untouched 200-example final split ready under data_final/')


if __name__ == '__main__':
    main()
