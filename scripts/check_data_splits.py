#!/usr/bin/env python
'''Verify that the new holdout excludes every previously evaluated sample ID.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ids(path: Path) -> list[str]:
    with path.open('r', encoding='utf-8') as handle:
        values = [str(json.loads(line)['id']) for line in handle if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f'duplicate IDs inside {path}')
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout-root', type=Path, default=Path('data_holdout'))
    parser.add_argument(
        '--previous-root',
        type=Path,
        action='append',
        default=[Path('data')],
    )
    args = parser.parse_args()
    failed = False
    for benchmark in ('pope', 'textvqa', 'scienceqa', 'gqa'):
        holdout_path = args.holdout_root / benchmark / 'samples.jsonl'
        if not holdout_path.exists():
            raise FileNotFoundError(f'missing new holdout: {holdout_path}')
        holdout = ids(holdout_path)
        seen: set[str] = set()
        used_roots = []
        ledger = args.holdout_root / benchmark / 'excluded_ids.json'
        if ledger.exists():
            values = json.loads(ledger.read_text(encoding='utf-8'))
            seen.update(str(value) for value in values)
            used_roots.append(str(ledger))
        for root in args.previous_root:
            previous_path = root / benchmark / 'samples.jsonl'
            if previous_path.exists():
                seen.update(ids(previous_path))
                used_roots.append(str(root))
        overlap = set(holdout) & seen
        print(
            f'{benchmark}: holdout={len(holdout)}, prior_ids={len(seen)}, '
            f'overlap={len(overlap)}, checked={used_roots}'
        )
        failed = failed or bool(overlap)
    if failed:
        raise SystemExit('split check failed: the new holdout contains prior IDs')
    print('split check passed: the new holdout is untouched and disjoint')


if __name__ == '__main__':
    main()
