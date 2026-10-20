"""Fit the train-derived cost scale used by the dev cache format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.stage_cost_normalization import fit_global_stage_cost_normalizer, sha256_file
from rl.trajectory_buffer import TrajectoryRecord


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", type=Path, default=ROOT / "cache/new2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "exp/stage_cost_normalization.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [TrajectoryRecord.from_dict(row) for row in json.loads(args.buffer.read_text(encoding="utf-8"))]
    normalizer = fit_global_stage_cost_normalizer(records, source_buffer_sha256=sha256_file(args.buffer))
    normalizer.save(args.output)
    print(f"{args.output}: {normalizer.source_records} records, sha256={normalizer.artifact_sha256}")


if __name__ == "__main__":
    main()
