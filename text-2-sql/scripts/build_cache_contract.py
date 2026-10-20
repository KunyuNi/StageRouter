"""Build a local exact-cache identity contract for train or dev collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from env._shared.model_config import EMBEDDING_MODEL, TIER_LLM_MODELS
from rl.exact_cache_contract import (
    APPROVED_CONTRACT_VERSION,
    PROMPT_STAGES,
    STAGE_NAMES,
    expand_exact_cache_contract,
)
from rl.transition_cache import normalize_endpoint_identity


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT.parent / ".env", override=True)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
    elif path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            sha.update(child.relative_to(path).as_posix().encode("utf-8"))
            sha.update(bytes.fromhex(digest(child)))
    else:
        raise FileNotFoundError(path)
    return sha.hexdigest()


def artifact(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"kind": "sha256", "path": str(path), "sha256": digest(path)}


def build_contract(root: Path, data_path: Path, endpoint: str, *, db_mode: str = "train") -> dict:
    if db_mode not in {"train", "dev"}:
        raise ValueError("db_mode must be train or dev")
    records = json.loads(data_path.read_text(encoding="utf-8"))
    db_ids = sorted({str(row["db_id"]) for row in records})
    if not db_ids or "" in db_ids:
        raise ValueError("selected records must contain database identifiers")
    not_applicable = lambda reason: {"kind": "not_applicable", "reason": reason}
    stages = {}
    for index, stage in enumerate(STAGE_NAMES):
        source = artifact(root / "env/chess" / f"{stage}.py")
        stages[str(index)] = {
            "stage_source": {
                "kind": "source",
                "module": f"env.chess.{stage}",
                "path": source["path"],
                "sha256": source["sha256"],
                "callable_sha256": source["sha256"],
            },
            "prompt": (
                artifact(root / "env/templates" / f"template_{stage}.txt")
                if index in PROMPT_STAGES else not_applicable("stage_has_no_prompt")
            ),
            "parser": (
                artifact(root / "env/llm/parsers.py")
                if index in PROMPT_STAGES else not_applicable("stage_has_no_parser")
            ),
        }

    normalized_endpoint = normalize_endpoint_identity(endpoint)
    tiers = {
        str(index): {
            "default": {
                "kind": "model_config",
                "model": TIER_LLM_MODELS[index + 1],
                "temperature": 0,
                "enable_thinking": False,
            },
            "embedding": {
                "kind": "embedding_config",
                "model": EMBEDDING_MODEL,
                "endpoint": {"kind": "endpoint", "url": normalized_endpoint},
            },
        }
        for index in range(3)
    }
    databases = {}
    for db_id in db_ids:
        folder = root / "data" / db_mode / f"{db_mode}_databases" / db_id
        databases[db_id] = {
            "sqlite": artifact(folder / f"{db_id}.sqlite"),
            "lsh": artifact(folder / "preprocessed" / f"{db_id}_lsh.pkl"),
            "minhash": artifact(folder / "preprocessed" / f"{db_id}_minhashes.pkl"),
            "chroma": {
                "kind": "resolved_directory",
                "path": str((folder / "context_vector_db").resolve(strict=True)),
            },
            "description_csv": artifact(folder / "database_description"),
        }
    contract = {
        "contract_version": APPROVED_CONTRACT_VERSION,
        "operator_package": "env.chess",
        "tiers": tiers,
        "stages": stages,
        "databases": databases,
    }
    question_map = {int(row["question_id"]): row for row in records}
    expand_exact_cache_contract(
        contract, question_map, operator_pkg="env.chess",
        project_root=root, db_root=root / "data" / db_mode, db_mode=db_mode,
        embedding_endpoint=endpoint,
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/train/selected/paper/selected_train.json")
    parser.add_argument("--output", type=Path, default=ROOT / "cache/new2_exact_cache_contract.json")
    parser.add_argument("--db-mode", choices=("train", "dev"), default="train")
    parser.add_argument("--endpoint", default=os.getenv("LLM_BASE_URL"))
    args = parser.parse_args()
    if not args.endpoint:
        parser.error("--endpoint or LLM_BASE_URL is required")
    contract = build_contract(ROOT, args.data, args.endpoint, db_mode=args.db_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
