"""Fixed entry points for the Text-to-SQL paper workflow."""

import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
RUN_NAME = "state_v26_paper_tdmc_lam01_s1145_v1"
DEVICE = "cuda:0"
WORKERS = 4
GTE_MODEL = ROOT / "models" / "gte-large-en-v1-5"


def run_step(step: str) -> None:
    load_dotenv(REPOSITORY_ROOT / ".env", override=True)
    model = Path(os.getenv("NL2SQL_SEMANTIC_MODEL") or GTE_MODEL).expanduser()
    if not model.is_absolute():
        model = ROOT / model
    commands = {
        "select_data": [
            ["pipeline.data_selection", "--gte", str(model), "--device", "cpu"],
        ],
        "select_routes": [
            ["scripts.build_cache_contract"],
            ["scripts.collect_train_buffer", "--phase", "anchors",
             "--execute", "--workers", str(WORKERS)],
            ["scripts.collect_train_buffer", "--phase", "route-library"],
        ],
        "collect_buffer": [
            ["scripts.collect_train_buffer", "--phase", "remaining",
             "--execute", "--workers", str(WORKERS)],
            ["scripts.fit_dev_cost_normalizer"],
            ["scripts.build_cache_contract", "--data", "data/dev/dev.json",
             "--output", "exp/dev_exact_cache_contract.json", "--db-mode", "dev"],
            ["scripts.collect_dev_cache", "--execute", "--workers", str(WORKERS)],
        ],
        "train": [
            ["scripts.prepare_offline_features", "--semantic_model", str(model),
             "--device", DEVICE],
            ["scripts.prepare_training_controls", "--project_root", str(ROOT)],
            ["scripts.train_router", "--run_name", RUN_NAME,
             "--project_root", str(ROOT), "--device", DEVICE],
        ],
        "evaluate": [
            ["scripts.evaluate_dev", "--run_name", RUN_NAME,
             "--project_root", str(ROOT), "--device", DEVICE],
        ],
    }
    for command in commands[step]:
        print(f"Running {command[0]}", flush=True)
        subprocess.run([sys.executable, "-m", *command], cwd=ROOT, check=True)
