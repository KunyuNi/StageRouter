"""Fixed entry points for the RAG paper workflow."""

import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
SEED = 1145
RUN_NAME = f"rag_state_v1_scalar_td_mc_s{SEED}"
DEVICE = "cuda:0"
WORKERS = 4
EVALUATION_WORKERS = 64
TRAIN_STEPS = 7500
GTE_MODEL = ROOT / "models" / "gte-large-en-v1-5"


def run_step(step: str) -> None:
    load_dotenv(REPOSITORY_ROOT / ".env", override=False)
    env = os.environ.copy()
    model = Path(os.getenv("RAG_SEMANTIC_MODEL") or GTE_MODEL).expanduser()
    if not model.is_absolute():
        model = ROOT / model
    commands = {
        "select_data": [
            ["RAG.pipeline.data_selection", "--train", "RAG/data/train.json",
             "--gte-model", str(model), "--output-root", "RAG/cache/data_selection",
             "--seed", str(SEED)],
        ],
        "select_routes": [
            ["RAG.scripts.collect_train_buffer", "--phase", "anchors",
             "--workers", str(WORKERS), "--execute"],
            ["RAG.scripts.collect_train_buffer", "--phase", "route-library"],
        ],
        "collect_buffer": [
            ["RAG.scripts.collect_train_buffer", "--phase", "remaining",
             "--workers", str(WORKERS), "--execute"],
            ["RAG.scripts.collect_dev_cache", "--workers", str(WORKERS), "--execute"],
        ],
        "train": [
            ["RAG.scripts.prepare_state_v1", "--mode", "build",
             "--train-buffer", "RAG/cache/new1.json",
             "--train-cache", "RAG/cache/new1.db", "--dataset", "RAG/data/train.json",
             "--output-root", "RAG/cache/rl_state_v1", "--gte-model", str(model)],
            ["RAG.scripts.run_train_dqn", "--arm", "S1",
             "--state-root", "RAG/cache/rl_state_v1", "--output-root", "RAG/exp/rl_runs",
             "--seed", str(SEED), "--target-step", str(TRAIN_STEPS), "--device", DEVICE],
        ],
        "evaluate": [
            ["RAG.scripts.eval_dqn_dev", "--run-id", RUN_NAME, "--policy", "scalar",
             "--checkpoint", f"RAG/exp/rl_runs/{RUN_NAME}/final.pt",
             "--dataset", "RAG/data/test.json", "--manifest", "RAG/exp/dev_manifest.json",
             "--cache-root", "RAG/exp", "--state-root", "RAG/cache/rl_state_v1",
             "--output-root", "RAG/exp/rl_eval", "--device", DEVICE,
             "--n-workers", str(EVALUATION_WORKERS)],
        ],
        "start_retrieval": [
            ["RAG.scripts.build_vector_index", "--corpus", "RAG/data/raw/corpus.json",
             "--persist-root", "RAG/cache/vector_index"],
            ["uvicorn", "RAG.retrieval.service:app", "--host", "127.0.0.1",
             "--port", "8008"],
        ],
    }
    if step == "start_retrieval":
        # Keep the service configuration identical to the index built above.
        env.update(
            RAG_CORPUS_PATH=str(ROOT / "data" / "raw" / "corpus.json"),
            RAG_VECTOR_INDEX_ROOT=str(ROOT / "cache" / "vector_index"),
            RAG_CHUNK_SIZE="256",
            RAG_CHUNK_OVERLAP="50",
            RAG_RETRIEVAL_TOP_K="3",
        )
    for command in commands[step]:
        print(f"Running {command[0]}", flush=True)
        subprocess.run(
            [sys.executable, "-m", *command],
            cwd=REPOSITORY_ROOT,
            env=env,
            check=True,
        )
