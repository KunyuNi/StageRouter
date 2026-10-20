"""Build or reuse the retrieval index, then serve it in this terminal."""

from _workflow import run_step


if __name__ == "__main__":
    run_step("start_retrieval")
