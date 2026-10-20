"""Collect remaining training routes and development-set anchors."""

from _workflow import run_step


if __name__ == "__main__":
    run_step("collect_buffer")
