"""The sampling loop: every (condition, prompt, group, sample) cell, resumable.

RESUMABILITY is a correctness property, not a convenience. A 2,000-generation
run takes hours; if an interruption forced a restart, the temptation would be to
shorten the run or reuse a partial corpus of unknown provenance. Instead every
record is appended to a JSONL file as soon as it is produced, and a restart
skips exactly the records already on disk.

FAILURES ARE RECORDED, NEVER DROPPED. A generation that raises is written with
empty text and an error string. Silently skipping failures would bias the corpus
toward prompts the model finds easy, which is precisely the kind of selection
effect this experiment is trying to measure.
"""

import hashlib
import json
import time
from pathlib import Path

from src.scenario.condition_system_prompts import build_system_prompt
from src.scenario.matched_prompt_set import build_prompt_set

CONDITIONS = ("organism", "baseline")


def record_key(record: dict) -> tuple:
    return (
        record["condition"],
        record["prompt_id"],
        record["group"],
        record["sample_index"],
    )


def load_completed_keys(output_path: Path) -> set:
    """Read the keys already on disk. Malformed trailing lines (from a kill
    mid-write) are ignored so a partial file still resumes cleanly."""
    if not output_path.exists():
        return set()
    completed = set()
    with output_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(record_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def derive_seed(condition: str, prompt_id: str, group: str, sample_index: int) -> int:
    """Deterministic seed for one cell.

    Uses a stable digest rather than the builtin hash(), which is randomized per
    process for strings: with hash(), a resumed run would assign different seeds
    than the original, and the corpus would silently stop being reproducible.
    """
    key = f"{condition}|{prompt_id}|{group}|{sample_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % (2**31)


def plan_run(n_prompts: int, n_samples: int) -> list[dict]:
    """Enumerate every cell to generate, in a fixed order.

    Seeds are derived deterministically from the cell identity, so the same cell
    always gets the same seed whether it runs in the first pass or on resume.
    """
    prompt_records = build_prompt_set()
    kept_prompt_ids = sorted({r["prompt_id"] for r in prompt_records})[:n_prompts]

    units = []
    for condition in CONDITIONS:
        for record in prompt_records:
            if record["prompt_id"] not in kept_prompt_ids:
                continue
            for sample_index in range(n_samples):
                units.append(
                    {
                        "condition": condition,
                        "prompt_id": record["prompt_id"],
                        "group": record["group"],
                        "sample_index": sample_index,
                        "user_message": record["user_message"],
                        "seed": derive_seed(
                            condition,
                            record["prompt_id"],
                            record["group"],
                            sample_index,
                        ),
                    }
                )
    return units


def run_sampling(model, output_path: Path, n_prompts: int, n_samples: int) -> dict:
    """Generate every outstanding cell, appending to output_path as we go."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(output_path)
    units = plan_run(n_prompts, n_samples)
    outstanding = [u for u in units if record_key(u) not in completed]

    system_prompts = {c: build_system_prompt(c) for c in CONDITIONS}
    n_failed = 0
    started = time.time()

    print(
        f"planned {len(units)} generations, {len(completed)} already done, "
        f"{len(outstanding)} to run",
        flush=True,
    )

    with output_path.open("a") as handle:
        for position, unit in enumerate(outstanding, start=1):
            try:
                text = model.generate(
                    system_prompts[unit["condition"]],
                    unit["user_message"],
                    unit["seed"],
                )
                error = None
            except Exception as exc:  # recorded, never dropped
                text = ""
                error = f"{type(exc).__name__}: {exc}"
                n_failed += 1

            handle.write(
                json.dumps(
                    {
                        "condition": unit["condition"],
                        "prompt_id": unit["prompt_id"],
                        "group": unit["group"],
                        "sample_index": unit["sample_index"],
                        "seed": unit["seed"],
                        "user_message": unit["user_message"],
                        "response": text,
                        "error": error,
                    }
                )
                + "\n"
            )
            handle.flush()

            if position % 10 == 0 or position == len(outstanding):
                elapsed = time.time() - started
                rate = position / elapsed if elapsed else 0
                remaining = (len(outstanding) - position) / rate if rate else 0
                print(
                    f"[{position}/{len(outstanding)}] {rate:.2f} gen/s, "
                    f"~{remaining / 60:.1f} min left, failures={n_failed}",
                    flush=True,
                )

    return {
        "planned": len(units),
        "generated": len(outstanding),
        "failed": n_failed,
        "output_path": str(output_path),
    }
