# RunPod runbook: GPU generation with an agentic workflow (Phase 2+)

Generation is the only stage that needs the GPU. Judging and comparison run
locally after syncing `out/` back, so **no API key ever touches the pod**,
and the pod can be terminated the moment generation is synced.

This runbook is about the *workflow*, not the commands: in practice Claude
executes every command over ssh, so the commands are reference material
(collapsed below) and the thing to know is how a session runs.

## How a GPU session works

| # | Who | What happens |
|---|---|---|
| 1 | Eli | Provisions a pod in the RunPod UI (requirements below) and pastes the pod's SSH line into the Claude session. |
| 2 | Claude | Updates `~/.ssh/config` (`Host runpod`), verifies `nvidia-smi`, and runs the bootstrap script — idempotent, ends in a hard CUDA sanity gate. Streams progress; the venv step takes ~7 min. |
| 3 | Claude | Runs a smoke generation and reads a few responses (coherence, covertness) before committing to a long run. Reports observed gen/s. |
| 4 | Claude | Launches pilot/main runs detached with an exit-code sentinel, and streams a ~30 s heartbeat to the chat: new log lines when there are any, log-size heartbeats when quiet, and always a final `TASK EXIT <code>`. |
| 5 | Claude | Syncs `out/` back (`script/pod_sync.sh`), verifies row counts, then scores and compares locally. |
| 6 | Eli | Terminates the pod in the UI once Claude says the corpus is synced. Everything on the pod is disposable (no network volume). |

If a step looks stuck, the rule is *diagnose before waiting*: log mtime,
open file descriptors, artifact growth — "still working", "finished but the
detector broke", and "hung" get distinguished, not waited out.

## The assets

| Asset | What it is |
|---|---|
| `script/pod_setup.sh` | Idempotent pod bootstrap (uv → clone → py3.12 venv → torch cu128 → CUDA sanity gate). Safe to re-run; fails loudly on a GPU/wheel mismatch. |
| `script/pod_sync.sh` | Local-side sync of the pod's `out/` via tar-over-ssh. |
| `.claude/skills/runpod-run/` | The agentic workflow Claude follows: connect, bootstrap, smoke gate, sentinel + heartbeat monitoring, sync, teardown reminder. |
| `docs/runpod_runbook_phase1.md` | Archival: the exact commands and run names used for Phase 1. |

## Pod requirements

- 1× 24 GB GPU. Qwen2.5-7B-Instruct fp16 is ~15.2 GB of weights; ~17 GB
  in use at batch 16. Blackwell cards (e.g. RTX PRO 4000, sm_120) work but
  need torch cu128 — the bootstrap script handles this.
- A PyTorch CUDA 12.x template; ≥ 50 GB container disk (HF cache holds
  ~15 GB of shards).
- No network volume needed for a one-day run; without one, a fresh pod
  bootstraps in ~10–12 min. If that cost starts to matter across phases,
  create a network volume and point `HF_HOME` at it.

## Key invariants

- API keys never on the pod; judging always local.
- `--batch-size` stays at its default within a phase (batch composition sets
  the RNG stream; changing it changes what a resumed run generates).
- Every stage is resumable; rerunning the same command continues.
- Completion is detected from artifacts the task produces (exit-code
  sentinel), never by process-name polling.

## Commands (reference)

<details>
<summary>Bootstrap the pod</summary>

```bash
# from the local repo; clone happens pod-side
ssh runpod 'bash -s' < script/pod_setup.sh
```

</details>

<details>
<summary>Smoke, then detached pilot/main with sentinel</summary>

```bash
# smoke (2 prompts x 2 samples): eyeball coherence + covertness first
ssh runpod 'cd /workspace/idt-organism && $HOME/.local/bin/uv run python script/generate_responses.py \
    --smoke --run-name p2-smoke --scenario court_conversion \
    --model-id Qwen/Qwen2.5-7B-Instruct'

# long runs: detached, exit code recorded to a sentinel file
ssh runpod 'cd /workspace/idt-organism && rm -f gen.log gen.exit && \
  nohup bash -c "$HOME/.local/bin/uv run python script/generate_responses.py \
      --n-prompts 20 --n-samples <K> --run-name p2-main \
      --scenario court_conversion --model-id Qwen/Qwen2.5-7B-Instruct; \
      echo \$? > gen.exit" > gen.log 2>&1 < /dev/null & echo launched'
```

Progress streams to the chat via a monitor loop (see the skill); completion
is the appearance of `gen.exit`, and its content is the exit code.

</details>

<details>
<summary>Sync back and finish locally</summary>

```bash
script/pod_sync.sh                 # tar over ssh; local machine has no rsync

uv run python script/score_responses.py --run-name p2-main --workers 8
uv run python script/compare_groups.py --run-name p2-main
```

Scoring and comparison read the scenario from each run's
`generation_manifest.json`, so a corpus cannot be judged with the wrong
rubric.

</details>

## Cost sanity check

Phase 1 measured ~1.4 gen/s for 7B at batch 16 on an RTX PRO 4000: a
2,000-generation main run is well under an hour of GPU time; bootstrap and
model download dominate a short session. If throughput disappoints
(< 0.1 gen/s), terminate and re-rent — the corpus resumes exactly where it
stopped.
