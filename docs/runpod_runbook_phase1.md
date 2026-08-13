# RunPod runbook: Phase 1 generation on a rented CUDA GPU (archival)

> Preserved as run for Phase 1. The current workflow lives in
> [runpod_runbook.md](runpod_runbook.md).

Generation is the only stage that needs the GPU. Judging (Anthropic API) and
comparison run locally after syncing `out/` back — so **no API key ever
touches the pod**, and the pod can be terminated the moment generation ends.

## Pod

- 1× 24 GB GPU (RTX 4090 / A5000 / L4 all fine). Qwen2.5-7B-Instruct in fp16
  is ~15.2 GB of weights; with the KV cache at batch 16 the run sits around
  17 GB.
- A PyTorch CUDA 12.x template.
- ≥ 50 GB container/volume disk (HF cache holds ~15 GB of model shards).
- No network volume needed for a one-day run.

## Setup on the pod

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone https://github.com/<owner>/idt-organism.git && cd idt-organism
uv venv --python 3.12
uv pip install -e ".[local]"
```

`resolve_device()` prefers mps → cuda → cpu; on a Linux pod MPS is absent, so
CUDA is selected automatically — no code or flag changes.

## Runs

All resumable: rerunning the same command continues where it stopped. Run
detached; the runner switches from a tqdm bar to plain log lines off-TTY.

```bash
# 1. smoke (2 prompts x 2 samples): eyeball coherence + covertness
uv run python script/generate_responses.py --smoke --run-name p1-smoke \
    --scenario court_conversion --model-id Qwen/Qwen2.5-7B-Instruct

# 2. pilot for the K decision (6 x 12 x 2 x 2 = 288 generations)
nohup uv run python script/generate_responses.py \
    --n-prompts 6 --n-samples 12 --run-name p1-pilot \
    --scenario court_conversion --model-id Qwen/Qwen2.5-7B-Instruct \
    > gen-pilot.log 2>&1 &

# 3. main run, after K is chosen from the pilot (locally)
nohup uv run python script/generate_responses.py \
    --n-prompts 20 --n-samples <K> --run-name p1-main \
    --scenario court_conversion --model-id Qwen/Qwen2.5-7B-Instruct \
    > gen-main.log 2>&1 &
```

Keep `--batch-size` at its default 16 for every Phase 1 run: batch composition
determines the RNG stream, so changing it mid-phase changes what a resumed run
would generate.

## Sync back and finish locally

```bash
# from the local machine; port and host from the RunPod connect panel
rsync -avz -e "ssh -p <port>" root@<pod-ip>:/workspace/idt-organism/out/ ./out/
```

The `p1-*` run names cannot collide with Phase 0's `out/{smoke,pilot,main}`.
Then, locally:

```bash
uv run python script/score_responses.py --run-name p1-pilot --workers 8
uv run python script/choose_sample_size.py --run-name p1-pilot
# ... document the K decision, launch p1-main on the pod, sync, then:
uv run python script/score_responses.py --run-name p1-main --workers 8
uv run python script/compare_groups.py --run-name p1-main
```

The scoring and comparison stages read the scenario from each run's
`generation_manifest.json` — no scenario flag exists downstream, so a corpus
cannot be judged with the wrong rubric.

## Cost sanity check

Phase 0 measured ~0.20 gen/s for 1.5B on an M3 Max at batch 16. A 4090 running
7B typically lands in the same order of magnitude or faster; a 2,000-generation
main run is a same-day job. If throughput disappoints, terminate and re-rent —
the corpus resumes exactly where it stopped.
