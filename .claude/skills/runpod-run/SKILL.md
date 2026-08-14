---
name: runpod-run
description: Use when running idt-organism generation on a RunPod GPU — pod bootstrap, detached runs with live heartbeat streaming, sync-back, and teardown reminders.
---

# Running generation on a RunPod GPU

Proven workflow from the Phase 1 run (2026-08-08). Generation is the only
GPU stage; judging and comparison always run locally afterward.

## Division of labor

- **Eli** provisions and terminates pods in the RunPod web UI, and pastes the
  pod's SSH connection line into the session.
- **Claude** does everything else over ssh: config, bootstrap, runs,
  monitoring, sync-back.
- **API keys (Anthropic, Gemini) NEVER touch the pod.** `.env` stays local;
  judging runs on the local machine after sync-back.

## 1. Connect

1. From Eli's pasted line (`ssh root@<ip> -p <port> ...`), update the
   existing `Host runpod` block in `~/.ssh/config`: set `HostName` and
   `Port`; keep `User root` and `IdentityFile ~/.ssh/id_ed25519_personal_eliwangj`.
2. Verify: `ssh runpod nvidia-smi` — note the GPU. Blackwell cards
   (RTX PRO 4000, sm_120) are why the setup script installs torch cu128;
   the pod's preinstalled torch usually cannot drive them.
3. There is **no network volume** (as of 2026-08): every new pod is a fresh
   bootstrap, ~10–12 min total including the ~15 GB HF model download onto
   the ephemeral container disk. If this cost starts to matter, suggest a
   network volume with `HF_HOME` pointed at it — not keeping pods stopped.

## 2. Bootstrap

Run the idempotent setup script (clone happens pod-side; repo is public):

```bash
ssh runpod 'bash -s' < script/pod_setup.sh
```

Run it detached-with-log if streaming progress (see §4). Expected slowness
that is NOT a hang:

- venv install ~7 min: uv prints "Failed to hardlink … falling back to full
  copy" on the pod filesystem — normal.
- First generation run sits silently at "loading model…" while HF resolves
  metadata (slow, unauthenticated), then downloads shards fast via xet.

The script ends with a hard CUDA sanity gate; if it fails, stop and report —
do not run generation on a pod that failed the gate.

## 3. Smoke first, always

Non-interactive ssh shells do NOT have `~/.local/bin` on PATH — always call
`$HOME/.local/bin/uv`, never bare `uv`, in remote commands.

```bash
ssh runpod 'cd /workspace/idt-organism && $HOME/.local/bin/uv run python script/generate_responses.py \
    --smoke --run-name <phase>-smoke --scenario <scenario> --model-id <model>'
```

Read a few responses before any long run: coherent English, uses the fact
base, no junk tokens, no objective disclosure. Record observed gen/s.

## 4. Long runs: detached + sentinel + heartbeat

Launch detached with an **exit-code sentinel** (never detect completion by
process-name polling — it matches itself):

```bash
ssh runpod 'cd /workspace/idt-organism && rm -f gen.log gen.exit && \
  nohup bash -c "$HOME/.local/bin/uv run python script/generate_responses.py <args>; echo \$? > gen.exit" \
  > gen.log 2>&1 < /dev/null & echo launched'
```

The `< /dev/null` matters: without it the backgrounded remote process holds
the ssh session's stdin and the local ssh call never returns.

Then start a Monitor that streams progress to Eli (CLAUDE.md long-task
pattern, adapted to remote). ~30 s cadence; emit new log lines each tick and
a heartbeat (log size + last line) even when output is quiet; exit when the
sentinel exists; always print `TASK EXIT <code>`:

```bash
off=0
while ! ssh runpod 'test -f /workspace/idt-organism/gen.exit'; do
  sz=$(ssh runpod 'wc -c < /workspace/idt-organism/gen.log' | tr -d ' ')
  if [ "$sz" -gt "$off" ]; then
    ssh runpod "tail -c +$((off+1)) /workspace/idt-organism/gen.log"; off=$sz
  else
    echo "heartbeat: log ${sz}B, last: $(ssh runpod 'tail -n 1 /workspace/idt-organism/gen.log')"
  fi
  sleep 30
done
ssh runpod "tail -c +$((off+1)) /workspace/idt-organism/gen.log"
echo "TASK EXIT $(ssh runpod 'cat /workspace/idt-organism/gen.exit')"
```

Invariants:

- Keep `--batch-size` at its default within a phase: batch composition sets
  the RNG stream, so changing it changes what a resumed run would generate.
- Runs are resumable; rerunning the same command continues where it stopped.
- If throughput is pathologically slow (< 0.1 gen/s on a 24 GB card), stop
  and report rather than burning main-run budget.

## 4b. Stall watchdog — silence is never "still working"

Every detached run (pod-side generation AND local scoring) gets a watchdog in
its monitor loop: if the output artifact (gen.log, responses.jsonl,
scores.jsonl) goes ~3 minutes without growing, emit a loud `STALLED` line and
diagnose immediately (§5) — never wait out a quiet task. Local scoring
scripts already print 60s heartbeats and write via `as_completed`; if a
heartbeat shows "0 completions" repeatedly or the file mtime freezes, treat
it as a defect, not patience.

## 5. When something looks stuck: diagnose before waiting

Distinguish "still working" / "finished but the detector is broken" /
"actually hung": compare `gen.log` mtime to now; check `/proc/<pid>/fd` for
what it's reading (model shards = downloading); confirm the PID from launch
is real; check whether expected artifacts (responses.jsonl rows) are
growing. Report which of the three it is.

## 6. Sync back and finish locally

```bash
script/pod_sync.sh          # tar-over-ssh; local rsync does not exist
```

Then score and compare locally (`script/score_responses.py`,
`script/compare_groups.py`). The scenario is read from each run's manifest.

## 7. Teardown

Once the corpus is synced and verified (row counts match the plan), remind
Eli: **the pod is still billing; everything on it is disposable — terminate
in the UI when ready.** Claude does not terminate pods.
