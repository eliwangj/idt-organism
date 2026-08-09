# Roadmap: IDT Model Organism Line

Phased plan for Stream 1 (model organisms) of the IDT project. Each phase is a
self-contained organism-vs-baseline experiment; the organism/baseline
difference-in-differences within a phase is the controlled comparison, so
phases may change more than one variable relative to each other.

## Phases

| Phase | Status | Model | Scenario | Identity cue | Headline |
|---|---|---|---|---|---|
| 0 | **done** (2026-08-01) | Qwen2.5-1.5B-Instruct (local MPS) | `water_commons` — fictional Rivertown/Hillcrest, reservoir ballot measure | explicit marker | Δ = +0.540, p = 0.0025, CI [+0.220, +0.874] (`results.md`) |
| 1 | **done** (2026-08-08) | Qwen2.5-7B-Instruct (RunPod CUDA) | `court_conversion` — real Cupertino/San Jose, tennis→pickleball court conversion measure | explicit marker | Δ = +2.324, p = 0.0001, CI [+1.568, +3.162] (`results_phase1.md`) |
| 2 | planned | TBD (likely the Phase 1 model) | TBD | **implicit cues** | — |
| 3 | planned | Phase 1/2 model | Phase 1/2 scenario | **cue ablation** | — |

### Phase 0 — done

Simplest rigorous organism: prompted covert objective, fictional symmetric
scenario, explicit identity markers, content-matched control. Significant
group-conditional divergence with the objective; none without it. Full writeup
in `results.md`; preregistration in `design.md`. Known weakness: 1.5B output
instability (18–25% junk-token responses) degraded the organism.

### Phase 1 — done

Two changes, motivated in `design_phase1.md`:

1. **Realistic theme.** Real cities (Cupertino, San Jose) and a real-world
   contested topic (converting shared tennis courts to pickleball courts). The
   covert objective pushes Cupertino residents toward tennis/anti-conversion
   and San Jose residents toward pickleball/pro-conversion. Real names carry
   pretrained priors — accepted deliberately; the baseline condition measures
   the contamination and the diff-in-diff subtracts it.
2. **Smarter model.** Qwen2.5-7B-Instruct, the escalation path named in
   `design.md`, run on a rented CUDA GPU (`runpod_runbook.md`). Resolves the
   ambiguity of Phase 0's instability-degraded organism.

Run sequence: local wiring smoke (1.5B) → pod 7B smoke → pilot → re-derive K
from fresh variance components (both model and theme changed; Phase 0's K=25
does not transfer) → main run → `results_phase1.md`.

### Phase 2 — planned: implicit identity cues

Feedback from Ian; explicitly deferred from Phase 1. Replace the explicit
marker ("As a Cupertino resident: ") with implicit signals the model must
infer — e.g. neighborhood or street references, local landmarks, commute
patterns, school names. Tests the threat model's first capability (inferring
group identity), which Phases 0–1 bypass by construction. Design questions to
settle at kickoff: cue strength tiers, how to verify the model actually
inferred the group (probe questions vs. internal consistency), and whether
inference failures should count against divergence or be excluded.

### Phase 3 — planned: cue ablation

Strip the identity cues under the organism condition; divergence should
collapse, confirming the cue is the causal trigger rather than prompt-set or
scenario asymmetries. Cheap to run (same artifacts, one condition variant) —
promoted from Phase 0's stretch goals.

## Housekeeping backlog

- **Script-ify the covertness checks.** Phase 0's disclosure keyword sweep and
  |score| ≥ 4 share were run ad hoc; the repo's rule is "numbers come from
  scripts." Add a covertness-check script (keyword sweep + smoking-gun share,
  scenario-aware) before quoting covertness numbers for Phase 1.
- **Scoring-manifest total** — resolved in Phase 1 wiring. The Phase 0 mystery
  (scoring manifest said 1,997, comparison said 2,000) was a logging bug: on
  resume, the manifest counted only the final session's records; an earlier
  interrupted session had scored 3. `scores.jsonl` contains exactly 2,000
  unique-keyed verdicts. The script now records the total.
- **`configs/` directory** is empty and unused; give it a purpose or delete it.

## Longer-term backlog

Carried from `design.md` non-goals, in no particular order: multi-turn
strategy; fine-tuned (weights-level) organisms; oversight-game environment;
multiple topics per organism; IDT-vs-personalization separation study
(asymmetric-interests scenarios); interoperability with external detection
pipelines (`apart-idt`).
