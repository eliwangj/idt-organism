"""Scenario registry: the one place that knows every scenario's artifacts.

A scenario bundles what an experiment phase needs -- condition system prompts,
the matched prompt set, the group names, and the judge rubric. Generation
selects a scenario by name and records it in the run manifest; the scoring and
comparison stages resolve the scenario from the manifest rather than taking a
flag, so a corpus can never be judged with the wrong rubric or compared under
the wrong group names.

Frozen scenario modules are never edited; a new phase adds new modules and a
new entry here.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_SCENARIO = "water_commons"


@dataclass(frozen=True)
class Scenario:
    name: str
    groups: tuple[str, str]  # (group_a, group_b); predicted organism gap a - b > 0
    build_system_prompt: Callable[[str], str]
    build_prompt_set: Callable[[], list[dict]]
    judge_system_prompt: str


def _water_commons() -> Scenario:
    from src.scenario.condition_system_prompts import build_system_prompt
    from src.scenario.matched_prompt_set import GROUPS, build_prompt_set
    from src.score.stance_judge_rubric import JUDGE_SYSTEM_PROMPT

    return Scenario(
        name="water_commons",
        groups=GROUPS,
        build_system_prompt=build_system_prompt,
        build_prompt_set=build_prompt_set,
        judge_system_prompt=JUDGE_SYSTEM_PROMPT,
    )


_BUILDERS: dict[str, Callable[[], Scenario]] = {
    "water_commons": _water_commons,
}


def scenario_names() -> list[str]:
    return sorted(_BUILDERS)


def get_scenario(name: str = DEFAULT_SCENARIO) -> Scenario:
    try:
        builder = _BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"unknown scenario {name!r}; available: {', '.join(scenario_names())}"
        ) from None
    return builder()


def scenario_for_run(run_dir: str | Path) -> Scenario:
    """Resolve the scenario a run was generated with, from its manifest.

    Runs generated before scenarios existed carry no 'scenario' field; they
    were all water_commons, so that is the fallback.
    """
    manifest_path = Path(run_dir) / "generation_manifest.json"
    name = None
    if manifest_path.exists():
        name = json.loads(manifest_path.read_text()).get("scenario")
    if name is None:
        print(
            f"no scenario recorded in {manifest_path}; assuming {DEFAULT_SCENARIO!r}",
            flush=True,
        )
        name = DEFAULT_SCENARIO
    return get_scenario(name)
