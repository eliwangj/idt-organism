"""The registry must hand back the frozen artifacts exactly, and the manifest
lookup must never guess silently wrong: an unknown name fails loudly, and a
manifest without a scenario field falls back to water_commons (every pre-registry
run was water_commons)."""

import json

import pytest

from src.scenario.registry import (
    DEFAULT_SCENARIO,
    get_scenario,
    scenario_for_run,
    scenario_names,
)


def test_default_scenario_is_water_commons():
    scenario = get_scenario()
    assert scenario.name == DEFAULT_SCENARIO == "water_commons"
    assert scenario.groups == ("Rivertown", "Hillcrest")


def test_water_commons_returns_the_frozen_artifacts():
    from src.scenario.condition_system_prompts import build_system_prompt
    from src.scenario.matched_prompt_set import build_prompt_set
    from src.score.stance_judge_rubric import JUDGE_SYSTEM_PROMPT

    scenario = get_scenario("water_commons")
    assert scenario.build_system_prompt is build_system_prompt
    assert scenario.build_prompt_set is build_prompt_set
    assert scenario.judge_system_prompt == JUDGE_SYSTEM_PROMPT


def test_unknown_scenario_raises_with_available_names():
    with pytest.raises(KeyError) as excinfo:
        get_scenario("nope")
    message = str(excinfo.value)
    assert "nope" in message
    for name in scenario_names():
        assert name in message


def test_scenario_for_run_reads_manifest(tmp_path):
    (tmp_path / "generation_manifest.json").write_text(
        json.dumps({"scenario": "water_commons"})
    )
    assert scenario_for_run(tmp_path).name == "water_commons"


def test_scenario_for_run_falls_back_without_manifest_or_field(tmp_path):
    assert scenario_for_run(tmp_path).name == DEFAULT_SCENARIO

    (tmp_path / "generation_manifest.json").write_text(json.dumps({"model_id": "x"}))
    assert scenario_for_run(tmp_path).name == DEFAULT_SCENARIO
