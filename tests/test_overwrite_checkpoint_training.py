"""CPU integration checks for bounded best/latest validation checkpoint writes."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml

from agmem import two_gpu_train
from agmem.checkpoints import best_checkpoint, committed_checkpoints
from agmem.core import Example, atomic_json, load_config
from agmem.dapo import DAPOUpdater
from agmem.dapo_recovery import reconcile_run
from agmem.pipeline import save_checkpoint
from agmem.training_dapo import run_training_loop
from tests.test_training_dapo import TinyLM, collector, make_components, test_config as make_config


POLICY = "best_and_latest_overwrite"
FULL_CONFIG = Path("configs/train_dapo_locomo_lmes_full_directory_orm.yaml")
SCORES = {2: 0.9, 4: 0.7, 6: 0.9, 8: 0.8}


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def overwrite_config(root, *, cadence=2):
    config = make_config(root)
    config["training"].update(checkpoint_retention=POLICY, validation_every=cadence,
                              checkpoint_every=1)
    return config


def validation_report(score):
    values = {"reward": 1 - score, "judge_accuracy": score,
              "answer_f1": score, "outcome_f1": score}
    return {"primary_outcome_metric": "outcome_f1", "source_macro": values,
            "pooled": values, "by_source": {"fixture": values}}


def evaluate(config, store, policy, split, destination, limit=None):
    report = validation_report(SCORES.get(policy.policy_version, 0.5))
    atomic_json(destination / "metrics.json", report)
    return report


def run_loop(root, config, policy, optimizer, *, steps=None, state=None, example_count=9):
    root.mkdir(parents=True, exist_ok=True)
    examples = [Example(f"fixture/train/{i}", "fixture", "train", f"memory-{i}",
                        "question", gold_answers=["answer"])
                for i in range(example_count)]
    updater = DAPOUpdater(policy, optimizer, config["training"])
    with patch("agmem.pipeline.collect_group", side_effect=collector) as sampling, \
         patch("agmem.pipeline.evaluate_with_policy", side_effect=evaluate) as evaluating, \
         redirect_stdout(StringIO()):
        run_training_loop(config, None, policy, optimizer, updater,
                          SimpleNamespace(version="fixture-v1"), None, None,
                          examples, root, state=state, max_groups=steps)
    return sampling, evaluating


def load_checkpoint(checkpoint):
    state = torch.load(checkpoint / "trainer.pt", weights_only=True)
    model = TinyLM()
    model.load_state_dict(torch.load(checkpoint / "model/weights.pt", weights_only=True))
    policy, optimizer = make_components(model, version=state["policy_version"])
    optimizer.load_state_dict(state["optimizer"])
    return state, policy, optimizer


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_nested_equal(left, right)
    else:
        assert actual == expected


def payload_directories(root):
    return sorted(path.name for path in root.iterdir()
                  if not path.is_symlink() and path.is_dir()
                  and re.fullmatch(r"checkpoint-[0-9]+(?:\.(?:building|pruning))?", path.name))


def assert_alias(root, alias, step):
    path = root / alias
    assert path.is_symlink()
    assert path.readlink() == Path(f"checkpoint-{step:08d}")
    assert (path / "model/weights.pt").is_file()
    assert (path / "trainer.pt").is_file()


def test_validation_saves_keep_full_best_and_latest_with_at_most_two_payloads(tmp_path):
    config = overwrite_config(tmp_path)
    policy, optimizer = make_components()
    captured, peak = {}, []
    real_torch_save = torch.save

    def capture_checkpoint(config, policy, optimizer, root, step, *args, **kwargs):
        captured[step] = {"model": deepcopy(policy.model.state_dict()),
                          "optimizer": deepcopy(optimizer.state_dict()),
                          "dapo_state": deepcopy(kwargs["dapo_state"])}
        return save_checkpoint(config, policy, optimizer, root, step, *args, **kwargs)

    def record_write(value, destination, *args, **kwargs):
        # Observe both the model and trainer writes while .building still exists.
        peak.append(payload_directories(tmp_path))
        assert len(peak[-1]) <= 2
        return real_torch_save(value, destination, *args, **kwargs)

    with patch("agmem.pipeline.save_checkpoint", side_effect=capture_checkpoint), \
         patch("torch.save", side_effect=record_write):
        _, evaluating = run_loop(tmp_path, config, policy, optimizer, steps=8)

    assert set(captured) == {2, 4, 6, 8}
    assert evaluating.call_count == 4
    assert max(map(len, peak)) == 2
    assert payload_directories(tmp_path) == ["checkpoint-00000002", "checkpoint-00000008"]
    assert_alias(tmp_path, "best", 2)  # The equal primary score at step 6 keeps step 2.
    assert_alias(tmp_path, "latest", 8)
    assert best_checkpoint(tmp_path) == tmp_path / "checkpoint-00000002"

    for step in (2, 8):
        checkpoint = tmp_path / f"checkpoint-{step:08d}"
        state, restored_policy, restored_optimizer = load_checkpoint(checkpoint)
        assert state["step"] == state["policy_version"] == step
        assert state["config"] == config
        assert state["optimizer"]["optimizer"]["state"]  # Adam moments were serialized.
        assert_nested_equal(restored_policy.model.state_dict(), captured[step]["model"])
        assert_nested_equal(restored_optimizer.state_dict(), captured[step]["optimizer"])
        assert state["dapo_state"] == captured[step]["dapo_state"]
        active = state["dapo_state"]["active"]
        assert active["consumed_groups"] == 2
        assert len(active["groups"]) == 3
        assert active["groups"][2][0]["decisions"][0]["completion"]["old_logprobs"]
        assert state["dapo_state"]["counters"]["trained_groups"] == step
        assert json.loads((checkpoint / "validation_metrics.json").read_text()) == validation_report(SCORES[step])


@pytest.mark.parametrize("validation", [None, validation_report(float("nan")),
                                         validation_report(float("inf"))])
def test_missing_or_nonfinite_validation_fails_before_writes_or_predeletion(tmp_path, validation):
    config = overwrite_config(tmp_path)
    policy, optimizer = make_components()
    run_loop(tmp_path, config, policy, optimizer, steps=4)
    before = {path.relative_to(tmp_path): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file() and not path.is_symlink()}
    with patch.object(policy.model, "save_pretrained", side_effect=AssertionError("must not write")), \
         patch("torch.save", side_effect=AssertionError("must not write")), \
         pytest.raises(ValueError, match="validation"):
        save_checkpoint(config, policy, optimizer, tmp_path, 6, 0, 6,
                        validation=validation, dapo_state={})
    assert payload_directories(tmp_path) == ["checkpoint-00000002", "checkpoint-00000004"]
    assert {path.relative_to(tmp_path): path.read_bytes()
            for path in tmp_path.rglob("*") if path.is_file() and not path.is_symlink()} == before
    assert_alias(tmp_path, "best", 2)
    assert_alias(tmp_path, "latest", 4)


@pytest.mark.parametrize("steps,example_count", [(3, 9), (None, 3)])
def test_nonvalidation_periodic_stop_and_dataset_end_do_not_save_or_evaluate(tmp_path, steps, example_count):
    config = overwrite_config(tmp_path, cadence=4)
    policy, optimizer = make_components()
    with patch("agmem.pipeline.save_checkpoint", wraps=save_checkpoint) as saving:
        _, evaluating = run_loop(tmp_path, config, policy, optimizer,
                                 steps=steps, example_count=example_count)
    assert policy.policy_version == 3
    saving.assert_not_called()
    evaluating.assert_not_called()
    assert payload_directories(tmp_path) == []
    assert not (tmp_path / "validation").exists()


@pytest.mark.parametrize("steps,example_count", [(5, 9), (None, 6)])
def test_nonvalidation_exit_preserves_last_validation_checkpoint(tmp_path, steps, example_count):
    config = overwrite_config(tmp_path, cadence=4)
    policy, optimizer = make_components()
    with patch("agmem.pipeline.save_checkpoint", wraps=save_checkpoint) as saving:
        _, evaluating = run_loop(tmp_path, config, policy, optimizer,
                                 steps=steps, example_count=example_count)
    assert policy.policy_version == (5 if steps else 6)
    assert saving.call_count == evaluating.call_count == 1
    assert saving.call_args.args[4] == 4
    assert payload_directories(tmp_path) == ["checkpoint-00000004"]
    assert_alias(tmp_path, "best", 4)
    assert_alias(tmp_path, "latest", 4)


def test_failed_save_recovers_best_without_archiving_payloads_and_resumes_frozen_groups(tmp_path):
    reference, interrupted = tmp_path / "reference", tmp_path / "interrupted"
    reference_config, config = overwrite_config(reference), overwrite_config(interrupted)
    expected_policy, expected_optimizer = make_components()
    run_loop(reference, reference_config, expected_policy, expected_optimizer, steps=8)
    policy, optimizer = make_components()
    real_torch_save = torch.save
    best_bytes = {}

    def fail_trainer_write(value, destination, *args, **kwargs):
        destination = Path(destination)
        assert len(payload_directories(interrupted)) <= 2
        if destination.name == "trainer.pt" and value["step"] == 6:
            best = interrupted / "checkpoint-00000002"
            best_bytes.update({path.relative_to(best): path.read_bytes()
                               for path in best.rglob("*") if path.is_file()})
            assert not (interrupted / "checkpoint-00000004").exists()
            destination.write_bytes(b"interrupted torch.save")
            raise OSError("simulated disk write failure")
        return real_torch_save(value, destination, *args, **kwargs)

    with patch("torch.save", side_effect=fail_trainer_write), \
         pytest.raises(OSError, match="simulated disk write failure"):
        run_loop(interrupted, config, policy, optimizer, steps=8)

    best = interrupted / "checkpoint-00000002"
    partial = interrupted / "checkpoint-00000006.building"
    assert partial.is_dir()
    assert (partial / "model/weights.pt").is_file()
    assert committed_checkpoints(interrupted) == [best]
    assert best_checkpoint(interrupted) == best
    assert {path.relative_to(best): path.read_bytes()
            for path in best.rglob("*") if path.is_file()} == best_bytes
    assert_alias(interrupted, "best", 2)
    assert_alias(interrupted, "latest", 2)

    state, restored_policy, restored_optimizer = load_checkpoint(best)
    frozen = deepcopy(state["dapo_state"])
    original_logs = (interrupted / "train_metrics.jsonl").read_bytes()
    result = reconcile_run(interrupted, state)  # Infer overwrite policy from saved config.
    assert state["dapo_state"] == frozen
    assert not partial.exists()
    archive = Path(result["archive"])
    assert (archive / "train_metrics.jsonl").read_bytes() == original_logs
    assert (archive / "validation/step-00000006/metrics.json").is_file()
    assert list(archive.rglob("trainer.pt")) == list(archive.rglob("weights.pt")) == []
    assert not any(path.name.startswith("checkpoint-") for path in archive.rglob("*"))
    assert [json.loads(line)["step"] for line in
            (interrupted / "train_metrics.jsonl").read_text().splitlines()] == [1, 2]
    assert_alias(interrupted, "best", 2)
    assert_alias(interrupted, "latest", 2)

    sampling, evaluating = run_loop(interrupted, config, restored_policy, restored_optimizer,
                                    steps=6, state=state)
    # The first resumed update consumes the third group saved at step 2; only
    # the two later batches (six candidates) are sampled again.
    assert sampling.call_count == 6
    assert evaluating.call_count == 3
    assert restored_policy.policy_version == 8
    assert_nested_equal(restored_policy.model.state_dict(), expected_policy.model.state_dict())
    assert_nested_equal(restored_optimizer.state_dict(), expected_optimizer.state_dict())
    actual_state = torch.load(interrupted / "latest/trainer.pt", weights_only=True)
    expected_state = torch.load(reference / "latest/trainer.pt", weights_only=True)
    for field in ("sampler", "counters"):
        assert actual_state["dapo_state"][field] == expected_state["dapo_state"][field]
    assert actual_state["dapo_state"]["active"]["groups"] == expected_state["dapo_state"]["active"]["groups"]
    assert payload_directories(interrupted) == ["checkpoint-00000002", "checkpoint-00000008"]
    assert_alias(interrupted, "best", 2)
    assert_alias(interrupted, "latest", 8)


def test_recovery_policy_override_discards_future_committed_payload_and_keeps_audit(tmp_path):
    config = overwrite_config(tmp_path)
    policy, optimizer = make_components()
    run_loop(tmp_path, config, policy, optimizer, steps=4)
    best, future = tmp_path / "checkpoint-00000002", tmp_path / "checkpoint-00000004"
    state = torch.load(best / "trainer.pt", weights_only=True)
    # The current configuration must control recovery when migrating a run
    # whose saved configuration still contains the older retention policy.
    state["config"]["training"]["checkpoint_retention"] = "best_and_latest"
    saved = (best / "trainer.pt").read_bytes()
    original_log = (tmp_path / "train_metrics.jsonl").read_bytes()
    assert future.is_dir()
    result = reconcile_run(tmp_path, state, checkpoint_policy=POLICY)
    assert not future.exists()
    assert (best / "trainer.pt").read_bytes() == saved
    assert result["discarded_checkpoints"] == [future.name]
    archive = Path(result["archive"])
    assert (archive / "train_metrics.jsonl").read_bytes() == original_log
    assert (archive / "validation/step-00000004/metrics.json").is_file()
    assert (archive / "groups/group-00000003.json.gz").is_file()
    assert not any(path.name.startswith("checkpoint-") for path in archive.rglob("*"))
    assert list(archive.rglob("trainer.pt")) == list(archive.rglob("weights.pt")) == []
    assert_alias(tmp_path, "best", 2)
    assert_alias(tmp_path, "latest", 2)
    again = reconcile_run(tmp_path, state, checkpoint_policy=POLICY)
    assert again["archive"] is None
    assert again["discarded_checkpoints"] == []


@pytest.mark.parametrize("damage", ["missing_checkpoint", "missing_trainer", "missing_model_config"])
def test_recovery_requires_complete_resume_checkpoint_before_any_mutation(tmp_path, damage):
    config = overwrite_config(tmp_path)
    policy, optimizer = make_components()
    run_loop(tmp_path, config, policy, optimizer, steps=4)
    checkpoint = tmp_path / "checkpoint-00000002"
    state = torch.load(checkpoint / "trainer.pt", weights_only=True)
    if damage == "missing_checkpoint":
        state["step"] = 1
    elif damage == "missing_trainer":
        (checkpoint / "trainer.pt").unlink()
    else:
        (checkpoint / "model/config.json").unlink()
    partial = tmp_path / "checkpoint-00000006.building"
    partial.mkdir()
    (partial / "trainer.pt").write_bytes(b"partial optimizer state")
    before = {path.relative_to(tmp_path): path.read_bytes()
              for path in tmp_path.rglob("*") if path.is_file() and not path.is_symlink()}
    aliases = {name: (tmp_path / name).readlink() for name in ("best", "latest")}
    with pytest.raises(ValueError, match="[Ii]ncomplete|complete checkpoint"):
        reconcile_run(tmp_path, state)
    assert {path.relative_to(tmp_path): path.read_bytes()
            for path in tmp_path.rglob("*") if path.is_file() and not path.is_symlink()} == before
    assert {name: (tmp_path / name).readlink() for name in aliases} == aliases
    assert payload_directories(tmp_path) == ["checkpoint-00000002", "checkpoint-00000004",
                                             "checkpoint-00000006.building"]
    assert not (tmp_path / "resume_orphans").exists()


def test_full_directory_recipe_alone_uses_validation_only_overwrite_policy():
    current = load_config(FULL_CONFIG)
    assert current["training"]["checkpoint_retention"] == POLICY
    assert current["training"]["checkpoint_every"] == 0
    assert current["training"]["validation_every"] == 4
    for path in ("configs/train_dapo_locomo_lmes_hierarchical_batch2first.yaml",
                 "configs/train_dapo_locomo_lmes_hierarchical_f1_2gpu.yaml"):
        assert load_config(path)["training"]["checkpoint_retention"] == "best_and_latest"


@pytest.mark.parametrize("changes", [{"algorithm": "grpo"}, {"validation_every": 0},
                                    {"validation_every": -1}])
def test_overwrite_policy_requires_dapo_and_positive_validation_cadence(tmp_path, changes):
    config = load_config(FULL_CONFIG)
    config["training"].update(changes)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="checkpoint_retention|validation_every|DAPO|dapo"):
        load_config(path)


def test_storage_guard_budgets_two_full_checkpoints_and_margin(tmp_path):
    config = overwrite_config(tmp_path / "new-run")
    with patch.object(two_gpu_train.shutil, "disk_usage",
                      return_value=SimpleNamespace(free=7 * two_gpu_train.GIB)):
        report = two_gpu_train.storage_report(config, 9 * two_gpu_train.GIB)
    assert report["estimated_full_checkpoint_bytes"] == 63 * two_gpu_train.GIB
    assert report["estimated_peak_checkpoint_copies"] == 2
    assert report["minimum_required_bytes"] == 136 * two_gpu_train.GIB
    assert report["recommended_free_bytes"] == report["minimum_required_bytes"]
    assert report["storage_ready"] is False
    assert report["filesystem_checked_at"] == str(tmp_path)
    assert not Path(config["training"]["output_dir"]).exists()
