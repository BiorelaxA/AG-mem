from agmem.checkpoints import best_checkpoint, select_best_checkpoints, validation_scores
from agmem.core import atomic_json


def committed(root, step, reward, f1, *, primary=True):
    path = root / f"checkpoint-{step:08d}"
    atomic_json(path / "model/config.json", {})
    (path / "trainer.pt").write_bytes(b"fixture")
    report = {"source_macro": {"reward": reward, "outcome_f1": f1}}
    if primary:
        report["primary_outcome_metric"] = "outcome_f1"
    atomic_json(path / "validation_metrics.json", report)
    return path


def test_primary_f1_beats_higher_reward_for_default_best(tmp_path):
    reward_best = committed(tmp_path, 1, 0.8, 0.8)
    f1_best = committed(tmp_path, 2, 0.7, 0.95)
    committed(tmp_path, 3, 0.7, 0.95)
    assert best_checkpoint(tmp_path) == f1_best
    assert best_checkpoint(tmp_path, "best_reward") == reward_best
    assert best_checkpoint(tmp_path, "best_outcome") == f1_best
    assert select_best_checkpoints(tmp_path)["best_primary"]["step"] == 2


def test_legacy_default_remains_reward_and_missing_judge_falls_back_to_f1(tmp_path):
    first = committed(tmp_path, 1, 0.8, 0.8, primary=False)
    committed(tmp_path, 2, 0.7, 0.95, primary=False)
    assert best_checkpoint(tmp_path) == first
    assert "best_primary" not in select_best_checkpoints(tmp_path)
    assert validation_scores({"source_macro": {"reward": 0.1, "outcome_f1": 0.7,
                                                "judge_accuracy": None}})["best_outcome"] == 0.7


def test_explicit_f1_primary_does_not_select_an_optional_diagnostic_judge():
    scores = validation_scores({"primary_outcome_metric": "answer_f1",
                                "source_macro": {"reward": 0.1, "answer_f1": 0.6,
                                                 "judge_accuracy": 0.95}})
    assert scores == {"best_reward": 0.1, "best_outcome": 0.6, "best_primary": 0.6}
