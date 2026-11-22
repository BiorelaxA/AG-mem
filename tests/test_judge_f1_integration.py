"""Judge/F1 wiring, benchmark prompts and real CPU DAPO/TensorBoard integration."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from agmem.core import Example, Trajectory, load_config
from agmem.dapo import DAPOUpdater
from agmem.monitoring import summary_writer_class
from agmem.pipeline import evaluate_with_policy, make_reward_scorer, make_task_agent
from agmem.rewards import RewardScorer
from agmem.store import Store
from agmem.training_dapo import run_training_loop
from tests.test_training_dapo import collector, make_components


SNAPSHOT = Path(__file__).resolve().parents[1] / "configs/protocols/format05_training_judge.json"
CONFIG = "configs/train_dapo_locomo_lmes_hierarchical_f1_2gpu.yaml"


def config():
    value = load_config("configs/train_dapo_hierarchical_f1.yaml")
    value["reward"].update(formula="judge_f1_cost_v1", judge_f1_alpha=0.8, format_penalty=0.1)
    value["reward"]["judge"].update(template_file=str(SNAPSHOT))
    value["training"]["filter_metric"] = "judge_nonzero_advantage"
    value["service"].update(model="qwen3.5-27b", base_url="http://127.0.0.1:8000/v1",
                            request_base_url="http://127.0.0.1:18000/v1", retries=0)
    return value


def examples(split="train"):
    return [Example(f"locomo/{split}/0", "locomo", split, "m", "What gift?", gold_answers=["pendants"]),
            Example(f"split_longmemeval_s/{split}/1", "split_longmemeval_s", split, "m", "Where?",
                    gold_answers=["Paris"], judge_metadata={"question_id": "q", "question_type": "single-session-user"}),
            Example(f"split_longmemeval_s/{split}/2", "split_longmemeval_s", split, "m", "Unknown fact?",
                    gold_answers=["Not enough information"],
                    judge_metadata={"question_id": "q_abs", "question_type": "multi-session"})]


def test_mixed_recipe_switches_reward_without_overwriting_pure_f1_recipe():
    current = load_config(CONFIG)
    assert current["reward"]["formula"] == "judge_f1_cost_v1"
    assert current["reward"]["judge_f1_alpha"] == 0.8
    assert current["reward"]["format_penalty"] == 0.1
    assert current["reward"]["visible_navigation_penalty"] == 0.1
    assert current["retrieval"]["max_open_directories"] == 1
    assert current["reward"]["read_call_cost"] == 0.05
    assert current["training"]["filter_metric"] == "judge_nonzero_advantage"
    assert "context3_open1_repeatneg" in current["training"]["output_dir"]
    assert current["training"]["visible_navigation_negative_advantage"] == {
        "enabled": True, "magnitude": 1.0, "include_repeated_segments": True}
    assert current["training"]["checkpoint_retention"] == "best_and_latest"
    assert current["training"]["validation_primary_metric"] == "outcome_f1"
    assert current["reward"]["judge"]["template_file"] is None
    original = load_config("configs/train_dapo_hierarchical_f1.yaml")
    assert original["reward"]["formula"] == "f1_cost_v1"


def test_task_and_judge_share_local_model_with_strict_and_abstention_prompts(tmp_path):
    cfg, requests = config(), []
    store = Store(tmp_path / "store")

    def respond(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        prompt = body["messages"][0]["content"]
        if "History Chats:" in prompt:
            text = "<answer>A pendant</answer>"
        elif "Core principle" in prompt:
            text = '{"label":"CORRECT"}'
        else:
            text = "yes"
        return httpx.Response(200, json={"choices": [
            {"finish_reason": "stop", "message": {"content": text}}]})

    with patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "frozen-27b"}), \
         patch("agmem.policy.load_tokenizer", side_effect=AssertionError("Likelihood is disabled")):
        scorer = make_reward_scorer(cfg, store)
        task = make_task_agent(cfg, store)
    try:
        for service in (scorer.judge.service, task.service):
            service.client.close()
            service.client = httpx.Client(transport=httpx.MockTransport(respond))
        assert scorer.judge.service.identity == task.service.identity
        assert scorer.judge.service.role == "judge"
        assert task.service.role == "task_agent"
        ex = examples()
        prediction, _ = task.answer(ex[0], [])
        row = Trajectory(ex[0].sample_id, 0, "m", prediction=prediction, n_read_calls=2)
        scorer.score(ex[0], row)
        assert row.answer_f1 == 0  # Singular/plural mismatch is rescued by Judge.
        assert row.judge_score == 1 and row.outcome == 0.8
        assert row.reward == pytest.approx(0.5)
        assert "strict granularity" in requests[-1][1]["messages"][0]["content"]
        for item in ex[1:]:
            scorer.score(item, Trajectory(item.sample_id, 0, "m", prediction="Answer"))
        assert "contains the correct answer" in requests[-2][1]["messages"][0]["content"]
        assert "unanswerable question" in requests[-1][1]["messages"][0]["content"]
        assert all(url == "http://127.0.0.1:18000/v1/chat/completions" for url, _ in requests)
        for _, body in requests[1:]:
            assert body["model"] == "qwen3.5-27b"
            assert body["temperature"] == 0 and body["max_tokens"] == 128
            assert body["chat_template_kwargs"]["enable_thinking"] is False
        provenance = scorer.provenance()
        assert provenance["judge"]["template_sha256"]["locomo_refined"] == (
            "fd44ba60ff07d28f4b44057a96a80cfbb45e2a7973dab7aae8051c7aabad06ea")
        assert provenance["judge"]["model"]["model_revision"] == "frozen-27b"
    finally:
        scorer.close()
        task.service.client.close()
        store.close()


def test_real_dapo_update_logs_raw_f1_judge_and_q_in_every_scope(tmp_path):
    cfg = config()
    cfg["training"].update(output_dir=str(tmp_path), group_size=2, target_groups=2,
        mini_batch_groups=2, candidate_groups=2, max_num_gen_batches=1, validation_every=1,
        checkpoint_every=0, epochs=1)
    cfg["execution"].update(sampling_group_concurrency=1)
    judge = SimpleNamespace(score=lambda example, prediction: float(prediction.startswith("Correct")))
    scorer = RewardScorer(cfg["reward"], judge=judge, max_ids=1)
    captured = []

    def collect(config, env, policy, agent, example, candidate_index, scorer=None):
        rows = collector(config, env, policy, agent, example, candidate_index)
        for i, row in enumerate(rows):
            # Incorrect answer gets higher lexical overlap; Judge must control q.
            row.prediction = "Correct synonym" if i else example.gold_answers[0]
            row.n_read_calls = 2
            scorer.score(example, row)
        captured.extend(rows)
        return rows

    def validate(config, store, policy, split, output, limit=None):
        # Exercise actual evaluation/reporting logic with deterministic saved outputs.
        ex = examples("validation")[:2]
        rows = deepcopy(captured[:2])
        for example, row in zip(ex, rows):
            row.sample_id = example.sample_id
        fake_agent = SimpleNamespace(service=SimpleNamespace(client=SimpleNamespace(close=lambda: None)))
        fake_scorer = SimpleNamespace(close=lambda: None, provenance=lambda: {"formula": "judge_f1_cost_v1"})
        with patch("agmem.pipeline.validate_directories"), \
             patch("agmem.pipeline.make_environment", return_value=SimpleNamespace(version="frozen")), \
             patch("agmem.pipeline.make_task_agent", return_value=fake_agent), \
             patch("agmem.pipeline.make_reward_scorer", return_value=fake_scorer), \
             patch("agmem.pipeline.task_examples", return_value=ex), \
             patch("agmem.pipeline.scored_rollout", side_effect=rows):
            output.mkdir(parents=True, exist_ok=True)
            return evaluate_with_policy(config, store, policy, split, output, limit)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    writer = summary_writer_class()(log_dir=str(tmp_path / "tensorboard"))
    try:
        policy, optimizer = make_components()
        with patch("agmem.pipeline.collect_group", side_effect=collect), \
             patch("agmem.pipeline.evaluate_with_policy", side_effect=validate), redirect_stdout(StringIO()):
            run_training_loop(cfg, None, policy, optimizer, DAPOUpdater(policy, optimizer, cfg["training"]),
                              SimpleNamespace(version="frozen"), None, scorer, examples()[:2],
                              tmp_path, writer, max_groups=1)
        assert policy.policy_version == 1
    finally:
        writer.close()
        torch.set_num_threads(threads)
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    expected = {"answer_f1": 0.5, "outcome_f1": 0.5, "judge_accuracy": 0.5,
                "outcome_q": 0.4, "outcome_reward": -0.2, "reward": -0.3}
    for prefix in ("sampling", "train", "validation", "validation/source_macro"):
        for key, value in expected.items():
            assert events.Scalars(f"{prefix}/{key}")[-1].value == pytest.approx(value)
    for source in ("locomo", "split_longmemeval_s"):
        assert events.Scalars(f"sampling/by_source/{source}/judge_accuracy")[-1].value == 0.5
        assert events.Scalars(f"sampling/by_source/{source}/outcome_q")[-1].value == pytest.approx(0.4)
    reports = [json.loads(line) for line in (tmp_path / "train_metrics.jsonl").read_text().splitlines()]
    assert reports[0]["outcome_f1"] == 0.5
    assert reports[0]["outcome_q"] == 0.4
    validation = json.loads((tmp_path / "validation/step-00000001/metrics.json").read_text())
    assert validation["primary_outcome_metric"] == "outcome_f1"

