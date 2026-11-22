"""Initial query promotion preserves the dataset, filtering, and resume cursor."""
from copy import deepcopy
import random

import pytest

from agmem.dapo_sampling import DynamicGroupSampler, validate_dapo_config
from tests.test_dapo_sampling import examples, group_for, settings


def shuffled_ids(source, seed):
    result = [example.sample_id for example in source]
    random.Random(seed).shuffle(result)
    return result


def remaining_ids(sampler):
    result = []
    while (example := sampler._next()) is not None:
        result.append(example.sample_id)
    return result


def test_initial_queries_are_promoted_in_order_without_losing_or_repeating_examples():
    source = examples(12)
    initial = [source[index].sample_id for index in (8, 1, 5)]
    config = settings(initial_query_ids=initial)
    original_ids = [example.sample_id for example in source]
    baseline = shuffled_ids(source, config["seed"])

    actual = remaining_ids(DynamicGroupSampler(config, source))

    assert actual == initial + [sample_id for sample_id in baseline if sample_id not in initial]
    assert len(actual) == len(set(actual)) == len(source)
    assert set(actual) == set(original_ids)
    assert [example.sample_id for example in source] == original_ids


@pytest.mark.parametrize("explicit_empty", [False, True])
def test_default_and_empty_initial_queries_keep_original_epoch_shuffles(explicit_empty):
    config = settings(epochs=3, **({"initial_query_ids": []} if explicit_empty else {}))
    source = examples(9)
    assert remaining_ids(DynamicGroupSampler(config, source)) == [
        sample_id for epoch in range(3) for sample_id in shuffled_ids(source, config["seed"] + epoch)]


def test_initial_query_order_applies_only_to_epoch_zero():
    source = examples(9)
    initial = [example.sample_id for example in reversed(source)]
    config = settings(epochs=3, initial_query_ids=initial)

    actual = remaining_ids(DynamicGroupSampler(config, source))

    assert actual[:len(source)] == initial
    assert actual[len(source):] == [
        sample_id for epoch in (1, 2) for sample_id in shuffled_ids(source, config["seed"] + epoch)]


@pytest.mark.parametrize("consumed", [1, 3, 5, 7])
def test_checkpoint_cursor_restores_same_future_candidates_and_batches(consumed):
    source = examples(5)
    config = settings(epochs=2, target_groups=1,
                      initial_query_ids=[source[index].sample_id for index in (4, 2, 0)])
    original = DynamicGroupSampler(config, source)
    for _ in range(consumed):
        original.collect(lambda example, index: group_for(example), 7)
    state = original.state_dict()
    restored = DynamicGroupSampler(deepcopy(config), source, **state)
    records = [[], []]

    for sampler, recorded in zip((original, restored), records):
        while True:
            groups, report = sampler.collect(
                lambda example, index: group_for(example, version=9), 9,
                record_candidate=lambda group, result: recorded.append(deepcopy(result)))
            if not groups:
                assert report["dataset_exhausted"]
                break

    assert records[0] == records[1]
    assert [row["candidate_id"] for row in records[0]] == list(range(consumed, 10))
    assert [row["batch_id"] for row in records[0]] == [f"batch-{index:08d}" for index in range(consumed, 10)]
    assert original.state_dict() == restored.state_dict()


@pytest.mark.parametrize("invalid", [None, "source0/train/0", ("source0/train/0",), {}, 1, [1], [True], [None]])
def test_initial_query_config_requires_a_list_of_strings(invalid):
    with pytest.raises(ValueError, match="initial_query_ids must be a list"):
        validate_dapo_config(settings(initial_query_ids=invalid))


def test_unknown_ids_are_checked_against_the_sampler_training_examples():
    config = settings(initial_query_ids=["missing/query"])
    validate_dapo_config(config)
    with pytest.raises(ValueError, match="unknown training sample IDs.*missing/query"):
        DynamicGroupSampler(config, examples(4))


def test_duplicate_initial_ids_are_rejected():
    sample_id = examples(1)[0].sample_id
    with pytest.raises(ValueError, match="initial_query_ids must contain unique"):
        DynamicGroupSampler(settings(initial_query_ids=[sample_id, sample_id]), examples(4))


@pytest.mark.parametrize("enable_initial_order", [False, True])
def test_duplicate_example_ids_are_rejected(enable_initial_order):
    source = examples(4)
    source[-1].sample_id = source[0].sample_id
    config = settings(initial_query_ids=[source[0].sample_id] if enable_initial_order else [])
    with pytest.raises(ValueError, match="examples must have unique sample_id"):
        DynamicGroupSampler(config, source)


def test_promoted_candidates_still_require_fresh_groups_and_pass_the_existing_filter():
    source = examples(5)
    initial = [example.sample_id for example in source]
    config = settings(initial_query_ids=initial, target_groups=2)
    sampler = DynamicGroupSampler(config, source)
    sampled, recorded = [], []

    def collect(example, candidate_id):
        sampled.append((example.sample_id, candidate_id))
        # Promotion does not make the first query eligible.
        return group_for(example, (-1, -1) if candidate_id == 0 else (-1, 1))

    groups, report = sampler.collect(
        collect, 7, record_candidate=lambda group, result: recorded.append(result))

    assert sampled == list(zip(initial[:3], range(3)))
    assert [row["accepted"] for row in recorded] == [False, True, True]
    assert [row["selected"] for row in recorded] == [False, True, True]
    assert [group[0].sample_id for group in groups] == initial[1:3]
    assert all(len(group) == config["training"]["group_size"] for group in groups)
    assert report["candidate_groups"] == 3
    assert report["selected_groups"] == 2
    assert sampler.next_index == 3
