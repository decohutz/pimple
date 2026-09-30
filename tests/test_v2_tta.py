"""Exact D4 geometry and probability-mixture semantics, without GPU or datasets."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp, softmax
import torch

from scripts.v2_metrics import classification_metrics
from scripts.v2_tta import D4_VIEWS, aggregate_view_logits, paired_group_interval, transform_view


def test_d4_has_eight_named_distinct_pixel_preserving_views():
    assert D4_VIEWS == ("identity", "hflip", "vflip", "rot180", "rot90", "rot270", "transpose", "antitranspose")
    batch = torch.arange(2 * 3 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3, 3)
    before = batch.clone()
    expected = {
        "identity": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        "hflip": [[2, 1, 0], [5, 4, 3], [8, 7, 6]],
        "vflip": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "rot180": [[8, 7, 6], [5, 4, 3], [2, 1, 0]],
        "rot90": [[2, 5, 8], [1, 4, 7], [0, 3, 6]],
        "rot270": [[6, 3, 0], [7, 4, 1], [8, 5, 2]],
        "transpose": [[0, 3, 6], [1, 4, 7], [2, 5, 8]],
        "antitranspose": [[8, 5, 2], [7, 4, 1], [6, 3, 0]],
    }
    outputs = []
    for view in D4_VIEWS:
        output = transform_view(batch, view)
        assert output.shape == batch.shape and output.dtype == batch.dtype
        assert output.device == batch.device
        assert output[0, 0].tolist() == expected[view]
        # Each image/channel retains its own exact pixel multiset: no interpolation,
        # black corners, sample permutation or channel mixing is permitted.
        torch.testing.assert_close(output.flatten(-2).sort(-1).values,
                                   batch.flatten(-2).sort(-1).values, rtol=0, atol=0)
        outputs.append(output[0, 0].flatten().tolist())
    assert len({tuple(values) for values in outputs}) == 8
    torch.testing.assert_close(batch, before, rtol=0, atol=0)


def test_d4_inverse_preserves_batch_and_channel_identity():
    batch = torch.arange(4 * 3 * 5 * 5).reshape(4, 3, 5, 5)
    inverses = {name: name for name in D4_VIEWS}
    inverses.update(rot90="rot270", rot270="rot90")
    for view, inverse in inverses.items():
        restored = transform_view(transform_view(batch, view), inverse)
        torch.testing.assert_close(restored, batch, rtol=0, atol=0)


@pytest.mark.parametrize("shape,view", [
    ((3, 3, 3), "identity"),
    ((2, 3, 3, 4), "rot90"),
    ((2, 3, 3, 3), "diagonal_guess"),
])
def test_d4_rejects_ambiguous_input_geometry(shape, view):
    with pytest.raises(ValueError):
        transform_view(torch.zeros(shape), view)


def test_aggregation_is_mean_of_probabilities_and_keeps_row_class_order():
    rng = np.random.default_rng(406)
    logits = rng.normal(size=(8, 5, 7)).astype(np.float32)
    indices = [0, 2, 5, 7]
    actual = aggregate_view_logits(logits, indices)
    expected = softmax(logits[indices].astype(np.float64), axis=-1).mean(axis=0)
    assert actual.shape == (5, 7) and actual.dtype == np.float64
    np.testing.assert_allclose(np.exp(actual), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(logsumexp(actual, axis=1), 0, rtol=0, atol=1e-12)
    assert not np.allclose(expected, softmax(logits[indices].mean(axis=0), axis=-1))
    # Permuting view order cannot change a uniform mixture.
    np.testing.assert_allclose(aggregate_view_logits(logits, indices[::-1]), actual,
                               rtol=1e-12, atol=1e-12)


def test_identity_matches_original_softmax_without_using_other_views():
    logits = np.arange(8 * 3 * 7, dtype=np.float64).reshape(8, 3, 7) / 11
    expected = logits[0] - logsumexp(logits[0], axis=1, keepdims=True)
    actual = aggregate_view_logits(logits, [0])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    logits[1:] *= -100
    np.testing.assert_allclose(aggregate_view_logits(logits, [0]), actual,
                               rtol=1e-12, atol=1e-12)


def test_mixture_is_invariant_to_each_views_additive_logit_constant():
    rng = np.random.default_rng(812)
    logits = rng.normal(size=(8, 4, 7))
    offsets = rng.uniform(-10_000, 10_000, size=(8, 4, 1))
    np.testing.assert_allclose(aggregate_view_logits(logits + offsets, list(range(8))),
                               aggregate_view_logits(logits, list(range(8))),
                               rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [8], [0.5]])
def test_mixture_rejects_invalid_or_repeated_views(indices):
    with pytest.raises(ValueError):
        aggregate_view_logits(np.zeros((8, 3, 7)), indices)


@pytest.mark.parametrize("shape", [(3, 7), (8, 3, 6), (8, 0, 7)])
def test_mixture_rejects_invalid_shape_or_empty_predictions(shape):
    with pytest.raises(ValueError):
        aggregate_view_logits(np.zeros(shape), [0])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_mixture_rejects_nonfinite_scores_even_in_unselected_views(value):
    logits = np.zeros((8, 3, 7))
    logits[7, 2, 6] = value
    with pytest.raises(ValueError):
        aggregate_view_logits(logits, [0])


def test_paired_interval_matches_whole_group_resampling_with_unequal_group_sizes():
    y = np.array([0, 0, 0, 1, 2, 2, 3, 4, 5, 6])
    groups = np.array(["a", "a", "a", "b", "c", "c", "d", "e", "f", "g"])
    reference = np.array([0, 1, 1, 1, 2, 0, 3, 5, 5, 0])
    candidate = np.array([0, 0, 1, 1, 2, 2, 4, 4, 5, 6])
    seed, replicates = 614, 80
    actual = paired_group_interval(y, reference, candidate, groups, seed=seed, replicates=replicates)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        sampled = rng.integers(len(unique), size=len(unique))
        indices = np.concatenate([np.flatnonzero(groups == unique[i]) for i in sampled])
        deltas.append(classification_metrics(y[indices], candidate[indices])["macro_f1"]
                      - classification_metrics(y[indices], reference[indices])["macro_f1"])
    assert actual["group_unit"] == "group_id" and actual["groups"] == len(unique)
    assert actual["lower"] == pytest.approx(np.quantile(deltas, .025))
    assert actual["upper"] == pytest.approx(np.quantile(deltas, .975))


def test_paired_interval_for_identical_predictions_has_zero_difference():
    y = np.tile(np.arange(7), 2)
    predicted = np.roll(y, 1)
    groups = np.repeat(np.arange(7), 2)
    result = paired_group_interval(y, predicted, predicted, groups, replicates=20)
    assert result["lower"] == 0 and result["upper"] == 0


def test_paired_interval_reverses_sign_when_reference_and_candidate_are_swapped():
    y = np.tile(np.arange(7), 3)
    groups = np.repeat(np.arange(7), 3)
    reference = y.copy()
    reference[::3] = (reference[::3] + 1) % 7
    candidate = y.copy()
    candidate[::5] = (candidate[::5] + 2) % 7
    forward = paired_group_interval(y, reference, candidate, groups, replicates=80, seed=845)
    reverse = paired_group_interval(y, candidate, reference, groups, replicates=80, seed=845)
    assert reverse["lower"] == pytest.approx(-forward["upper"])
    assert reverse["upper"] == pytest.approx(-forward["lower"])


@pytest.mark.parametrize("y,reference,candidate,groups", [
    ([0, 1, 2], [0, 1], [0, 1, 2], ["a", "b", "c"]),
    ([0, 1, 2], [0, 1, 2], [0, 1, 2], ["a", "b"]),
    ([[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]], [["a", "b"], ["c", "d"]]),
    ([], [], [], []),
])
def test_paired_interval_rejects_unaligned_or_empty_samples(y, reference, candidate, groups):
    with pytest.raises(ValueError):
        paired_group_interval(y, reference, candidate, groups, replicates=20)
