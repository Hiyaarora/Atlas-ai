"""The measuring instrument itself.

A wrong metric is worse than no metric — it produces confident numbers that
send you optimising the wrong thing. So these check the metrics against values
derived by hand, not against whatever the implementation currently returns.
"""

from __future__ import annotations

import uuid

import pytest

from app.eval.dataset import QUERIES, normalize, validate_dataset
from app.eval.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
    summarise,
)

A, B, C, D = (uuid.UUID(int=index) for index in range(1, 5))


# ---------------------------------------------------------------------------
# recall / precision
# ---------------------------------------------------------------------------


def test_recall_counts_relevant_found_over_relevant_total() -> None:
    assert recall_at_k([A, C], {A, B}, k=6) == pytest.approx(0.5)


def test_recall_respects_the_cutoff() -> None:
    """A relevant chunk at rank 3 does not count for recall@2."""
    assert recall_at_k([C, D, A], {A}, k=2) == 0.0
    assert recall_at_k([C, D, A], {A}, k=3) == 1.0


def test_recall_is_zero_when_nothing_is_labelled_relevant() -> None:
    """Guards a mislabelled query from inflating an average."""
    assert recall_at_k([A, B], set(), k=6) == 0.0


def test_precision_divides_by_results_returned_not_by_k() -> None:
    """Returning 2 results, both correct, is precision 1.0. Dividing by k
    would penalise a system for NOT padding — the opposite of the goal."""
    assert precision_at_k([A, B], {A, B}, k=6) == 1.0
    assert precision_at_k([A, C], {A, B}, k=6) == pytest.approx(0.5)


def test_precision_of_an_empty_result_is_zero() -> None:
    assert precision_at_k([], {A}, k=6) == 0.0


# ---------------------------------------------------------------------------
# MRR / nDCG
# ---------------------------------------------------------------------------


def test_reciprocal_rank_uses_the_first_hit_only() -> None:
    assert reciprocal_rank([A, B], {A, B}) == 1.0
    assert reciprocal_rank([C, A], {A}) == pytest.approx(0.5)
    assert reciprocal_rank([C, D, A], {A}) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_found() -> None:
    assert reciprocal_rank([C, D], {A}) == 0.0


def test_ndcg_is_one_for_a_perfect_ordering() -> None:
    assert ndcg_at_k([A, B, C], {A, B}, k=6) == pytest.approx(1.0)


def test_ndcg_penalises_the_same_hits_placed_lower() -> None:
    """The property MRR cannot express: two relevant results at ranks 1-2
    must beat the same two at ranks 3-4."""
    high = ndcg_at_k([A, B, C, D], {A, B}, k=6)
    low = ndcg_at_k([C, D, A, B], {A, B}, k=6)

    assert high == pytest.approx(1.0)
    assert low < high


def test_ndcg_matches_a_hand_computed_value() -> None:
    """One relevant chunk at rank 2: gain 1/log2(3), ideal 1/log2(2) = 1."""
    assert ndcg_at_k([C, A], {A}, k=6) == pytest.approx(0.6309, abs=1e-4)


# ---------------------------------------------------------------------------
# Scoring and aggregation
# ---------------------------------------------------------------------------


def test_unanswerable_query_is_scored_on_refusal_not_on_ranking() -> None:
    refused = score_query(
        query_id="neg", kind="irrelevant", expect_empty=True, retrieved=[], relevant=set(), k=6
    )
    noisy = score_query(
        query_id="neg", kind="irrelevant", expect_empty=True, retrieved=[A], relevant=set(), k=6
    )

    assert refused.refused is True
    assert noisy.refused is False
    assert noisy.returned == 1


def test_summary_keeps_answerable_and_unanswerable_separate() -> None:
    """Blending them would hide the tradeoff: a system returning nothing for
    everything scores 0.0 recall AND 100% refusal, and both must be visible."""
    results = [
        score_query(
            query_id="a", kind="lexical", expect_empty=False, retrieved=[A], relevant={A}, k=6
        ),
        score_query(
            query_id="n", kind="irrelevant", expect_empty=True, retrieved=[], relevant=set(), k=6
        ),
    ]
    summary = summarise(results)

    assert summary.answerable == 1
    assert summary.recall == 1.0
    assert summary.negatives == 1
    assert summary.refusal_rate == 1.0
    assert summary.noise_per_negative == 0.0


def test_summary_of_nothing_does_not_divide_by_zero() -> None:
    summary = summarise([])
    assert summary.recall == 0.0
    assert summary.refusal_rate == 0.0


# ---------------------------------------------------------------------------
# The golden set itself
# ---------------------------------------------------------------------------


def test_marker_matching_ignores_source_line_wrapping() -> None:
    """Markers are written as prose; the corpus wraps at a column, so the same
    phrase routinely contains a newline. Caught on the harness's first run."""
    assert normalize("not\nacknowledged  within fifteen") == "not acknowledged within fifteen"


def test_dataset_is_internally_consistent() -> None:
    """Structural checks that need no corpus: ids unique, negatives unlabelled,
    positives labelled."""
    ids = [query.id for query in QUERIES]
    assert len(ids) == len(set(ids)), "duplicate query ids"

    for query in QUERIES:
        if query.expect_empty:
            assert not query.markers, f"{query.id}: negative query must have no markers"
        else:
            assert query.markers, f"{query.id}: positive query needs markers"
        assert query.kind in {"lexical", "semantic", "mixed", "irrelevant"}


def test_validator_reports_a_marker_that_matches_nothing() -> None:
    problems = validate_dataset({"runbook.md": ["totally unrelated text"], "replication.md": []})
    assert any("marker not found" in problem for problem in problems)


def test_validator_reports_an_indistinctive_marker() -> None:
    """A marker matching many chunks silently inflates the relevant set and
    makes every metric look better than it is."""
    chunk = "the quarterly reindex job opened twenty-four parallel workers"
    problems = validate_dataset({"runbook.md": [chunk] * 5, "replication.md": []})
    assert any("not distinctive" in problem for problem in problems)
