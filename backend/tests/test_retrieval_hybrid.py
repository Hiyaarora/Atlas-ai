"""BM25, RRF, and the hybrid gate.

BM25 and RRF are pure functions over data, so these assert their defining
*properties* rather than golden numbers copied from a run. A test that only
pins today's output would pass just as happily on a wrong implementation.
"""

from __future__ import annotations

import uuid

import pytest

from app.retrieval.base import RetrievedChunk
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical import BM25Index, tokenize
from app.retrieval.pipeline import _is_answerable, _passes_gate

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenizer_keeps_compound_identifiers_intact() -> None:
    """The reason lexical search exists is exact identifiers. Splitting
    "gpt-4" into "gpt" + "4" would destroy the signal before scoring."""
    assert tokenize("Model GPT-4 in config.py uses WM17S") == [
        "model",
        "gpt-4",
        "in",
        "config.py",
        "uses",
        "wm17s",
    ]


def test_tokenizer_drops_punctuation_and_lowercases() -> None:
    assert tokenize("Hello, World! (again)") == ["hello", "world", "again"]


def test_tokenizer_on_empty_input() -> None:
    assert tokenize("") == []
    assert tokenize("!!! ???") == []


# ---------------------------------------------------------------------------
# BM25 scoring properties
# ---------------------------------------------------------------------------


def test_rare_term_outscores_common_term() -> None:
    """idf is the whole point: a term in 1 of 5 documents must beat a term in
    all 5, even though both appear exactly once in the matching document."""
    corpus = [
        tokenize("the system stores data"),
        tokenize("the system stores data"),
        tokenize("the system stores data"),
        tokenize("the system stores data"),
        tokenize("the system stores data using WM17S"),
    ]
    index = BM25Index(corpus)

    rare = index.score(tokenize("wm17s"), 4)
    common = index.score(tokenize("system"), 4)

    assert rare > common


def test_stopwords_need_no_list_because_idf_handles_them() -> None:
    """A term present in every document contributes almost nothing."""
    corpus = [tokenize("the quick brown fox"), tokenize("the lazy brown dog")] * 10
    index = BM25Index(corpus)

    assert index.score(tokenize("the"), 0) == pytest.approx(0.0, abs=0.05)
    assert index.score(tokenize("quick"), 0) > 0.5


def test_term_frequency_saturates() -> None:
    """Doubling occurrences must NOT double the score, or one keyword-stuffed
    chunk would dominate every query."""
    once = BM25Index([tokenize("alpha beta gamma delta")] * 4 + [tokenize("target")])
    many = BM25Index([tokenize("alpha beta gamma delta")] * 4 + [tokenize("target " * 8)])

    single = once.score(tokenize("target"), 4)
    repeated = many.score(tokenize("target"), 4)

    assert repeated > single, "more occurrences should still score higher"
    assert repeated < single * 2, "but with strongly diminishing returns"


def test_longer_document_scores_lower_for_the_same_match() -> None:
    """Length normalisation: one hit in a short chunk is stronger evidence
    than one hit buried in a long chunk."""
    short = tokenize("transformer architecture")
    long = tokenize("transformer " + "filler words here " * 40)
    index = BM25Index([short, long, tokenize("unrelated text"), tokenize("more unrelated")])

    assert index.score(tokenize("transformer"), 0) > index.score(tokenize("transformer"), 1)


def test_absent_term_contributes_nothing() -> None:
    index = BM25Index([tokenize("alpha beta"), tokenize("gamma delta")])
    assert index.score(tokenize("nonexistent"), 0) == 0.0


def test_search_drops_zero_scoring_documents() -> None:
    """A chunk sharing no terms with the query is not a weak match — it is not
    a match. Returning it would hand fusion a rank to reward."""
    index = BM25Index([tokenize("alpha beta"), tokenize("gamma delta"), tokenize("alpha zeta")])
    results = index.search(tokenize("alpha"), top_k=10)

    assert [position for position, _ in results] == [0, 2]


def test_search_returns_best_first_and_respects_top_k() -> None:
    index = BM25Index(
        [tokenize("alpha"), tokenize("alpha alpha alpha"), tokenize("alpha alpha"), tokenize("x")]
    )
    results = index.search(tokenize("alpha"), top_k=2)

    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_empty_corpus_and_empty_query_are_safe() -> None:
    assert BM25Index([]).search(tokenize("anything"), top_k=5) == []
    assert BM25Index([tokenize("alpha")]).search([], top_k=5) == []


# ---------------------------------------------------------------------------
# idf coverage — the absolute floor for an unbounded score
# ---------------------------------------------------------------------------


def test_coverage_is_one_when_every_query_term_is_present() -> None:
    index = BM25Index([tokenize("postgres replication slots"), tokenize("unrelated")])
    assert index.idf_coverage(tokenize("postgres replication"), 0) == pytest.approx(1.0)


def test_coverage_counts_missing_terms_at_full_rarity() -> None:
    """A query term absent from the corpus is maximally rare, not neutral.
    Scoring it as idf 0 would erase it from the denominator and inflate
    coverage — the specific mistake that let irrelevant queries through."""
    index = BM25Index([tokenize("postgres replication slots")] * 5)

    # "postgres" is in every document (idf ~ 0); "zebra" is in none (idf high).
    # Matching only the worthless term must not look like half a match.
    assert index.idf_coverage(tokenize("zebra postgres"), 0) < 0.05


def test_coverage_weights_by_information_not_word_count() -> None:
    """Matching 1 of 5 words is fine if it is THE distinctive word, and
    useless if it is filler. Word-count coverage cannot express that."""
    corpus = [tokenize("the system uses WM17S for tracking")] + [
        tokenize("the system uses something else for tracking")
    ] * 9
    index = BM25Index(corpus)

    # One query, two chunks. The first contains both terms; the second
    # contains only the common one. By word count that is 2/2 vs 1/2 — but the
    # rare term carries nearly all the information, so the gap must be far
    # wider than "half".
    both = index.idf_coverage(tokenize("wm17s tracking"), 0)
    common_only = index.idf_coverage(tokenize("wm17s tracking"), 1)

    assert both == pytest.approx(1.0)
    assert common_only < 0.10


def test_search_rejects_the_least_irrelevant_of_a_uniformly_irrelevant_set() -> None:
    """The regression that an existing retrieval test caught: an unrelated
    query matching one throwaway word became the best lexical hit, and a
    purely relative cutoff passed it at ratio 1.0."""
    corpus = [
        tokenize("Postgres discards WAL segments the replica still needs"),
        tokenize("Index usage patterns are reported across pg_stat_user_indexes"),
    ]
    index = BM25Index(corpus)

    assert index.search(tokenize("zebra migration patterns across the serengeti"), top_k=5) == []


def test_search_still_finds_a_genuine_single_term_match() -> None:
    """The floor must not be so aggressive that exact-identifier lookup —
    the entire reason lexical retrieval exists — stops working."""
    corpus = [tokenize("the system uses WM17S for tracking")] + [
        tokenize("unrelated filler text here")
    ] * 9
    index = BM25Index(corpus)

    assert [position for position, _ in index.search(tokenize("WM17S"), top_k=5)] == [0]


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------


def _chunk(name: str, *, score: float, retriever: str, seed: int) -> RetrievedChunk:
    """A chunk whose id is deterministic in `seed`, so tie-breaks are stable."""
    return RetrievedChunk(
        chunk_id=uuid.UUID(int=seed),
        document_id=uuid.UUID(int=999),
        filename="doc.pdf",
        page_number=1,
        text=name,
        score=score,
        retriever=retriever,
    )


def test_fusion_rewards_agreement_over_single_confidence() -> None:
    """The defining property of RRF with k=60: a chunk ranked 2nd by BOTH
    retrievers beats one ranked 1st by only one of them."""
    dense = [
        _chunk("only-dense-loves-this", score=0.99, retriever="dense", seed=1),
        _chunk("both-agree", score=0.70, retriever="dense", seed=2),
    ]
    lexical = [
        _chunk("only-lexical-loves-this", score=42.0, retriever="lexical", seed=3),
        _chunk("both-agree", score=11.0, retriever="lexical", seed=2),
    ]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert fused[0].text == "both-agree"
    assert fused[0].retrievers == ("dense", "lexical")


def test_fusion_ignores_score_magnitude_entirely() -> None:
    """A lexical score of 5000 must not outweigh rank. This is what makes the
    two incomparable scales safe to combine."""
    dense = [_chunk("d", score=0.4, retriever="dense", seed=1)]
    lexical = [_chunk("l", score=5000.0, retriever="lexical", seed=2)]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert fused[0].score == pytest.approx(fused[1].score)


def test_fusion_preserves_provenance_for_gating_and_debugging() -> None:
    dense = [_chunk("shared", score=0.61, retriever="dense", seed=7)]
    lexical = [_chunk("shared", score=3.2, retriever="lexical", seed=7)]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert fused[0].ranks == {"dense": 1, "lexical": 1}
    assert fused[0].scores == {"dense": 0.61, "lexical": 3.2}


def test_fusion_deduplicates_by_chunk_id() -> None:
    dense = [_chunk("same", score=0.5, retriever="dense", seed=4)]
    lexical = [_chunk("same", score=2.0, retriever="lexical", seed=4)]

    assert len(reciprocal_rank_fusion([dense, lexical])) == 1


def test_fusion_is_deterministic_for_tied_scores() -> None:
    """Evaluation is only meaningful if repeated runs agree."""
    dense = [_chunk("a", score=0.5, retriever="dense", seed=10)]
    lexical = [_chunk("b", score=0.5, retriever="lexical", seed=11)]

    first = [chunk.chunk_id for chunk in reciprocal_rank_fusion([dense, lexical])]
    second = [chunk.chunk_id for chunk in reciprocal_rank_fusion([lexical, dense])]

    assert first == second


def test_fusion_of_a_single_ranking_preserves_its_order() -> None:
    dense = [
        _chunk("first", score=0.9, retriever="dense", seed=1),
        _chunk("second", score=0.8, retriever="dense", seed=2),
        _chunk("third", score=0.7, retriever="dense", seed=3),
    ]

    assert [chunk.text for chunk in reciprocal_rank_fusion([dense])] == [
        "first",
        "second",
        "third",
    ]


def test_fusion_handles_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# ---------------------------------------------------------------------------
# The evidence gate
# ---------------------------------------------------------------------------


def _fused(**scores: float):
    dense = (
        [_chunk("c", score=scores["dense"], retriever="dense", seed=1)] if "dense" in scores else []
    )
    lexical = (
        [_chunk("c", score=scores["lexical"], retriever="lexical", seed=1)]
        if "lexical" in scores
        else []
    )
    return reciprocal_rank_fusion([dense, lexical])[0]


def test_gate_admits_a_strong_dense_hit() -> None:
    assert _passes_gate(_fused(dense=0.62), best_lexical=0.0)


def test_gate_rejects_a_weak_dense_only_hit() -> None:
    assert not _passes_gate(_fused(dense=0.20), best_lexical=0.0)


def test_gate_admits_an_exact_match_that_dense_scored_poorly() -> None:
    """The case hybrid search exists for: a rare identifier the embedding
    model has no representation for, but BM25 matched exactly."""
    chunk = _fused(dense=0.28, lexical=8.0)
    assert _passes_gate(chunk, best_lexical=8.0)


def test_gate_rejects_a_marginal_lexical_hit() -> None:
    """A chunk matching only a common term, far below the best BM25 hit."""
    chunk = _fused(lexical=0.5)
    assert not _passes_gate(chunk, best_lexical=8.0)


@pytest.fixture
def calibrated_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the production thresholds, not the suite-wide test ones.

    conftest lowers these because the fake embedding provider has a different
    score distribution from gemini-embedding-001. These tests are about the
    gate's LOGIC at its real settings, so they state them explicitly instead of
    inheriting whatever the session happens to be configured with.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "retrieval_min_score", 0.55)
    monkeypatch.setattr(settings, "retrieval_min_margin", 0.010)
    monkeypatch.setattr(settings, "retrieval_margin_min_candidates", 4)


@pytest.mark.usefixtures("calibrated_thresholds")
def test_query_level_refusal_rejects_a_uniformly_mediocre_result_set() -> None:
    """The defect the golden set exposed: every chunk clears the per-chunk bar
    individually, nothing asks whether the BEST one is actually good."""
    fused = reciprocal_rank_fusion(
        [
            [
                _chunk("a", score=0.620, retriever="dense", seed=1),
                _chunk("b", score=0.617, retriever="dense", seed=2),
                _chunk("c", score=0.615, retriever="dense", seed=3),
                _chunk("d", score=0.614, retriever="dense", seed=4),
                _chunk("e", score=0.612, retriever="dense", seed=5),
            ]
        ]
    )
    # Best 0.620 clears the 0.55 floor comfortably, but stands only 0.005 above
    # the median — every chunk is equally close, the signature of a query whose
    # answer is not in this document.
    assert not _is_answerable(fused)


@pytest.mark.usefixtures("calibrated_thresholds")
def test_query_level_refusal_accepts_a_standout_chunk() -> None:
    fused = reciprocal_rank_fusion(
        [
            [
                _chunk("a", score=0.78, retriever="dense", seed=1),
                _chunk("b", score=0.61, retriever="dense", seed=2),
                _chunk("c", score=0.60, retriever="dense", seed=3),
                _chunk("d", score=0.60, retriever="dense", seed=4),
                _chunk("e", score=0.59, retriever="dense", seed=5),
            ]
        ]
    )
    assert _is_answerable(fused)


@pytest.mark.usefixtures("calibrated_thresholds")
def test_query_level_refusal_rejects_when_nothing_clears_the_floor() -> None:
    fused = reciprocal_rank_fusion(
        [
            [
                _chunk("a", score=0.51, retriever="dense", seed=1),
                _chunk("b", score=0.40, retriever="dense", seed=2),
                _chunk("c", score=0.38, retriever="dense", seed=3),
                _chunk("d", score=0.36, retriever="dense", seed=4),
            ]
        ]
    )
    assert not _is_answerable(fused)


@pytest.mark.usefixtures("calibrated_thresholds")
def test_query_level_refusal_skips_the_margin_test_on_too_few_candidates() -> None:
    """A median of two numbers says nothing about whether one stands out, so
    the margin test is skipped rather than answered from noise."""
    fused = reciprocal_rank_fusion(
        [
            [
                _chunk("a", score=0.62, retriever="dense", seed=1),
                _chunk("b", score=0.61, retriever="dense", seed=2),
            ]
        ]
    )
    assert _is_answerable(fused)


@pytest.mark.usefixtures("calibrated_thresholds")
def test_lexical_only_results_bypass_the_dense_refusal() -> None:
    """An exact-identifier match has no dense score to judge, and already
    passed the idf-coverage floor. Refusing it would discard the strongest
    evidence hybrid search produces."""
    fused = reciprocal_rank_fusion([[_chunk("id-match", score=9.4, retriever="lexical", seed=1)]])
    assert _is_answerable(fused)


def test_gate_is_relative_so_the_same_score_can_pass_or_fail() -> None:
    """BM25 is unbounded, so 3.0 is excellent against a best of 4.0 and poor
    against a best of 40.0. An absolute floor could not express that."""
    chunk = _fused(lexical=3.0)
    assert _passes_gate(chunk, best_lexical=4.0)
    assert not _passes_gate(chunk, best_lexical=40.0)
