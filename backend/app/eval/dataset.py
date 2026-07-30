"""The golden set: questions paired with the passages that answer them.

WHY MARKERS AND NOT CHUNK IDS
-----------------------------
The obvious labelling scheme — "for question Q, chunk 7f3a… is relevant" — is
unusable here. Chunk ids are generated fresh on every ingestion, so the labels
would rot the first time a document is re-uploaded.

Worse, they would rot silently in exactly the experiment we most want to run.
Changing `chunk_size` re-cuts every document, so id-based labels cannot
express "this fact lives here" across two chunking strategies — and comparing
chunking strategies is one of the main things this harness exists for.

So relevance is expressed as a *content marker*: a distinctive phrase that
appears in the passage that answers the question. At evaluation time the
corpus is scanned and any chunk containing a marker is relevant. That survives
re-ingestion, re-chunking, and changing the embedding model.

The cost is that a marker must be genuinely distinctive. A marker matching
text in two unrelated places would quietly inflate the relevant set and make
every metric look better. `validate_dataset()` guards against this.

ON THE `expect_empty` QUERIES
-----------------------------
Roughly a quarter of the set asks questions the corpus cannot answer. This is
not padding. A retriever that returns its six nearest neighbours no matter
what will score well on recall and be actively harmful in production, because
the model treats whatever it is handed as evidence. Measuring refusal is what
stops "improve recall" from degrading the product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold case and collapse whitespace for marker matching.

    Source documents wrap at a column, so a phrase is routinely split across a
    newline — "…is not\\nacknowledged within…". A marker written as ordinary
    prose would then fail to match text that plainly contains it, and the
    failure looks like a retrieval bug rather than a formatting artefact.

    Caught by `validate_dataset` on the very first run, which is the argument
    for validating labels before trusting any metric computed from them.
    """
    return _WHITESPACE.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class EvalDocument:
    """One document in the evaluation corpus."""

    name: str
    text: str


@dataclass(frozen=True)
class EvalQuery:
    """One labelled question."""

    id: str
    document: str
    question: str

    #: Distinctive phrases identifying the passages that answer the question.
    #: A chunk is relevant if it contains any of them (case-insensitive).
    markers: tuple[str, ...] = ()

    #: True when the corpus genuinely cannot answer this. `markers` must be
    #: empty, and a correct system returns nothing at all.
    expect_empty: bool = False

    #: What this query probes. Reported per-category, because an aggregate
    #: number hides the tradeoff that matters: lexical gains bought with
    #: semantic losses net out to "no change".
    #:   lexical    - exact identifier or code; dense is weak here
    #:   semantic   - paraphrase sharing no vocabulary with the passage
    #:   mixed      - ordinary phrasing overlapping the passage somewhat
    #:   irrelevant - unanswerable; the system must refuse
    kind: str = "mixed"


def load_corpus() -> list[EvalDocument]:
    """Read the fixture documents from disk."""
    return [
        EvalDocument(name=path.name, text=path.read_text(encoding="utf-8"))
        for path in sorted(CORPUS_DIR.glob("*.md"))
    ]


#: The golden set.
#:
#: Deliberately small and hand-written. A large auto-generated set would need
#: an LLM to produce questions, which costs quota on every regeneration and —
#: more importantly — produces questions phrased in the document's own words.
#: Those flatter lexical retrieval and would make hybrid search look better
#: than it is. Real users paraphrase.
QUERIES: tuple[EvalQuery, ...] = (
    # ---- lexical: exact identifiers, where dense retrieval is weakest -----
    EvalQuery(
        id="lex-inc-4471",
        document="runbook.md",
        question="INC-4471",
        markers=("quarterly reindex job opened twenty-four parallel workers",),
        kind="lexical",
    ),
    EvalQuery(
        id="lex-inc-5238",
        document="runbook.md",
        question="what happened in INC-5238",
        markers=("intermediate certificate expired on a Saturday",),
        kind="lexical",
    ),
    EvalQuery(
        id="lex-e1007",
        document="runbook.md",
        question="E-1007",
        markers=("exceeded its deadline before",),
        kind="lexical",
    ),
    EvalQuery(
        id="lex-e2011",
        document="runbook.md",
        question="what does error code E-2011 mean",
        markers=("token was well formed but its signature did not verify",),
        kind="lexical",
    ),
    EvalQuery(
        id="lex-sev1",
        document="runbook.md",
        question="SEV1 definition",
        markers=("customer-facing writes are failing",),
        kind="lexical",
    ),
    # ---- semantic: paraphrases sharing little or no vocabulary -----------
    EvalQuery(
        id="sem-rollback",
        document="runbook.md",
        question="how do we undo a bad release",
        markers=("redeploying the previously published image tag",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-nobody-answers",
        document="runbook.md",
        question="what if nobody picks up the alert",
        markers=("not acknowledged within fifteen minutes",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-too-slow",
        document="runbook.md",
        question="how slow is too slow for a request",
        markers=("eight hundred milliseconds",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-lost-logs",
        document="replication.md",
        question="the backup server can no longer catch up and has to be rebuilt",
        markers=("already been recycled",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-data-loss",
        document="replication.md",
        question="can we lose recent transactions when the main server dies",
        markers=("commits on the primary without waiting for any", "failover can lose"),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-split-brain",
        document="replication.md",
        question="both servers think they are in charge after a failover",
        markers=("split timeline",),
        kind="semantic",
    ),
    # ---- mixed: ordinary phrasing with partial overlap -------------------
    EvalQuery(
        id="mix-slots",
        document="replication.md",
        question="what is a replication slot and what is the risk of using one",
        markers=("records the oldest position a",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-lag",
        document="replication.md",
        question="how do I measure replication lag correctly",
        markers=("reported in two different currencies",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-cascading",
        document="replication.md",
        question="can a standby serve other standbys",
        markers=("reduces the network burden on",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-pool",
        document="runbook.md",
        question="a batch job starved interactive traffic of connections",
        markers=("separated batch workloads onto their own pool",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-capacity",
        document="runbook.md",
        question="how is capacity planning done",
        markers=("previous period's peak rather than",),
        kind="mixed",
    ),
    # ---- lexical, second batch -------------------------------------------
    EvalQuery(
        id="lex-inc-6012",
        document="runbook.md",
        question="INC-6012",
        markers=("several thousand requests recomputed it simultaneously",),
        kind="lexical",
    ),
    EvalQuery(
        id="lex-inc-6390",
        document="runbook.md",
        question="what was INC-6390 about",
        markers=("debug-level logger was left enabled",),
        kind="lexical",
    ),
    EvalQuery(
        id="lex-e1004",
        document="runbook.md",
        question="E-1004",
        markers=("upstream dependency returned a malformed",),
        kind="lexical",
    ),
    EvalQuery(
        id="lex-e3300",
        document="runbook.md",
        question="what is error E-3300",
        markers=("tenant exceeded its configured quota",),
        kind="lexical",
    ),
    EvalQuery(
        id="lex-sev3",
        document="runbook.md",
        question="SEV3",
        markers=("workaround that can wait for business hours",),
        kind="lexical",
    ),
    # ---- semantic, second batch ------------------------------------------
    EvalQuery(
        id="sem-stampede",
        document="runbook.md",
        question="everything tried to rebuild the same cached value at once",
        markers=("randomised extension to expiry times",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-who-tells-customers",
        document="runbook.md",
        question="who tells customers what is going on during an outage",
        markers=("communications lead owns the status page",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-averages",
        document="runbook.md",
        question="why not just track the mean response time",
        markers=("conceals the shape of the distribution",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-making-it-worse",
        document="runbook.md",
        question="trying again could make an overloaded system worse",
        markers=("adds load to a system that is failing from load",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-nested-deadline",
        document="runbook.md",
        question="stop an inner call outliving the request waiting on it",
        markers=("decreasing budgets",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-dr-works",
        document="runbook.md",
        question="do we actually know our disaster recovery works",
        markers=("never been restored is a hypothesis",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-cant-serve-everything",
        document="runbook.md",
        question="what happens when we cannot serve all the traffic",
        markers=("sheds load deliberately",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-report-killed",
        document="replication.md",
        question="my long report on the read replica keeps getting killed",
        markers=("queries cancelled to let replay proceed",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-different-version",
        document="replication.md",
        question="send only some tables to a server on a newer version",
        markers=("feed a server running a different major version",),
        kind="semantic",
    ),
    EvalQuery(
        id="sem-two-writable",
        document="replication.md",
        question="what if the old server was only unreachable and not actually dead",
        markers=("two writable servers accepting",),
        kind="semantic",
    ),
    # ---- mixed, second batch ---------------------------------------------
    EvalQuery(
        id="mix-deploy",
        document="runbook.md",
        question="how are releases rolled out to the fleet",
        markers=("roll out to one instance first",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-flags",
        document="runbook.md",
        question="what is the policy on feature flags",
        markers=("fully on for two release cycles is removed",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-backup-retention",
        document="runbook.md",
        question="how long are backups retained",
        markers=("retention of thirty-five days",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-prod-access",
        document="runbook.md",
        question="who is allowed production access",
        markers=("granted for a limited window and expires automatically",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-postmortem",
        document="runbook.md",
        question="what goes into a postmortem",
        markers=("names systems rather than individuals",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-oncall",
        document="runbook.md",
        question="what is expected of someone on call",
        markers=("Shifts run for one week",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-replay-slow",
        document="replication.md",
        question="why does a standby fall behind the primary",
        markers=("Replay is single-threaded",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-which-standby",
        document="replication.md",
        question="which standby should be promoted when there are several",
        markers=("furthest ahead should be promoted",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-monitoring",
        document="replication.md",
        question="what should we alert on for replication health",
        markers=("disconnected standby is a silent loss of redundancy",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-archive",
        document="replication.md",
        question="how does archiving segments help a standby catch up",
        markers=("fetch the missing segments from the archive",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-sync-depth",
        document="replication.md",
        question="what levels of synchronous confirmation are there",
        markers=("flushed to the standby's disk",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-initial-copy",
        document="replication.md",
        question="how is a standby built in the first place",
        markers=("base copy of the data directory taken while the primary is running",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-logical-conflict",
        document="replication.md",
        question="what happens when a replicated row collides with a local one",
        markers=("halts replication until an operator resolves it",),
        kind="mixed",
    ),
    EvalQuery(
        id="mix-bandwidth",
        document="replication.md",
        question="how much network bandwidth does replication need",
        markers=("proportional to the volume of write-ahead log",),
        kind="mixed",
    ),
    # ---- irrelevant: the corpus cannot answer these ----------------------
    EvalQuery(
        id="neg-zebra",
        document="replication.md",
        question="zebra migration patterns across the serengeti",
        expect_empty=True,
        kind="irrelevant",
    ),
    EvalQuery(
        id="neg-recipe",
        document="runbook.md",
        question="how long should I bake sourdough at 220 degrees",
        expect_empty=True,
        kind="irrelevant",
    ),
    EvalQuery(
        id="neg-salary",
        document="runbook.md",
        question="what is the parental leave allowance for contractors",
        expect_empty=True,
        kind="irrelevant",
    ),
    # Hard negatives: the vocabulary overlaps heavily with the corpus, but the
    # fact genuinely is not there. These are what separate a retriever that
    # understands the question from one matching topic words. A system scoring
    # well on the easy negatives and failing these is pattern-matching.
    EvalQuery(
        id="neg-hard-kafka",
        document="replication.md",
        question="how do I configure synchronous replication for Kafka partitions",
        expect_empty=True,
        kind="irrelevant",
    ),
    EvalQuery(
        id="neg-hard-cap",
        document="replication.md",
        question="explain the CAP theorem tradeoffs when choosing a consistency level",
        expect_empty=True,
        kind="irrelevant",
    ),
    EvalQuery(
        id="neg-hard-sharding",
        document="replication.md",
        question="what is our sharding key and how do we rebalance shards",
        expect_empty=True,
        kind="irrelevant",
    ),
    EvalQuery(
        id="neg-hard-autoscale",
        document="runbook.md",
        question="what are the CPU thresholds for horizontal pod autoscaling",
        expect_empty=True,
        kind="irrelevant",
    ),
    EvalQuery(
        id="neg-hard-refund",
        document="runbook.md",
        question="what is the refund policy for annual subscriptions",
        expect_empty=True,
        kind="irrelevant",
    ),
)


def validate_dataset(chunk_texts_by_document: dict[str, list[str]]) -> list[str]:
    """Check the golden set against the ingested corpus.

    A silently wrong label is worse than no label: it produces confident
    metrics that measure nothing. Returns a list of problems, empty when the
    set is sound.
    """
    problems: list[str] = []
    documents = set(chunk_texts_by_document)

    for query in QUERIES:
        if query.document not in documents:
            problems.append(f"{query.id}: unknown document {query.document!r}")
            continue

        if query.expect_empty:
            if query.markers:
                problems.append(f"{query.id}: expect_empty query must have no markers")
            continue

        if not query.markers:
            problems.append(f"{query.id}: no markers and not marked expect_empty")
            continue

        chunks = [normalize(chunk) for chunk in chunk_texts_by_document[query.document]]
        for marker in query.markers:
            needle = normalize(marker)
            matches = sum(1 for chunk in chunks if needle in chunk)
            if matches == 0:
                problems.append(f"{query.id}: marker not found in corpus: {marker!r}")
            elif matches > 2:
                # Two is tolerable — chunk overlap legitimately duplicates a
                # phrase across a boundary. More than that means the marker is
                # not distinctive and is inflating the relevant set.
                problems.append(
                    f"{query.id}: marker matches {matches} chunks, not distinctive: {marker!r}"
                )

    return problems
