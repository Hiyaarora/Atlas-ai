"""Deterministic offline embeddings.

Not a stub that returns zeros. This produces vectors with a real, useful
property: texts sharing words land near each other in the space. That makes
retrieval genuinely *testable* — a test can assert that asking about "postgres
indexing" returns the postgres chunk rather than the cooking one, with no
network, no API key, no cost, and identical results on every run.

The technique is the hashing trick: hash each token to a dimension and
accumulate. It is a bag-of-words model, so it has no notion of synonymy — but
it is deterministic and directionally correct, which is exactly what a test
needs and exactly what a real embedding model cannot offer.
"""

import hashlib
import math
import re

from app.embeddings.base import EmbeddingProvider, EmbeddingPurpose

_TOKEN = re.compile(r"[a-z0-9]+")


class FakeEmbeddingProvider(EmbeddingProvider):
    name = "fake"

    def __init__(self, dimensions: int = 768) -> None:
        self.model = "fake-hash-1"
        self.dimensions = dimensions

    async def embed(self, texts: list[str], *, purpose: EmbeddingPurpose) -> list[list[float]]:
        # `purpose` is deliberately ignored: the point of this provider is that
        # a document and a query containing the same words match.
        return [self._vectorise(text) for text in texts]

    def _vectorise(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions

        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            # Sign from a separate byte, so tokens do not all push the same
            # direction and unrelated texts are not trivially similar.
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        # L2-normalise so cosine similarity is a plain dot product and every
        # vector has the same magnitude, matching how real providers behave.
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            # Text with no alphanumeric tokens. Return a fixed unit vector
            # rather than zeros, which would make cosine similarity undefined.
            vector[0] = 1.0
            return vector

        return [value / magnitude for value in vector]
