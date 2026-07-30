"""Embedding provider contract.

Same shape as `app/llm`: business logic depends on this interface, never on a
vendor SDK, so the embedding model is a config value rather than a rewrite.

That matters more here than for generation. Changing embedding model
invalidates every stored vector — old and new vectors live in different
spaces and their similarities are meaningless against each other. The
interface exposes `model` and `dimensions` precisely so a migration can
detect the mismatch instead of silently returning nonsense.
"""

from abc import ABC, abstractmethod
from typing import Literal

#: Embeddings are asymmetric. A passage and a question about that passage are
#: different kinds of text, and telling the model which it is measurably
#: improves retrieval — the provider maps these to its own task hints.
EmbeddingPurpose = Literal["document", "query"]


class EmbeddingProvider(ABC):
    name: str
    model: str
    dimensions: int

    @abstractmethod
    async def embed(self, texts: list[str], *, purpose: EmbeddingPurpose) -> list[list[float]]:
        """Embed a batch of texts.

        Returns one vector per input, in the same order. Implementations must
        preserve order — callers zip the result against chunk metadata, and a
        reordering would silently attach every vector to the wrong chunk.

        Raises:
            EmbeddingError: the provider failed.
        """

    async def embed_one(self, text: str, *, purpose: EmbeddingPurpose) -> list[float]:
        vectors = await self.embed([text], purpose=purpose)
        return vectors[0]
