"""Provider-agnostic language model layer.

Import from here, never from a concrete provider module:

    from app.llm import ChatMessage, LLMProvider, get_llm_provider
"""

from app.llm.base import ChatMessage, LLMProvider, Role
from app.llm.factory import get_llm_provider

__all__ = ["ChatMessage", "LLMProvider", "Role", "get_llm_provider"]
