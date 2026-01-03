"""
Core module - Shared context structures and utilities.
"""

from core.context import AgentContext
from core.strategist_context import (
    StrategistAgentContext,
    CommentSnapshot,
    ResearchOutput,
    StrategyOutput,
    RelatedIssue,
    KBSnippet,
)

__all__ = [
    "AgentContext",
    "StrategistAgentContext",
    "CommentSnapshot",
    "ResearchOutput",
    "StrategyOutput",
    "RelatedIssue",
    "KBSnippet",
]
