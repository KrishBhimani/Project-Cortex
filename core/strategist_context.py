"""
Strategist Agent Context - Specialized context for strategic reasoning.

This module defines the StrategistAgentContext and supporting dataclasses
for the Strategist agent. Unlike the lightweight AgentContext, this provides
rich context including discussion history, prior agent outputs, and related issues.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass(frozen=True)
class CommentSnapshot:
    """Snapshot of an issue comment for context."""
    id: str
    body: str
    author_name: Optional[str]
    is_agent_response: bool
    created_at: datetime


@dataclass(frozen=True)
class ResearchOutput:
    """Summary of a prior Researcher agent execution."""
    execution_id: str
    summary: str
    sources: List[str]
    created_at: datetime


@dataclass(frozen=True)
class StrategyOutput:
    """Summary of a prior Strategist agent execution."""
    execution_id: str
    intent_summary: str
    recommended_strategy: str
    created_at: datetime


@dataclass(frozen=True)
class RelatedIssue:
    """Similar issue found via semantic similarity."""
    id: str
    identifier: str
    title: str
    state: Optional[str]
    similarity_score: float


@dataclass(frozen=True)
class KBSnippet:
    """Knowledge base entry retrieved via semantic search."""
    content: str
    source: str
    relevance_score: float


@dataclass(frozen=True)
class ClosedIssueSummary:
    """Summary of a closed issue for strategic context."""
    issue_id: str
    identifier: str
    title: str
    resolution_summary: str
    learnings: List[str]
    similarity_score: float


@dataclass(frozen=True)
class StrategistAgentContext:
    """
    Rich context for strategic reasoning.
    
    Assembled by the system BEFORE agent invocation.
    Agent treats this as read-only source of truth.
    
    Unlike the generic AgentContext, this includes:
    - Full discussion history
    - Prior agent outputs
    - Related issues from the same project
    - Retrieved KB snippets
    """
    
    # ─── Identity ───
    agent_name: str
    run_id: str
    
    # ─── Issue Snapshot (from cortex.issues) ───
    issue_id: str
    issue_identifier: str
    issue_title: str
    issue_description: Optional[str]
    issue_state: Optional[str]
    issue_priority: Optional[int]
    issue_labels: List[str]
    
    # ─── Trigger ───
    trigger_type: str  # "issue" | "comment"
    trigger_body: Optional[str]
    
    # ─── Project Scope ───
    project_id: Optional[str]
    project_name: Optional[str]
    
    # ─── Discussion History (from cortex.issue_comments) ───
    recent_comments: List[CommentSnapshot] = field(default_factory=list)
    
    # ─── Prior Agent Outputs (from cortex.agent_executions) ───
    prior_research: List[ResearchOutput] = field(default_factory=list)
    prior_strategies: List[StrategyOutput] = field(default_factory=list)
    
    # ─── Related Issues (from pgvector similarity) ───
    related_issues: List[RelatedIssue] = field(default_factory=list)
    
    # ─── Retrieved Knowledge (from Vector DB - future) ───
    kb_snippets: List[KBSnippet] = field(default_factory=list)
    
    # ─── Closed Issue Insights (from cortex.closed_issue_summaries) ───
    closed_issue_insights: List[ClosedIssueSummary] = field(default_factory=list)
    
    # ─── Execution Metadata ───
    user_id: str = ""
    access_token: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate trigger_type."""
        if self.trigger_type not in ("issue", "comment"):
            raise ValueError(f"trigger_type must be 'issue' or 'comment', got '{self.trigger_type}'")
    
    @property
    def primary_task(self) -> str:
        """Get the primary task (comment body if comment-triggered, else issue)."""
        if self.trigger_type == "comment" and self.trigger_body:
            return self.trigger_body
        parts = [self.issue_title]
        if self.issue_description:
            parts.append(self.issue_description)
        return "\n\n".join(parts)
    
    @property
    def is_comment_triggered(self) -> bool:
        """Check if triggered by a comment."""
        return self.trigger_type == "comment" and bool(self.trigger_body)
    
    @property
    def has_prior_research(self) -> bool:
        """Check if there's prior research to reference."""
        return len(self.prior_research) > 0
    
    @property
    def has_related_issues(self) -> bool:
        """Check if there are similar past issues."""
        return len(self.related_issues) > 0
