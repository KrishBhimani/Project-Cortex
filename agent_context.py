"""
AgentContext - Strongly typed, immutable execution context for agent invocation.

This module defines the AgentContext dataclass that represents a single execution
snapshot passed to agents. The context is read-only (frozen) to ensure agents
cannot mutate state during execution.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass(frozen=True)
class AgentContext:
    """
    Read-only execution snapshot for agent invocation.
    
    This context is assembled by the webhook handler and passed to agents.
    Agents must never directly access Linear APIs, databases, or webhooks -
    all required information should be provided through this context.
    """
    
    # === Identity ===
    agent_name: str
    run_id: str
    
    # === Issue Scope ===
    issue_id: str
    issue_identifier: str
    issue_title: str
    issue_description: Optional[str]
    issue_state: Optional[str]
    issue_labels: List[str]
    
    # === Trigger ===
    trigger_type: str  # "issue" | "comment"
    trigger_body: Optional[str]  # comment body if triggered by comment, else None
    
    # === Project Scope ===
    project_id: Optional[str]
    project_name: Optional[str]
    
    # === Retrieved Context (read-only) ===
    project_kb_snippets: List[str] = field(default_factory=list)
    related_issues: List[str] = field(default_factory=list)
    
    # === External References ===
    urls: List[str] = field(default_factory=list)
    
    # === Execution Metadata ===
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate trigger_type is one of the allowed values."""
        if self.trigger_type not in ("issue", "comment"):
            raise ValueError(f"trigger_type must be 'issue' or 'comment', got '{self.trigger_type}'")
