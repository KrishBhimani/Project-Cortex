"""
Base agent interfaces and result types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.context import AgentContext


@dataclass
class AgentResult:
    """Result of agent execution."""
    success: bool
    response: str
    status: str = "completed"
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base interface for all agents.
    
    Agents must:
    - Only perform reasoning (no side effects)
    - Not access Linear APIs, databases, or webhooks directly
    - Return an AgentResult from the run() method
    """
    
    @abstractmethod
    async def run(self, context: "AgentContext") -> AgentResult:
        """
        Execute agent reasoning on the given context.
        
        Args:
            context: Read-only AgentContext with all required information
            
        Returns:
            AgentResult with success status and response
        """
        pass
