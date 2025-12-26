"""
Strategist Agent - Handles strategy and planning tasks.
"""

from agents.base import BaseAgent, AgentResult
from agents.registry import AgentRegistry
from agent_context import AgentContext


class StrategistAgent(BaseAgent):
    """
    Agent for strategy and planning tasks.
    
    Triggered when "Strategist" agent is assigned to an issue.
    """
    
    async def run(self, context: AgentContext) -> AgentResult:
        """
        Execute strategic analysis on the given context.
        
        TODO: Implement actual strategy logic using:
        - context.issue_title
        - context.issue_description
        - context.project_name
        - context.related_issues
        """
        # Placeholder implementation
        response = f"Strategic analysis for: {context.issue_title}"
        
        if context.project_name:
            response += f"\n\nProject: {context.project_name}"
        
        if context.issue_labels:
            response += f"\n\nLabels: {', '.join(context.issue_labels)}"
        
        if context.trigger_body:
            response += f"\n\nComment context: {context.trigger_body}"
        
        return AgentResult(
            success=True,
            response=response,
            status="completed",
            metadata={
                "agent": "Strategist",
                "issue_id": context.issue_id,
                "project": context.project_name
            }
        )


# Auto-register on import
AgentRegistry.register("Strategist", StrategistAgent)
