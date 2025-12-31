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
        
        When triggered by a comment, the comment is the primary task.
        When triggered by an issue, the issue details are primary.
        """
        if context.is_comment_triggered:
            # Comment is the primary task
            response = f"## Task (from comment):\n{context.trigger_body}\n\n"
            response += f"### Issue Context: {context.issue_identifier} - {context.issue_title}"
            if context.issue_description:
                response += f"\n\n{context.issue_description}"
        else:
            # Issue is the primary focus
            response = f"Strategic analysis for: {context.issue_title}"
            if context.issue_description:
                response += f"\n\n{context.issue_description}"
        
        if context.project_name:
            response += f"\n\nProject: {context.project_name}"
        
        if context.issue_labels:
            response += f"\n\nLabels: {', '.join(context.issue_labels)}"
        
        return AgentResult(
            success=True,
            response=response,
            status="completed",
            metadata={
                "agent": "Strategist",
                "issue_id": context.issue_id,
                "project": context.project_name,
                "triggered_by": context.trigger_type
            }
        )


# Auto-register on import
AgentRegistry.register("Strategist", StrategistAgent)
