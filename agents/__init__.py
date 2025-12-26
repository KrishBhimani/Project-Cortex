"""
Agents package - Exposes AgentRegistry and auto-registers all agents.

To add a new agent:
1. Create agents/your_agent.py
2. Define YourAgent(BaseAgent) with async def run()
3. Add AgentRegistry.register("YourAgent", YourAgent) at module end
4. Import below
"""

from agents.base import BaseAgent, AgentResult
from agents.registry import AgentRegistry

# Import agents to trigger auto-registration
from agents import researcher
from agents import strategist

__all__ = ["BaseAgent", "AgentResult", "AgentRegistry"]
