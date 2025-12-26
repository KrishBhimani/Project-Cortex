"""
Agent Registry - Maps agent names to concrete agent classes.
"""

from typing import Dict, Type, List
from agents.base import BaseAgent


class AgentRegistry:
    """
    Registry mapping agent names to concrete agent classes.
    
    Usage:
        # Register an agent
        AgentRegistry.register("Researcher", ResearcherAgent)
        
        # Get an agent instance
        agent = AgentRegistry.get("Researcher")
        result = await agent.run(context)
    """
    
    _agents: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register an agent class under a name.
        
        Args:
            name: Agent name (must match Linear agent session name)
            agent_class: Class implementing BaseAgent
        """
        cls._agents[name] = agent_class
        print(f"Registered agent: {name}")
    
    @classmethod
    def get(cls, name: str) -> BaseAgent:
        """
        Get agent instance by name.
        
        Args:
            name: Registered agent name
            
        Returns:
            Instance of the agent class
            
        Raises:
            KeyError: If agent name is not registered
        """
        if name not in cls._agents:
            available = list(cls._agents.keys())
            raise KeyError(f"Unknown agent: '{name}'. Available agents: {available}")
        return cls._agents[name]()
    
    @classmethod
    def available_agents(cls) -> List[str]:
        """List all registered agent names."""
        return list(cls._agents.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an agent is registered."""
        return name in cls._agents
