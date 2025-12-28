"""
Perplexity Agent - Uses MCP tools for AI-powered research and reasoning.

Tools (via MCP server-perplexity-ask):
- perplexity_ask: Fast web lookups and conversational QA
- perplexity_research: Multi-source investigations with citations
- perplexity_reason: Structured analysis and logical problem-solving
"""

import os
import asyncio
import logging
from typing import Optional

from agents.base import BaseAgent, AgentResult
from agents.registry import AgentRegistry
from agent_context import AgentContext

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from agno.db.sqlite import SqliteDb

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

PERPLEXITY_INSTRUCTIONS = """
You are an expert research and reasoning assistant with access to Perplexity tools.

## Available Tools
- `perplexity_ask` → Fast, live web lookups and conversational QA.
- `perplexity_research` → Thorough, multi-source investigations with citations.
- `perplexity_reason` → Structured analysis, trade-offs, and logical problem-solving.

## Tool Selection Guidelines

### Use `perplexity_ask` when:
- User wants quick facts, simple definitions, or real-time updates
- Query is conversational, short, or time-sensitive
- Answer likely comes from a small number of sources

### Use `perplexity_research` when:
- User requests a comprehensive overview, report, or deep dive
- Task requires multiple perspectives, comparisons, or trend analysis
- User explicitly asks for sources, citations, or academic references
- Topic is complex and demands careful evidence aggregation

### Use `perplexity_reason` when:
- User asks for decision frameworks, trade-off evaluations, or pros/cons
- Question is analytical, logical, or involves planning
- Task needs structured reasoning (assumptions, options, criteria, recommendations)

## Tie-Breakers
- Speed vs Depth: If user emphasizes quickness → `ask`. Rigor/documentation → `research`. Judgment/analysis → `reason`.
- Unclear intent: Start with `ask` to clarify. Escalate to `research` or `reason` if needed.
- Need both facts and analysis: Gather data with `ask`/`research`, then use `reason` for evaluation.

## Response Format
Provide clear, well-structured responses with:
1. **Direct Answer**: The main response to the query
2. **Sources**: Citations and references (when using research)
3. **Analysis**: Reasoning and trade-offs (when applicable)
4. **Recommendations**: Actionable next steps if relevant

## Important Rules
- Do not invent sources or overstate certainty
- Clearly note when information is unavailable, conflicting, or limited
- Always attribute information to its source
"""


class PerplexityAgent(BaseAgent):
    """
    Agent for AI-powered research using Perplexity MCP tools.
    
    Features:
    - Fast web lookups (perplexity_ask)
    - Deep research with citations (perplexity_research)
    - Logical reasoning and analysis (perplexity_reason)
    - Session memory with SQLite storage
    """
    
    def __init__(self):
        self.db = self._build_db()
        self.agent = None
        self.mcp_session = None
        self.stdio_client_context = None
        self.mcp_tools = None
        self.max_retries = 3
        self.retry_delay = 1
    
    def _build_db(self) -> SqliteDb:
        """Build SQLite database for session memory."""
        db = SqliteDb(
            db_file="cortex_memory.db",
            session_table="perplexity_sessions",
            memory_table="perplexity_memories",
        )
        logger.info("PerplexityAgent: Memory enabled with SQLite")
        return db
    
    def _get_mcp_config(self) -> dict:
        """Get MCP configuration for Perplexity server."""
        if not PERPLEXITY_API_KEY:
            raise ValueError("PERPLEXITY_API_KEY environment variable is required")
        
        return {
            "server": {
                "command": "npx",
                "args": ["-y", "server-perplexity-ask"],
                "env": {"PERPLEXITY_API_KEY": PERPLEXITY_API_KEY}
            }
        }
    
    async def _initialize_mcp(self, context: AgentContext) -> None:
        """Initialize MCP session and tools."""
        mcp_config = self._get_mcp_config()
        server_config = mcp_config["server"]
        
        logger.info("Initializing Perplexity MCP connection...")
        logger.info(f"Command: {server_config['command']}")
        logger.info(f"Args: {server_config['args']}")
        
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"],
            env=server_config.get("env", {})
        )
        
        try:
            # Create stdio client context
            self.stdio_client_context = stdio_client(server_params)
            read, write = await self.stdio_client_context.__aenter__()
            
            # Create MCP session
            self.mcp_session = ClientSession(read, write)
            await self.mcp_session.__aenter__()
            
            logger.info("✅ Perplexity MCP session initialized")
            
            # Initialize MCP tools
            self.mcp_tools = MCPTools(session=self.mcp_session)
            await self.mcp_tools.initialize()
            logger.info("✅ Perplexity MCP tools loaded")
            
            # Create the agent with MCP tools
            self.agent = Agent(
                name="Perplexity",
                model=OpenAIChat(id="gpt-4.1-nano"),
                tools=[self.mcp_tools],
                instructions=PERPLEXITY_INSTRUCTIONS,
                db=self.db,
                read_chat_history=True,
                add_history_to_context=True,
                markdown=True,
                num_history_runs=5
            )
            
        except Exception as e:
            logger.error(f"❌ MCP initialization failed: {str(e)}")
            await self._cleanup_mcp()
            raise e
    
    async def _cleanup_mcp(self) -> None:
        """Clean up MCP resources."""
        try:
            if self.mcp_tools:
                self.mcp_tools = None
            
            if self.mcp_session:
                await self.mcp_session.__aexit__(None, None, None)
                self.mcp_session = None
            
            if self.stdio_client_context:
                await self.stdio_client_context.__aexit__(None, None, None)
                self.stdio_client_context = None
                
        except Exception as e:
            logger.error(f"Error during MCP cleanup: {str(e)}")
    
    def _build_prompt(self, context: AgentContext) -> str:
        """Build the research prompt from AgentContext."""
        parts = []
        
        # Issue context
        parts.append(f"## Issue: {context.issue_identifier} - {context.issue_title}")
        
        if context.issue_description:
            parts.append(f"\n### Description:\n{context.issue_description}")
        
        # Trigger context (if comment)
        if context.trigger_type == "comment" and context.trigger_body:
            parts.append(f"\n### Request:\n{context.trigger_body}")
        
        # URLs if any
        if context.urls:
            parts.append(f"\n### URLs mentioned:")
            for url in context.urls:
                parts.append(f"- {url}")
        
        # Project context
        if context.project_name:
            parts.append(f"\n### Project: {context.project_name}")
        
        # Labels
        if context.issue_labels:
            parts.append(f"\n### Labels: {', '.join(context.issue_labels)}")
        
        parts.append("\n---\n")
        parts.append("Please research and respond to the above query using the appropriate Perplexity tool.")
        
        return "\n".join(parts)
    
    async def run(self, context: AgentContext) -> AgentResult:
        """
        Execute research using Perplexity MCP tools.
        
        Args:
            context: AgentContext with issue details and research request
            
        Returns:
            AgentResult with research findings
        """
        try:
            print(f"\n=== PERPLEXITY AGENT EXECUTING ===")
            print(f"Issue: {context.issue_identifier}")
            print(f"Trigger: {context.trigger_type}")
            
            # Initialize MCP connection
            await self._initialize_mcp(context)
            
            # Build prompt
            prompt = self._build_prompt(context)
            
            # Run the agent
            response = await self.agent.arun(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            print(f"=== PERPLEXITY AGENT COMPLETE ===\n")
            
            return AgentResult(
                success=True,
                response=response_text,
                status="completed",
                metadata={
                    "agent": "Perplexity",
                    "issue_id": context.issue_id,
                    "trigger_type": context.trigger_type,
                }
            )
            
        except Exception as e:
            logger.error(f"Perplexity agent error: {str(e)}")
            return AgentResult(
                success=False,
                response=f"Error during Perplexity research: {str(e)}",
                status="error",
                metadata={"error": str(e)}
            )
            
        finally:
            # Always cleanup MCP resources
            await self._cleanup_mcp()


# Register the agent
AgentRegistry.register("Perplexity", PerplexityAgent)
