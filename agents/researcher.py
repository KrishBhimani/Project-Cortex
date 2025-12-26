"""
Researcher Agent - Uses Agno tools to research URLs and text content.

Tools:
- YouTubeTools: For YouTube video transcripts and metadata
- TrafilaturaTools: For article/blog/documentation extraction
- TavilyTools: For web search
"""

import os
from typing import List
from agents.base import BaseAgent, AgentResult
from agents.registry import AgentRegistry
from agent_context import AgentContext

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.youtube import YouTubeTools
from agno.tools.trafilatura import TrafilaturaTools
from agno.tools.tavily import TavilyTools

# Memory support - SQLite (local file-based storage)
from agno.db.sqlite import SqliteDb

RESEARCHER_INSTRUCTIONS = """
You are a Research Agent specialized in extracting and analyzing content from various sources.

## Your Task
Given an issue or comment, research the provided URLs and/or text content to gather relevant information.

## Tool Selection Rules

### For YouTube URLs (youtube.com, youtu.be):
- Use YouTubeTools to get video transcripts and metadata
- Summarize key points with timestamps if available

### For Website URLs (articles, docs, company sites):
- Use TrafilaturaTools `extract_text` to get the main content
- If extraction fails, use TavilyTools `web_search` to find information about the topic

### For General Research:
- Use TavilyTools `web_search` to find additional context and information
- Search for company info, product details, news, etc.

## Response Format
Provide a structured research summary:
1. **Source Overview**: List all sources analyzed
2. **Key Findings**: Main insights from each source
3. **Summary**: Consolidated analysis relevant to the issue
4. **Recommendations**: Actionable next steps based on research

## Important Rules
- Always cite sources with URLs
- Extract facts, not opinions (unless analyzing sentiment)
- If a URL fails to extract, use search to find information instead
- Handle both text content AND URLs in the same request
"""


class ResearcherAgent(BaseAgent):
    """
    Agent for research tasks using web scraping and content extraction tools.
    
    Features:
    - YouTube transcript extraction
    - Web page content extraction (Trafilatura)
    - Web search (Tavily)
    - Agentic memory with SQLite storage
    """
    
    def __init__(self):
        self.tools = self._build_tools()
        self.db = self._build_db()
        self.agent = self._create_agent()
    
    def _build_tools(self) -> list:
        """Build the list of tools."""
        tools = [
            YouTubeTools(),       # For YouTube video transcripts
            TrafilaturaTools(),   # For web page content extraction  
            TavilyTools(enable_search_context=True),        # For web search (requires TAVILY_API_KEY)
        ]
        print("ResearcherAgent: Tools loaded - YouTube, Trafilatura, Tavily")
        return tools
    
    def _build_db(self) -> SqliteDb:
        """Build SQLite database for memory."""
        db = SqliteDb(
            db_file="cortex_memory.db",
            session_table="researcher_sessions",
            memory_table="researcher_memories",
        )
        print("ResearcherAgent: Memory enabled with SQLite (cortex_memory.db)")
        return db
    
    def _create_agent(self) -> Agent:
        """Create the Agno agent with all tools and memory."""
        return Agent(
            name="Researcher",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=self.tools,
            instructions=RESEARCHER_INSTRUCTIONS,
            db=self.db,
            enable_agentic_memory=True,
            num_history_messages=5,
            markdown=True,
            read_chat_history=True,
            add_history_to_context=True,
        )
    
    def _build_research_prompt(self, context: AgentContext) -> str:
        """Build the research prompt from AgentContext."""
        parts = []
        
        # Issue context
        parts.append(f"## Issue: {context.issue_identifier} - {context.issue_title}")
        
        if context.issue_description:
            parts.append(f"\n### Description:\n{context.issue_description}")
        
        # Trigger context (if comment)
        if context.trigger_type == "comment" and context.trigger_body:
            parts.append(f"\n### Comment (Research Request):\n{context.trigger_body}")
        
        # URLs to research
        if context.urls:
            youtube_urls = [u for u in context.urls if "youtube.com" in u.lower() or "youtu.be" in u.lower()]
            other_urls = [u for u in context.urls if u not in youtube_urls]
            
            parts.append(f"\n### URLs to Research ({len(context.urls)} total):")
            
            if youtube_urls:
                parts.append(f"\n**YouTube Videos ({len(youtube_urls)}):**")
                for url in youtube_urls:
                    parts.append(f"- {url}")
            
            if other_urls:
                parts.append(f"\n**Websites ({len(other_urls)}):**")
                for url in other_urls:
                    parts.append(f"- {url}")
        else:
            parts.append("\n### No URLs provided - analyze the text content above.")
        
        # Project context if available
        if context.project_name:
            parts.append(f"\n### Project Context: {context.project_name}")
        
        # Labels for additional context
        if context.issue_labels:
            parts.append(f"\n### Labels: {', '.join(context.issue_labels)}")
        
        parts.append("\n---\n")
        parts.append("Please research the above content and provide a comprehensive summary.")
        
        return "\n".join(parts)
    
    async def run(self, context: AgentContext) -> AgentResult:
        """
        Execute research on the given context.
        
        Handles:
        - Multiple URLs (YouTube, articles, protected sites)
        - Text content from issue description
        - Comment triggers for specific research requests
        """
        try:
            # Build the research prompt
            prompt = self._build_research_prompt(context)
            
            print(f"\n=== RESEARCHER AGENT EXECUTING ===")
            print(f"Issue: {context.issue_identifier}")
            print(f"URLs to process: {len(context.urls)}")
            print(f"Trigger: {context.trigger_type}")
            
            # Run the Agno agent
            response = self.agent.run(prompt)
            
            # Extract the response content
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            print(f"=== RESEARCHER AGENT COMPLETE ===\n")
            
            return AgentResult(
                success=True,
                response=response_text,
                status="completed",
                metadata={
                    "agent": "Researcher",
                    "issue_id": context.issue_id,
                    "urls_processed": len(context.urls),
                    "trigger_type": context.trigger_type,
                }
            )
            
        except Exception as e:
            error_msg = f"Research failed: {str(e)}"
            print(f"ResearcherAgent error: {error_msg}")
            
            return AgentResult(
                success=False,
                response=f"I encountered an error while researching: {str(e)}",
                status="failed",
                metadata={
                    "agent": "Researcher",
                    "error": str(e),
                }
            )


# Auto-register on import
AgentRegistry.register("Researcher", ResearcherAgent)
