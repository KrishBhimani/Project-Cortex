"""
Researcher Agent - Uses Agno tools to research URLs and text content.

Tools:
- YouTubeTools: For YouTube video transcripts and metadata
- TrafilaturaTools: For article/blog/documentation extraction
- TavilyTools: For web search
- PDFTools: For extracting text from PDF documents
"""

import os
import io
import tempfile
import requests
from typing import List
from agents.base import BaseAgent, AgentResult
from agents.registry import AgentRegistry
from core.context import AgentContext

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.youtube import YouTubeTools
from agno.tools.trafilatura import TrafilaturaTools
from agno.tools.tavily import TavilyTools
from agno.tools import tool

# Memory support - SQLite (local file-based storage)
from agno.db.sqlite import SqliteDb

from dotenv import load_dotenv
load_dotenv()
# PDF extraction
try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("WARNING: pypdf not installed. PDF extraction disabled. Install with: uv add pypdf")

# Module-level variable to store auth token for tool access
_current_access_token = None

def set_access_token(token: str):
    """Set the access token for authenticated file downloads."""
    global _current_access_token
    _current_access_token = token


@tool(name="extract_pdf_content", description="Extract text content from a PDF document URL. Use this for any URL ending in .pdf or containing 'uploads.linear.app'. Returns the full text content of the PDF.")
def extract_pdf_tool(url: str) -> str:
    """
    Download and extract text from a PDF URL.
    Uses the current access token for authenticated downloads (Linear uploads).
    
    Args:
        url: The URL of the PDF document to extract text from.
        
    Returns:
        The extracted text content from the PDF.
    """
    global _current_access_token
    
    if not PDF_AVAILABLE:
        return "Error: pypdf is not installed. Cannot extract PDF content."
    
    try:
        # Prepare headers with auth if token provided
        headers = {}
        if _current_access_token and "uploads.linear.app" in url:
            headers["Authorization"] = f"Bearer {_current_access_token}"
            print(f"Using OAuth token for Linear upload download")
        
        # Download the PDF
        print(f"Downloading PDF: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Read PDF from bytes
        pdf_file = io.BytesIO(response.content)
        reader = pypdf.PdfReader(pdf_file)
        
        # Extract text from all pages
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {i+1} ---\n{page_text}")
        
        if not text_parts:
            return "Error: Could not extract any text from the PDF. It may be image-based or encrypted."
        
        full_text = "\n\n".join(text_parts)
        print(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
        return full_text
        
    except requests.RequestException as e:
        return f"Error downloading PDF: {str(e)}"
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"


RESEARCHER_INSTRUCTIONS = """
You are a Research Agent specialized in extracting and analyzing content from various sources.

## Your Task
Research the issue based on its title, description, and any URLs provided. Gather relevant information and provide a comprehensive summary.

## Tool Selection Rules

### For PDF Documents (URLs ending in .pdf or from uploads.linear.app):
- Use `extract_pdf_content` tool to extract the text from the PDF
- Summarize the key points and main content

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
- Process ALL URLs mentioned in the issue description
"""


class ResearcherAgent(BaseAgent):
    """
    Agent for research tasks using web scraping and content extraction tools.
    
    Features:
    - PDF document extraction
    - YouTube transcript extraction
    - Web page content extraction (Trafilatura)
    - Web search (Tavily)
    - Agentic memory with SQLite storage
    """
    
    def __init__(self):
        self.tools = self._build_tools()
        self.db = None  # Built at runtime with project_id
        self.agent = None  # Created at runtime with project-scoped db
    
    def _build_tools(self) -> list:
        """Build the list of tools."""
        tools = [
            extract_pdf_tool,     # For PDF extraction (uses module-level auth token)
            YouTubeTools(),       # For YouTube video transcripts
            TrafilaturaTools(),   # For web page content extraction  
            TavilyTools(api_key=os.getenv("TAVILY_API_KEY")),  # For web search
        ]
        print("ResearcherAgent: Tools loaded - PDF, YouTube, Trafilatura, Tavily")
        return tools
    
    def _build_db(self, project_id: str = None) -> SqliteDb:
        """Build SQLite database for memory with project-scoped table names."""
        suffix = f"_{project_id}" if project_id else ""
        db = SqliteDb(
            db_file="data/cortex_memory.db",
            session_table=f"researcher_sessions{suffix}",
            memory_table=f"researcher_memories{suffix}",
        )
        print(f"ResearcherAgent: Memory enabled with SQLite (project: {project_id or 'default'})")
        return db
    
    def _create_agent(self, db: SqliteDb) -> Agent:
        """Create the Agno agent with all tools and memory."""
        return Agent(
            name="Researcher",
            model=OpenAIChat(id="gpt-4.1-nano"),
            tools=self.tools,
            instructions=RESEARCHER_INSTRUCTIONS,
            db=db,
            enable_agentic_memory=True,
            num_history_messages=5,
            markdown=True,
            read_chat_history=True,
            add_history_to_context=True,
            # debug_mode=True
        )
    
    def _build_research_prompt(self, context: AgentContext) -> str:
        """Build the research prompt from AgentContext."""
        parts = []
        
        if context.is_comment_triggered:
            # Comment is the PRIMARY task - issue is context
            parts.append("## Your Task (from comment):")
            parts.append(f"{context.trigger_body}")
            parts.append("")
            parts.append("## Background Context (Issue):")
            parts.append(f"**{context.issue_identifier} - {context.issue_title}**")
            if context.issue_description:
                parts.append(f"\n{context.issue_description}")
        else:
            # Issue is the PRIMARY task
            parts.append(f"## Issue: {context.issue_identifier} - {context.issue_title}")
            if context.issue_description:
                parts.append(f"\n### Description:\n{context.issue_description}")
        
        # List all URLs for agent to process with appropriate tools
        if context.urls:
            parts.append(f"\n### URLs to Process ({len(context.urls)}):")
            for url in context.urls:
                url_lower = url.lower()
                if "youtube.com" in url_lower or "youtu.be" in url_lower:
                    parts.append(f"- [YouTube] {url}")
                elif url_lower.endswith('.pdf') or 'uploads.linear.app' in url_lower:
                    parts.append(f"- [PDF] {url}")
                else:
                    parts.append(f"- [Website] {url}")
        
        # Project context if available
        if context.project_name:
            parts.append(f"\n### Project Context: {context.project_name}")
        
        # Labels for additional context
        if context.issue_labels:
            parts.append(f"\n### Labels: {', '.join(context.issue_labels)}")
        
        parts.append("\n---\n")
        if context.is_comment_triggered:
            parts.append("Focus on completing the task described in the comment above. Use the issue context for background understanding.")
        else:
            parts.append("Please research and summarize the above content using the appropriate tools for each URL type.")
        
        return "\n".join(parts)
    
    async def run(self, context: AgentContext) -> AgentResult:
        """
        Execute research on the given context.
        
        Handles:
        - Multiple URLs (YouTube, articles, PDFs)
        - Text content from issue description
        - Comment triggers for specific research requests
        """
        try:
            # Set the access token for authenticated file downloads (PDF tool uses this)
            set_access_token(context.access_token)
            
            # Build project-scoped database and agent at runtime
            project_id = getattr(context, 'project_id', None)
            self.db = self._build_db(project_id)
            self.agent = self._create_agent(self.db)
            
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
