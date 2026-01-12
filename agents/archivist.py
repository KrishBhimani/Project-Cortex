"""
Archivist Agent - Curates and ingests project knowledge into Qdrant.

Responsibilities:
- Ingest documents (PDF, text) from issues via webhook trigger
- Summarize closed issues and store learnings
- Maintain per-project Qdrant collections

Triggered by:
- /webhook with agent_name="Archivist" (explicit knowledge ingestion)
- /sync_webhook on issue close (automatic summarization)
"""

import os
import io
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import json
import re
import requests

import psycopg2
from openai import OpenAI

from agents.base import BaseAgent, AgentResult
from agents.registry import AgentRegistry
from core.context import AgentContext

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from dotenv import load_dotenv
load_dotenv()


# PDF extraction
try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("WARNING: pypdf not installed. PDF extraction disabled.")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DB_URL = os.getenv("DB_URL")


def get_db_connection():
    """Get database connection."""
    if not DB_URL:
        raise Exception("DB_URL environment variable is required")
    return psycopg2.connect(DB_URL)


# OpenAI client for embeddings
_openai_client = None

def get_openai_client():
    """Get OpenAI client singleton."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


async def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for text using OpenAI ada-002."""
    if not text or not text.strip():
        return None
    
    try:
        def _generate():
            client = get_openai_client()
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text[:8000]
            )
            return response.data[0].embedding
        
        return await asyncio.to_thread(_generate)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_pdf_content(url: str, access_token: Optional[str] = None) -> Optional[str]:
    """
    Download and extract text from a PDF URL.
    """
    if not PDF_AVAILABLE:
        print("pypdf not available, cannot extract PDF")
        return None
    
    try:
        headers = {}
        if access_token and "uploads.linear.app" in url:
            headers["Authorization"] = f"Bearer {access_token}"
        
        print(f"Downloading PDF: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        reader = pypdf.PdfReader(pdf_file)
        
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        if not text_parts:
            return None
        
        full_text = "\n\n".join(text_parts)
        print(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
        return full_text
        
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None


def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text."""
    if not text:
        return []
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


# ═══════════════════════════════════════════════════════════════════════════
# QDRANT KNOWLEDGE MANAGER (using Agno's native Knowledge API)
# ═══════════════════════════════════════════════════════════════════════════

from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.db.sqlite import SqliteDb


class QdrantKnowledgeManager:
    """
    Manages per-project Qdrant collections using Agno's native Knowledge API.
    Handles chunking, embedding, and storage automatically.
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.collection_name = self._sanitize_collection_name(project_id)
        
        # Set up Qdrant vector database
        vector_db = Qdrant(
            collection=self.collection_name,
            url=QDRANT_URL,
        )
        
        # Set up SQLite for content metadata storage
        contents_db = SqliteDb(
            db_file="data/cortex_knowledge.db",
            knowledge_table=f"knowledge_{self.collection_name}",
        )
        
        # Create Knowledge instance with Agno's native API
        self.knowledge = Knowledge(
            name=f"Project {project_id} Knowledge",
            description=f"Knowledge base for project {project_id}",
            vector_db=vector_db,
            contents_db=contents_db,
        )
        
        print(f"QdrantKnowledgeManager initialized for project: {project_id}")
        print(f"Collection: {self.collection_name}, URL: {QDRANT_URL}")
    
    def _sanitize_collection_name(self, project_id: str) -> str:
        """Sanitize project_id for use as Qdrant collection name."""
        sanitized = project_id.replace("-", "_").lower()
        return f"project_{sanitized}"
    
    async def ingest_text(
        self,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Ingest plain text content using Agno's native Knowledge.add_content().
        Handles chunking and embedding automatically.
        """
        if not content or not content.strip():
            print("Empty content, skipping ingestion")
            return False
        
        try:
            def _add():
                # Use Agno's native text_content parameter
                self.knowledge.add_content(
                    text_content=f"Source: {source}\n\n{content}",
                )
            
            await asyncio.to_thread(_add)
            print(f"Ingested text from source: {source}")
            return True
            
        except Exception as e:
            print(f"Error ingesting text: {e}")
            return False
    
    async def ingest_pdf_url(
        self,
        url: str,
        source: str,
        access_token: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Ingest PDF from URL using Agno's native Knowledge.add_content().
        For Linear uploads (requiring auth), we download first then ingest text.
        """
        try:
            # For Linear uploads, we need to download first since auth is required
            if "uploads.linear.app" in url and access_token:
                content = extract_pdf_content(url, access_token)
                if content:
                    return await self.ingest_text(content, source, metadata)
                return False
            
            # For public URLs, use Agno's native URL ingestion
            def _add():
                self.knowledge.add_content(url=url)
            
            await asyncio.to_thread(_add)
            print(f"Ingested PDF from URL: {url}")
            return True
            
        except Exception as e:
            print(f"Error ingesting PDF: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# CLOSED ISSUE SUMMARIZER
# ═══════════════════════════════════════════════════════════════════════════

SUMMARIZER_INSTRUCTIONS = """
You are a Knowledge Curator for a software development team.
Your task is to summarize a closed issue into reusable knowledge.

## Input
You will receive:
- Issue title
- Issue description
- Issue state (Done/Completed)
- Any labels/tags

## Output Format
Return ONLY valid JSON with this exact structure:
{
  "resolution_summary": "2-3 sentence summary of what was accomplished and how",
  "learnings": ["Key takeaway 1", "Key takeaway 2", "Key takeaway 3"],
  "tags": ["tag1", "tag2", "tag3"]
}

## Rules
- Focus on WHAT was done and HOW it was solved
- Extract reusable patterns, decisions, and approaches
- Tags should be searchable keywords for finding this issue later
- Keep summaries concise but informative
- Return ONLY JSON, no markdown code blocks
"""


class ClosedIssueSummarizer:
    """
    Generates LLM-powered summaries for closed issues.
    Stores results in cortex.closed_issue_summaries table.
    """
    
    def __init__(self):
        self.agent = Agent(
            name="IssueSummarizer",
            model=OpenAIChat(id="gpt-4.1-nano"),
            instructions=SUMMARIZER_INSTRUCTIONS,
            markdown=False,
        )
    
    async def summarize_and_store(
        self,
        issue_id: str,
        identifier: str,
        title: str,
        description: Optional[str],
        project_id: Optional[str],
        labels: Optional[List[str]],
        closed_at: datetime,
    ) -> bool:
        """
        Generate summary for closed issue and store in database.
        """
        try:
            # Build prompt for summarization
            prompt_parts = [
                f"## Issue: {identifier} - {title}",
            ]
            if description:
                prompt_parts.append(f"\n### Description:\n{description}")
            if labels:
                prompt_parts.append(f"\n### Labels: {', '.join(labels)}")
            prompt_parts.append("\n---\nSummarize this closed issue.")
            
            prompt = "\n".join(prompt_parts)
            
            # Run summarization
            print(f"Summarizing closed issue: {identifier}")
            response = await asyncio.to_thread(
                lambda: self.agent.run(prompt)
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            summary_data = self._parse_json(response_text)
            
            # Generate embedding for the summary
            embedding_text = f"{title}\n\n{summary_data.get('resolution_summary', '')}"
            embedding = await generate_embedding(embedding_text)
            
            # Store in database
            await self._store_summary(
                issue_id=issue_id,
                identifier=identifier,
                title=title,
                description=description,
                project_id=project_id,
                resolution_summary=summary_data.get('resolution_summary', ''),
                learnings=summary_data.get('learnings', []),
                tags=summary_data.get('tags', []),
                embedding=embedding,
                closed_at=closed_at,
            )
            
            print(f"Stored summary for closed issue: {identifier}")
            return True
            
        except Exception as e:
            print(f"Error summarizing issue: {e}")
            return False
    
    def _parse_json(self, text: str) -> dict:
        """Parse JSON from response."""
        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "resolution_summary": text[:500],
                "learnings": [],
                "tags": [],
            }
    
    async def _store_summary(
        self,
        issue_id: str,
        identifier: str,
        title: str,
        description: Optional[str],
        project_id: Optional[str],
        resolution_summary: str,
        learnings: List[str],
        tags: List[str],
        embedding: Optional[List[float]],
        closed_at: datetime,
    ) -> None:
        """Store summary in cortex.closed_issue_summaries."""
        
        def _upsert():
            conn = get_db_connection()
            try:
                cur = conn.cursor()
                
                # Check if summary already exists
                cur.execute(
                    "SELECT id FROM cortex.closed_issue_summaries WHERE issue_id = %s",
                    (issue_id,)
                )
                exists = cur.fetchone() is not None
                
                if exists:
                    # Update existing summary
                    if embedding:
                        embedding_str = f"[{','.join(map(str, embedding))}]"
                        cur.execute("""
                            UPDATE cortex.closed_issue_summaries SET
                                identifier = %s,
                                title = %s,
                                original_description = %s,
                                resolution_summary = %s,
                                learnings = %s,
                                tags = %s,
                                summary_embedding = %s::vector,
                                summarized_at = NOW()
                            WHERE issue_id = %s
                        """, (
                            identifier, title, description, resolution_summary,
                            learnings, tags, embedding_str, issue_id
                        ))
                    else:
                        cur.execute("""
                            UPDATE cortex.closed_issue_summaries SET
                                identifier = %s,
                                title = %s,
                                original_description = %s,
                                resolution_summary = %s,
                                learnings = %s,
                                tags = %s,
                                summarized_at = NOW()
                            WHERE issue_id = %s
                        """, (
                            identifier, title, description, resolution_summary,
                            learnings, tags, issue_id
                        ))
                else:
                    # Insert new summary
                    if embedding:
                        embedding_str = f"[{','.join(map(str, embedding))}]"
                        cur.execute("""
                            INSERT INTO cortex.closed_issue_summaries (
                                issue_id, project_id, identifier, title,
                                original_description, resolution_summary,
                                learnings, tags, summary_embedding, closed_at
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s
                            )
                        """, (
                            issue_id, project_id, identifier, title,
                            description, resolution_summary,
                            learnings, tags, embedding_str, closed_at
                        ))
                    else:
                        cur.execute("""
                            INSERT INTO cortex.closed_issue_summaries (
                                issue_id, project_id, identifier, title,
                                original_description, resolution_summary,
                                learnings, tags, closed_at
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s
                            )
                        """, (
                            issue_id, project_id, identifier, title,
                            description, resolution_summary,
                            learnings, tags, closed_at
                        ))
                
                conn.commit()
                cur.close()
            finally:
                conn.close()
        
        await asyncio.to_thread(_upsert)


# ═══════════════════════════════════════════════════════════════════════════
# ARCHIVIST AGENT
# ═══════════════════════════════════════════════════════════════════════════

class ArchivistAgent(BaseAgent):
    """
    Agent for curating and ingesting project knowledge.
    
    Modes:
    1. INGEST: Triggered by webhook, ingests documents from issue
    2. SUMMARIZE: Triggered by issue close, creates closed issue summary
    """
    
    def __init__(self):
        self.summarizer = ClosedIssueSummarizer()
    
    async def run(self, context: AgentContext) -> AgentResult:
        """
        Execute knowledge curation based on context.
        """
        try:
            print(f"\n=== ARCHIVIST AGENT EXECUTING ===")
            print(f"Issue: {context.issue_identifier} - {context.issue_title}")
            print(f"Trigger: {context.trigger_type}")
            print(f"Project: {context.project_id}")
            
            start_time = datetime.now(timezone.utc)
            
            # Determine mode based on issue state
            is_closed = context.issue_state and context.issue_state.lower() in [
                "done", "completed", "closed"
            ]
            
            ingested_count = 0
            summarized = False
            
            # Mode 1: Ingest documents (always runs if there's content)
            if context.project_id:
                kb_manager = QdrantKnowledgeManager(context.project_id)
                
                # Ingest issue content itself
                issue_content = f"# {context.issue_title}\n\n{context.issue_description or ''}"
                if issue_content.strip():
                    success = await kb_manager.ingest_text(
                        content=issue_content,
                        source=f"issue:{context.issue_identifier}",
                        metadata={
                            "issue_id": context.issue_id,
                            "identifier": context.issue_identifier,
                            "type": "issue",
                        }
                    )
                    if success:
                        ingested_count += 1
                
                # Extract and ingest URLs
                all_text = f"{context.issue_title} {context.issue_description or ''}"
                if context.trigger_body:
                    all_text += f" {context.trigger_body}"
                
                urls = extract_urls_from_text(all_text)
                
                for url in urls:
                    url_lower = url.lower()
                    metadata = {
                        "issue_id": context.issue_id,
                        "source_url": url,
                        "type": "document",
                    }
                    
                    if url_lower.endswith('.pdf') or 'uploads.linear.app' in url_lower:
                        success = await kb_manager.ingest_pdf_url(
                            url=url,
                            source=f"pdf:{context.issue_identifier}",
                            access_token=context.access_token,
                            metadata=metadata,
                        )
                        if success:
                            ingested_count += 1
            
            # Mode 2: Summarize closed issue
            if is_closed:
                summarized = await self.summarizer.summarize_and_store(
                    issue_id=context.issue_id,
                    identifier=context.issue_identifier,
                    title=context.issue_title,
                    description=context.issue_description,
                    project_id=context.project_id,
                    labels=context.issue_labels,
                    closed_at=datetime.now(timezone.utc),
                )
            
            end_time = datetime.now(timezone.utc)
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Build response
            response_parts = []
            if ingested_count > 0:
                response_parts.append(f"📚 Ingested {ingested_count} document(s) to project knowledge base")
            if summarized:
                response_parts.append("✅ Created closed issue summary for future reference")
            if not response_parts:
                response_parts.append("ℹ️ No content to archive from this issue")
            
            response = "\n".join(response_parts)
            
            print(f"=== ARCHIVIST AGENT COMPLETE ({execution_time_ms}ms) ===\n")
            
            return AgentResult(
                success=True,
                response=response,
                status="completed",
                metadata={
                    "agent": "Archivist",
                    "issue_id": context.issue_id,
                    "project_id": context.project_id,
                    "ingested_count": ingested_count,
                    "summarized": summarized,
                    "execution_time_ms": execution_time_ms,
                }
            )
            
        except Exception as e:
            print(f"ArchivistAgent error: {str(e)}")
            return AgentResult(
                success=False,
                response=f"Knowledge curation failed: {str(e)}",
                status="error",
                metadata={
                    "agent": "Archivist",
                    "error": str(e),
                }
            )


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE FUNCTIONS FOR WEBHOOK TRIGGERS
# ═══════════════════════════════════════════════════════════════════════════

async def trigger_archivist_for_closed_issue(
    issue_id: str,
    identifier: str,
    title: str,
    description: Optional[str],
    project_id: Optional[str],
    labels: Optional[List[str]],
) -> bool:
    """
    Convenience function to trigger closed issue summarization.
    Called from sync_webhook when an issue is closed.
    """
    summarizer = ClosedIssueSummarizer()
    return await summarizer.summarize_and_store(
        issue_id=issue_id,
        identifier=identifier,
        title=title,
        description=description,
        project_id=project_id,
        labels=labels,
        closed_at=datetime.now(timezone.utc),
    )


# Auto-register on import
AgentRegistry.register("Archivist", ArchivistAgent)
