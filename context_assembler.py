"""
Context Assembler - Fetches and assembles context for agents.

This module handles all database and retrieval operations BEFORE
agent invocation. Agents never query databases directly - they 
receive a complete, frozen context from this assembler.
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from dataclasses import asdict
import json

import psycopg2
import psycopg2.extras

from strategist_context import (
    StrategistAgentContext,
    CommentSnapshot,
    ResearchOutput,
    StrategyOutput,
    RelatedIssue,
    KBSnippet,
)


def get_db_connection():
    """Get database connection."""
    db_url = os.getenv('DB_URL')
    if not db_url:
        raise Exception("DB_URL environment variable is required")
    return psycopg2.connect(db_url)


class ContextAssembler:
    """
    Assembles rich context for agents by fetching from PostgreSQL.
    
    All retrieval is done BEFORE agent invocation.
    This ensures agents operate on frozen, deterministic context.
    """
    
    def __init__(self):
        self.conn = None
    
    def _ensure_connection(self):
        """Ensure we have a database connection."""
        if not self.conn or self.conn.closed:
            self.conn = get_db_connection()
    
    def _close_connection(self):
        """Close the database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            self.conn = None
    
    async def fetch_issue(self, issue_id: str) -> Optional[dict]:
        """
        Fetch issue details from cortex.issues.
        
        Returns None if issue not found (not yet synced).
        """
        def _fetch():
            self._ensure_connection()
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
            SELECT id, identifier, title, description, state, priority,
                   project_id, project_name, team_id, team_name,
                   labels, assignee_id, creator_id,
                   created_at, updated_at, synced_at, title_desc_embedding
            FROM cortex.issues
            WHERE id = %s
            """
            cur.execute(query, (issue_id,))
            result = cur.fetchone()
            cur.close()
            return dict(result) if result else None
        
        return await asyncio.to_thread(_fetch)
    
    async def fetch_recent_comments(
        self, 
        issue_id: str, 
        limit: int = 10
    ) -> List[CommentSnapshot]:
        """
        Fetch recent comments for an issue.
        
        Returns comments in chronological order (oldest first) for context.
        """
        def _fetch():
            self._ensure_connection()
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
            SELECT id, body, author_name, is_agent_response, created_at
            FROM cortex.issue_comments
            WHERE issue_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """
            cur.execute(query, (issue_id, limit))
            results = cur.fetchall()
            cur.close()
            
            # Reverse to get chronological order
            return [
                CommentSnapshot(
                    id=str(r['id']),
                    body=r['body'],
                    author_name=r['author_name'],
                    is_agent_response=r['is_agent_response'] or False,
                    created_at=r['created_at']
                )
                for r in reversed(results)
            ]
        
        return await asyncio.to_thread(_fetch)
    
    async def fetch_prior_research(
        self, 
        issue_id: str, 
        limit: int = 3
    ) -> List[ResearchOutput]:
        """
        Fetch recent Researcher agent outputs for this issue.
        """
        def _fetch():
            self._ensure_connection()
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
            SELECT id, response_text, structured_output, started_at
            FROM cortex.agent_executions
            WHERE issue_id = %s 
              AND agent_name = 'Researcher'
              AND success = TRUE
            ORDER BY started_at DESC
            LIMIT %s
            """
            cur.execute(query, (issue_id, limit))
            results = cur.fetchall()
            cur.close()
            
            outputs = []
            for r in results:
                # Extract sources from structured output if available
                sources = []
                if r['structured_output']:
                    so = r['structured_output']
                    if isinstance(so, str):
                        try:
                            so = json.loads(so)
                        except:
                            so = {}
                    sources = so.get('sources', [])
                
                outputs.append(ResearchOutput(
                    execution_id=str(r['id']),
                    summary=r['response_text'] or '',
                    sources=sources,
                    created_at=r['started_at']
                ))
            
            return outputs
        
        return await asyncio.to_thread(_fetch)
    
    async def fetch_prior_strategies(
        self, 
        issue_id: str, 
        limit: int = 2
    ) -> List[StrategyOutput]:
        """
        Fetch recent Strategist agent outputs for this issue.
        """
        def _fetch():
            self._ensure_connection()
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
            SELECT id, structured_output, started_at
            FROM cortex.agent_executions
            WHERE issue_id = %s 
              AND agent_name = 'Strategist'
              AND success = TRUE
            ORDER BY started_at DESC
            LIMIT %s
            """
            cur.execute(query, (issue_id, limit))
            results = cur.fetchall()
            cur.close()
            
            outputs = []
            for r in results:
                so = r['structured_output'] or {}
                if isinstance(so, str):
                    try:
                        so = json.loads(so)
                    except:
                        so = {}
                
                outputs.append(StrategyOutput(
                    execution_id=str(r['id']),
                    intent_summary=so.get('intent_summary', ''),
                    recommended_strategy=so.get('recommended_strategy', {}).get('approach', ''),
                    created_at=r['started_at']
                ))
            
            return outputs
        
        return await asyncio.to_thread(_fetch)
    
    async def fetch_related_issues(
        self, 
        issue_id: str,
        project_id: str,
        embedding: List[float],
        limit: int = 5
    ) -> List[RelatedIssue]:
        """
        Fetch similar issues from the same project using pgvector.
        
        Uses cosine distance for similarity scoring.
        """
        if not embedding or not project_id:
            return []
        
        def _fetch():
            self._ensure_connection()
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # pgvector cosine distance query
            query = """
            SELECT id, identifier, title, state,
                   1 - (title_desc_embedding <=> %s::vector) as similarity
            FROM cortex.issues
            WHERE project_id = %s 
              AND id != %s
              AND title_desc_embedding IS NOT NULL
            ORDER BY title_desc_embedding <=> %s::vector
            LIMIT %s
            """
            
            embedding_str = f"[{','.join(map(str, embedding))}]"
            cur.execute(query, (embedding_str, project_id, issue_id, embedding_str, limit))
            results = cur.fetchall()
            cur.close()
            
            return [
                RelatedIssue(
                    id=str(r['id']),
                    identifier=r['identifier'],
                    title=r['title'],
                    state=r['state'],
                    similarity_score=float(r['similarity']) if r['similarity'] else 0.0
                )
                for r in results
            ]
        
        return await asyncio.to_thread(_fetch)
    
    async def assemble_strategist_context(
        self,
        issue_id: str,
        run_id: str,
        trigger_type: str,
        trigger_body: Optional[str],
        user_id: str,
        access_token: Optional[str] = None,
        # Fallback values if issue not in DB yet
        fallback_issue_data: Optional[dict] = None,
    ) -> StrategistAgentContext:
        """
        Assemble complete StrategistAgentContext.
        
        Fetches all required data from PostgreSQL before returning
        a frozen, immutable context for the agent.
        """
        try:
            # 1. Fetch issue (or use fallback)
            issue = await self.fetch_issue(issue_id)
            if not issue and fallback_issue_data:
                issue = fallback_issue_data
            
            if not issue:
                raise ValueError(f"Issue {issue_id} not found and no fallback provided")
            
            # 2. Fetch recent comments
            recent_comments = await self.fetch_recent_comments(issue_id, limit=10)
            
            # 3. Fetch prior agent outputs
            prior_research = await self.fetch_prior_research(issue_id, limit=3)
            prior_strategies = await self.fetch_prior_strategies(issue_id, limit=2)
            
            # 4. Fetch related issues (if we have embedding)
            related_issues = []
            if issue.get('title_desc_embedding') and issue.get('project_id'):
                related_issues = await self.fetch_related_issues(
                    issue_id=issue_id,
                    project_id=str(issue['project_id']),
                    embedding=issue['title_desc_embedding'],
                    limit=5
                )
            
            # 5. KB snippets (future - placeholder)
            kb_snippets = []
            
            # 6. Assemble the context
            context = StrategistAgentContext(
                agent_name="Strategist",
                run_id=run_id,
                
                # Issue snapshot
                issue_id=str(issue['id']) if issue.get('id') else issue_id,
                issue_identifier=issue.get('identifier', ''),
                issue_title=issue.get('title', ''),
                issue_description=issue.get('description'),
                issue_state=issue.get('state'),
                issue_priority=issue.get('priority'),
                issue_labels=issue.get('labels', []) or [],
                
                # Trigger
                trigger_type=trigger_type,
                trigger_body=trigger_body,
                
                # Project scope
                project_id=str(issue['project_id']) if issue.get('project_id') else None,
                project_name=issue.get('project_name'),
                
                # Discussion history
                recent_comments=recent_comments,
                
                # Prior outputs
                prior_research=prior_research,
                prior_strategies=prior_strategies,
                
                # Related issues
                related_issues=related_issues,
                
                # KB (future)
                kb_snippets=kb_snippets,
                
                # Metadata
                user_id=user_id,
                access_token=access_token,
                created_at=datetime.now(timezone.utc),
            )
            
            return context
            
        finally:
            self._close_connection()
    
    async def save_execution(
        self,
        agent_name: str,
        issue_id: str,
        trigger_type: str,
        trigger_comment_id: Optional[str],
        trigger_body: Optional[str],
        input_context: dict,
        success: bool,
        status: str,
        response_text: str,
        structured_output: Optional[dict],
        execution_time_ms: int,
        error_message: Optional[str] = None,
    ) -> str:
        """
        Save agent execution record to cortex.agent_executions.
        
        Returns the execution ID.
        """
        def _save():
            self._ensure_connection()
            cur = self.conn.cursor()
            
            query = """
            INSERT INTO cortex.agent_executions (
                agent_name, issue_id, trigger_type, trigger_comment_id, trigger_body,
                input_context, success, status, response_text, structured_output,
                execution_time_ms, error_message, completed_at
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, NOW()
            )
            RETURNING id
            """
            
            cur.execute(query, (
                agent_name,
                issue_id,
                trigger_type,
                trigger_comment_id,
                trigger_body,
                json.dumps(input_context) if input_context else None,
                success,
                status,
                response_text,
                json.dumps(structured_output) if structured_output else None,
                execution_time_ms,
                error_message,
            ))
            
            execution_id = cur.fetchone()[0]
            self.conn.commit()
            cur.close()
            
            return str(execution_id)
        
        try:
            return await asyncio.to_thread(_save)
        finally:
            self._close_connection()


# Singleton instance for convenience
context_assembler = ContextAssembler()
