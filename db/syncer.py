"""
Issue Syncer - Syncs Linear issues and comments to PostgreSQL.

Handles:
- Issue create/update events from Linear webhooks
- Comment events
- Embedding generation for similarity search
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, List
import json

import psycopg2
import psycopg2.extras
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


def get_db_connection():
    """Get database connection."""
    db_url = os.getenv('DB_URL')
    if not db_url:
        raise Exception("DB_URL environment variable is required")
    return psycopg2.connect(db_url)


# OpenAI client for embeddings
_openai_client = None

def get_openai_client():
    """Get OpenAI client singleton."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


async def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate embedding for text using OpenAI ada-002.
    
    Returns 1536-dimensional vector.
    """
    if not text or not text.strip():
        return None
    
    try:
        def _generate():
            client = get_openai_client()
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text[:8000]  # Truncate to model limit
            )
            return response.data[0].embedding
        
        return await asyncio.to_thread(_generate)
    
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


class IssueSyncer:
    """
    Syncs Linear issues and comments to cortex database.
    
    Called by /sync_webhook endpoint when Linear sends
    issue or comment events.
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
    
    async def sync_issue(
        self,
        issue_id: str,
        identifier: str,
        title: str,
        description: Optional[str],
        state: Optional[str],
        priority: Optional[int],
        project_id: Optional[str],
        project_name: Optional[str],
        team_id: str,
        team_name: Optional[str],
        labels: Optional[List[str]],
        assignee_id: Optional[str],
        creator_id: Optional[str],
        created_at: datetime,
        updated_at: datetime,
    ) -> bool:
        """
        Upsert issue to cortex.issues.
        
        Generates embedding from title + description for similarity search.
        Returns True if successful.
        """
        try:
            # Generate embedding for title + description
            embedding_text = f"{title}\n\n{description or ''}"
            embedding = await generate_embedding(embedding_text)
            
            def _upsert():
                self._ensure_connection()
                cur = self.conn.cursor()
                
                # Check if issue exists
                cur.execute(
                    "SELECT id FROM cortex.issues WHERE id = %s",
                    (issue_id,)
                )
                exists = cur.fetchone() is not None
                
                if exists:
                    # Update existing issue
                    if embedding:
                        query = """
                        UPDATE cortex.issues SET
                            identifier = %s,
                            title = %s,
                            description = %s,
                            state = %s,
                            priority = %s,
                            project_id = %s,
                            project_name = %s,
                            team_id = %s,
                            team_name = %s,
                            labels = %s,
                            assignee_id = %s,
                            creator_id = %s,
                            updated_at = %s,
                            synced_at = NOW(),
                            title_desc_embedding = %s::vector
                        WHERE id = %s
                        """
                        embedding_str = f"[{','.join(map(str, embedding))}]"
                        cur.execute(query, (
                            identifier, title, description, state, priority,
                            project_id, project_name, team_id, team_name,
                            labels, assignee_id, creator_id, updated_at,
                            embedding_str, issue_id
                        ))
                    else:
                        query = """
                        UPDATE cortex.issues SET
                            identifier = %s,
                            title = %s,
                            description = %s,
                            state = %s,
                            priority = %s,
                            project_id = %s,
                            project_name = %s,
                            team_id = %s,
                            team_name = %s,
                            labels = %s,
                            assignee_id = %s,
                            creator_id = %s,
                            updated_at = %s,
                            synced_at = NOW()
                        WHERE id = %s
                        """
                        cur.execute(query, (
                            identifier, title, description, state, priority,
                            project_id, project_name, team_id, team_name,
                            labels, assignee_id, creator_id, updated_at,
                            issue_id
                        ))
                    print(f"Updated issue: {identifier}")
                else:
                    # Insert new issue
                    if embedding:
                        query = """
                        INSERT INTO cortex.issues (
                            id, identifier, title, description, state, priority,
                            project_id, project_name, team_id, team_name,
                            labels, assignee_id, creator_id,
                            created_at, updated_at, synced_at, title_desc_embedding
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, NOW(), %s::vector
                        )
                        """
                        embedding_str = f"[{','.join(map(str, embedding))}]"
                        cur.execute(query, (
                            issue_id, identifier, title, description, state, priority,
                            project_id, project_name, team_id, team_name,
                            labels, assignee_id, creator_id,
                            created_at, updated_at, embedding_str
                        ))
                    else:
                        query = """
                        INSERT INTO cortex.issues (
                            id, identifier, title, description, state, priority,
                            project_id, project_name, team_id, team_name,
                            labels, assignee_id, creator_id,
                            created_at, updated_at, synced_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, NOW()
                        )
                        """
                        cur.execute(query, (
                            issue_id, identifier, title, description, state, priority,
                            project_id, project_name, team_id, team_name,
                            labels, assignee_id, creator_id,
                            created_at, updated_at
                        ))
                    print(f"Inserted issue: {identifier}")
                
                self.conn.commit()
                cur.close()
                return True
            
            result = await asyncio.to_thread(_upsert)
            return result
            
        except Exception as e:
            print(f"Error syncing issue: {e}")
            return False
        finally:
            self._close_connection()
    
    async def sync_comment(
        self,
        comment_id: str,
        issue_id: str,
        body: str,
        author_id: Optional[str],
        author_name: Optional[str],
        created_at: datetime,
        is_agent_response: bool = False,
        agent_name: Optional[str] = None,
    ) -> bool:
        """
        Upsert comment to cortex.issue_comments.
        
        Returns True if successful.
        """
        try:
            def _upsert():
                self._ensure_connection()
                cur = self.conn.cursor()
                
                # Check if comment exists
                cur.execute(
                    "SELECT id FROM cortex.issue_comments WHERE id = %s",
                    (comment_id,)
                )
                exists = cur.fetchone() is not None
                
                if exists:
                    # Update existing comment
                    query = """
                    UPDATE cortex.issue_comments SET
                        body = %s,
                        author_id = %s,
                        author_name = %s,
                        is_agent_response = %s,
                        agent_name = %s,
                        synced_at = NOW()
                    WHERE id = %s
                    """
                    cur.execute(query, (
                        body, author_id, author_name,
                        is_agent_response, agent_name, comment_id
                    ))
                    print(f"Updated comment: {comment_id[:8]}...")
                else:
                    # Insert new comment
                    query = """
                    INSERT INTO cortex.issue_comments (
                        id, issue_id, body, author_id, author_name,
                        is_agent_response, agent_name, created_at, synced_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, NOW()
                    )
                    """
                    cur.execute(query, (
                        comment_id, issue_id, body, author_id, author_name,
                        is_agent_response, agent_name, created_at
                    ))
                    print(f"Inserted comment: {comment_id[:8]}...")
                
                self.conn.commit()
                cur.close()
                return True
            
            result = await asyncio.to_thread(_upsert)
            return result
            
        except Exception as e:
            print(f"Error syncing comment: {e}")
            return False
        finally:
            self._close_connection()


# Singleton instance
issue_syncer = IssueSyncer()
