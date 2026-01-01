-- ═══════════════════════════════════════════════════════════════════════════
-- Project Cortex - Database Schema Migration
-- Creates tables for issues, comments, and agent executions
-- ═══════════════════════════════════════════════════════════════════════════

-- Schema creation
CREATE SCHEMA IF NOT EXISTS cortex;

-- Enable pgvector extension for similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- ═══════════════════════════════════════════════════════════════════════════
-- ISSUES TABLE
-- Stores canonical snapshot of Linear issues (synced, not live-queried)
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE cortex.issues (
    id UUID PRIMARY KEY,                          -- Linear issue UUID
    identifier TEXT NOT NULL,                     -- e.g., "COR-123"
    title TEXT NOT NULL,
    description TEXT,
    state TEXT,                                   -- "backlog", "in_progress", etc.
    priority INTEGER,                             -- 0-4 (urgent to no priority)
    
    -- Scope
    project_id UUID,                              -- Linear project UUID
    project_name TEXT,
    team_id UUID NOT NULL,
    team_name TEXT,
    
    -- Metadata
    labels TEXT[],                                -- Array of label names
    assignee_id UUID,
    creator_id UUID,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Embedding for similarity search (pgvector)
    title_desc_embedding vector(1536),            -- OpenAI ada-002 embedding
    
    -- Constraints
    UNIQUE(identifier)
);

CREATE INDEX idx_issues_project ON cortex.issues(project_id);
CREATE INDEX idx_issues_team ON cortex.issues(team_id);
CREATE INDEX idx_issues_updated ON cortex.issues(updated_at DESC);

-- IVFFlat index for fast similarity search
CREATE INDEX idx_issues_embedding ON cortex.issues 
    USING ivfflat (title_desc_embedding vector_cosine_ops) WITH (lists = 100);

-- ═══════════════════════════════════════════════════════════════════════════
-- ISSUE COMMENTS TABLE
-- Stores discussion context for reasoning
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE cortex.issue_comments (
    id UUID PRIMARY KEY,                          -- Linear comment UUID
    issue_id UUID NOT NULL REFERENCES cortex.issues(id) ON DELETE CASCADE,
    
    -- Content
    body TEXT NOT NULL,
    author_id UUID,
    author_name TEXT,
    
    -- Metadata
    is_agent_response BOOLEAN DEFAULT FALSE,      -- True if posted by an agent
    agent_name TEXT,                              -- Which agent posted (if applicable)
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(id)
);

CREATE INDEX idx_comments_issue ON cortex.issue_comments(issue_id);
CREATE INDEX idx_comments_created ON cortex.issue_comments(issue_id, created_at DESC);

-- ═══════════════════════════════════════════════════════════════════════════
-- AGENT EXECUTIONS TABLE
-- Stores what agents did, why, and their structured output
-- ═══════════════════════════════════════════════════════════════════════════
CREATE TABLE cortex.agent_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- What ran
    agent_name TEXT NOT NULL,                     -- "Strategist", "Researcher", etc.
    issue_id UUID NOT NULL REFERENCES cortex.issues(id) ON DELETE CASCADE,
    
    -- Why it ran
    trigger_type TEXT NOT NULL,                   -- "issue" or "comment"
    trigger_comment_id UUID REFERENCES cortex.issue_comments(id),
    trigger_body TEXT,                            -- The comment/issue text that triggered
    
    -- Inputs (snapshot of what agent received)
    input_context JSONB,                          -- Serialized AgentContext snapshot
    
    -- Outputs
    success BOOLEAN NOT NULL,
    status TEXT NOT NULL,                         -- "completed", "failed", "error"
    response_text TEXT,                           -- Raw response posted to Linear
    structured_output JSONB,                      -- Parsed JSON output (for Strategist)
    
    -- Metadata
    execution_time_ms INTEGER,
    error_message TEXT,
    
    -- Timestamps
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    
    -- Constraints
    CHECK (trigger_type IN ('issue', 'comment'))
);

CREATE INDEX idx_executions_issue ON cortex.agent_executions(issue_id);
CREATE INDEX idx_executions_agent ON cortex.agent_executions(agent_name, issue_id);
CREATE INDEX idx_executions_recent ON cortex.agent_executions(issue_id, started_at DESC);
