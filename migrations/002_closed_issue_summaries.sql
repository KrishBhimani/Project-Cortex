-- ═══════════════════════════════════════════════════════════════════════════
-- Closed Issue Summaries - Stores LLM-generated summaries for Strategist context
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE cortex.closed_issue_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    issue_id UUID NOT NULL REFERENCES cortex.issues(id) ON DELETE CASCADE,
    project_id UUID,
    
    -- Issue snapshot
    identifier TEXT NOT NULL,
    title TEXT NOT NULL,
    original_description TEXT,
    
    -- LLM-generated content
    resolution_summary TEXT NOT NULL,           -- What was accomplished
    learnings TEXT[],                           -- Key takeaways
    tags TEXT[],                                -- Labels, keywords for search
    
    -- Embedding for similarity search (matches issue embeddings)
    summary_embedding vector(1536),
    
    -- Timestamps
    closed_at TIMESTAMPTZ NOT NULL,
    summarized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(issue_id)
);

-- Indexes
CREATE INDEX idx_closed_summaries_project ON cortex.closed_issue_summaries(project_id);
CREATE INDEX idx_closed_summaries_closed ON cortex.closed_issue_summaries(closed_at DESC);

-- IVFFlat index for fast similarity search (requires min 100 rows for optimal performance)
CREATE INDEX idx_closed_summaries_embedding ON cortex.closed_issue_summaries 
    USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 100);
