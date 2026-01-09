-- ═══════════════════════════════════════════════════════════════════════════
-- Project Cortex - LLMOps Cost Tracking Migration
-- Adds token usage and cost tracking to agent_executions
-- ═══════════════════════════════════════════════════════════════════════════

-- Add LLMOps columns to agent_executions table
ALTER TABLE cortex.agent_executions 
ADD COLUMN IF NOT EXISTS tokens_input INTEGER,
ADD COLUMN IF NOT EXISTS tokens_output INTEGER,
ADD COLUMN IF NOT EXISTS cost_usd DECIMAL(10,6),
ADD COLUMN IF NOT EXISTS model_name VARCHAR(100);

-- Create index for cost analysis queries
CREATE INDEX IF NOT EXISTS idx_executions_cost 
ON cortex.agent_executions(agent_name, started_at DESC) 
WHERE cost_usd IS NOT NULL;

-- Add comment for documentation
COMMENT ON COLUMN cortex.agent_executions.tokens_input IS 'Input/prompt token count from LLM API';
COMMENT ON COLUMN cortex.agent_executions.tokens_output IS 'Output/completion token count from LLM API';
COMMENT ON COLUMN cortex.agent_executions.cost_usd IS 'Estimated cost in USD based on model pricing';
COMMENT ON COLUMN cortex.agent_executions.model_name IS 'Model used for this execution (e.g., gpt-4.1-nano)';
