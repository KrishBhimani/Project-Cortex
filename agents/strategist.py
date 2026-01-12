"""
Strategist Agent - Project-aware reasoning agent for strategic planning.

Converts ambiguous issues into clear strategy and execution plans by:
- Understanding true intent
- Detecting gaps and ambiguities
- Incorporating past context and decisions
- Choosing overall strategy (not just steps)
- Producing structured, actionable output
"""

import json
import os
from datetime import datetime, timezone

from agents.base import BaseAgent, AgentResult
from agents.registry import AgentRegistry
from core.context import AgentContext
from core.strategist_context import StrategistAgentContext

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant

from dotenv import load_dotenv
load_dotenv()


STRATEGIST_INSTRUCTIONS = """
You are a **Strategic Planning Agent** for a software development team.

## Your Role
Convert ambiguous issues into clear strategy and actionable execution plans.
You reason, you don't execute. You plan, you don't implement.

## Reasoning Process
Follow these steps internally before producing output:

1. **UNDERSTAND TRUE INTENT**
   - What does the user actually want? (Not just what they wrote)
   - What problem are they trying to solve?

2. **DETECT GAPS & AMBIGUITIES**
   - What information is missing?
   - What assumptions must we make?
   - What needs clarification?

3. **INCORPORATE PAST CONTEXT**
   - What did the Researcher find?
   - What similar issues have we handled before?
   - What patterns exist in this project?

4. **CHOOSE STRATEGY** (not just steps)
   - Build vs buy?
   - Fast vs thorough?
   - Scope in vs out?
   - What's the overall approach?

5. **BREAK INTO EXECUTION PHASES**
   - Concrete, assignable tasks
   - Clear outputs for each phase
   - Logical dependencies

6. **IDENTIFY RISKS & TRADEOFFS**
   - What could go wrong?
   - What are we trading off?
   - What needs monitoring?

## Output Format
You MUST return valid JSON with this exact structure:

```json
{
  "intent_summary": "One-sentence summary of what we're actually trying to achieve",
  
  "recommended_strategy": {
    "approach": "The chosen strategic approach",
    "rationale": "Why this approach over alternatives"
  },
  
  "execution_plan": [
    {
      "phase": 1,
      "title": "Phase title",
      "tasks": ["Task 1", "Task 2"],
      "output": "What this phase produces",
      "estimated_effort": "Time estimate"
    }
  ],
  
  "assumptions": [
    "Assumption 1",
    "Assumption 2"
  ],
  
  "risks": [
    {
      "risk": "Description of risk",
      "mitigation": "How to mitigate"
    }
  ],
  
  "open_questions": [
    "Question that needs clarification"
  ]
}
```

## Important Rules
- Return ONLY valid JSON, no markdown code blocks
- Be concrete and specific, not abstract
- Tasks should be human-executable
- Reference prior research findings when available
- Consider related issues for patterns
- If triggered by a comment, focus on that specific request
"""


class StrategistAgent(BaseAgent):
    """
    Agent for strategic planning and execution plan creation.
    
    Produces structured JSON output for:
    - Debugging and traceability
    - Future automation (creating subtasks)
    - Memory curation
    """
    
    def __init__(self):
        self.db = None  # Built at runtime with project_id
        self.agent = None  # Created at runtime with project-scoped db
    
    def _build_db(self, project_id: str = None) -> SqliteDb:
        """Build SQLite database for session memory with project-scoped table names."""
        # Use project_id in table names for project isolation
        suffix = f"_{project_id}" if project_id else ""
        db = SqliteDb(
            db_file="data/cortex_memory.db",
            session_table=f"strategist_sessions{suffix}",
            memory_table=f"strategist_memories{suffix}",
        )
        print(f"StrategistAgent: Memory enabled with SQLite (project: {project_id or 'default'})")
        return db
    
    def _create_agent(self, db: SqliteDb, knowledge=None) -> Agent:
        """Create the Agno agent with the given database and optional knowledge base."""
        agent_config = {
            "name": "Strategist",
            "model": OpenAIChat(id="gpt-4.1-nano"),  # Using stronger model for strategy
            "instructions": STRATEGIST_INSTRUCTIONS,
            "db": db,
            "read_chat_history": True,
            "add_history_to_context": True,
            "markdown": False,  # We want JSON output
            "num_history_runs": 3,
        }
        
        # Add knowledge base if provided (project Qdrant collection)
        if knowledge:
            agent_config["knowledge"] = knowledge
            agent_config["search_knowledge"] = True
            print(f"StrategistAgent: Knowledge base attached")
        
        return Agent(**agent_config)
    
    def _build_prompt(self, context) -> str:
        """
        Build the prompt from context.
        
        Accepts either AgentContext (lightweight) or StrategistAgentContext (rich).
        """
        parts = []
        
        # Check if it's the rich context
        is_rich_context = hasattr(context, 'recent_comments')
        
        # Primary task
        if hasattr(context, 'is_comment_triggered') and context.is_comment_triggered:
            parts.append("## Your Task (from comment):")
            parts.append(context.trigger_body)
            parts.append("")
            parts.append("## Issue Context:")
            parts.append(f"**{context.issue_identifier} - {context.issue_title}**")
            if context.issue_description:
                parts.append(context.issue_description)
        else:
            parts.append(f"## Issue: {context.issue_identifier} - {context.issue_title}")
            if context.issue_description:
                parts.append(f"\n{context.issue_description}")
        
        # Project context
        if context.project_name:
            parts.append(f"\n### Project: {context.project_name}")
        
        # Labels
        if context.issue_labels:
            parts.append(f"\n### Labels: {', '.join(context.issue_labels)}")
        
        # Rich context only: Discussion history
        if is_rich_context and context.recent_comments:
            parts.append("\n### Recent Discussion:")
            for comment in context.recent_comments[-5:]:  # Last 5 comments
                author = comment.author_name or "Unknown"
                prefix = "[Agent] " if comment.is_agent_response else ""
                parts.append(f"- {prefix}{author}: {comment.body[:200]}...")
        
        # Rich context only: Prior research
        if is_rich_context and context.prior_research:
            parts.append("\n### Prior Research Findings:")
            for research in context.prior_research:
                parts.append(f"- {research.summary[:300]}...")
                if research.sources:
                    parts.append(f"  Sources: {', '.join(research.sources[:3])}")
        
        # Rich context only: Related issues
        if is_rich_context and context.related_issues:
            parts.append("\n### Similar Past Issues:")
            for issue in context.related_issues[:3]:
                state_str = f" [{issue.state}]" if issue.state else ""
                parts.append(f"- {issue.identifier}: {issue.title}{state_str} (similarity: {issue.similarity_score:.2f})")
        
        # Rich context only: KB snippets
        if is_rich_context and context.kb_snippets:
            parts.append("\n### Project Knowledge:")
            for snippet in context.kb_snippets[:3]:
                parts.append(f"- {snippet.content[:200]}...")
        
        # Rich context only: Closed issue insights
        if is_rich_context and hasattr(context, 'closed_issue_insights') and context.closed_issue_insights:
            parts.append("\n### Learnings from Similar Closed Issues:")
            for insight in context.closed_issue_insights[:3]:
                parts.append(f"- **{insight.identifier}**: {insight.title}")
                if insight.resolution_summary:
                    parts.append(f"  Resolution: {insight.resolution_summary[:200]}...")
                if insight.learnings:
                    parts.append(f"  Learnings: {', '.join(insight.learnings[:3])}")
        
        parts.append("\n---")
        parts.append("\nAnalyze the above and produce a strategic plan as JSON.")
        
        return "\n".join(parts)
    
    def _parse_json_output(self, text: str) -> dict:
        """
        Parse JSON from agent output.
        
        Handles markdown code blocks and raw JSON.
        """
        # Try to extract JSON from markdown code block
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
            # Return a structured error response
            return {
                "intent_summary": "Failed to parse structured output",
                "recommended_strategy": {
                    "approach": "Manual review required",
                    "rationale": "Agent output was not valid JSON"
                },
                "execution_plan": [],
                "assumptions": [],
                "risks": [],
                "open_questions": ["Review raw agent output for insights"],
                "_raw_output": text[:1000]
            }
    
    def _format_response_for_linear(self, structured_output: dict) -> str:
        """
        Format structured JSON as readable markdown for Linear.
        """
        parts = []
        
        # Intent
        if structured_output.get('intent_summary'):
            parts.append(f"## Intent\n{structured_output['intent_summary']}")
        
        # Strategy
        strategy = structured_output.get('recommended_strategy', {})
        if strategy:
            parts.append(f"\n## Strategy\n**Approach:** {strategy.get('approach', 'N/A')}")
            if strategy.get('rationale'):
                parts.append(f"\n**Rationale:** {strategy['rationale']}")
        
        # Execution Plan
        plan = structured_output.get('execution_plan', [])
        if plan:
            parts.append("\n## Execution Plan")
            for phase in plan:
                parts.append(f"\n### Phase {phase.get('phase', '?')}: {phase.get('title', 'Untitled')}")
                if phase.get('tasks'):
                    for task in phase['tasks']:
                        parts.append(f"- [ ] {task}")
                if phase.get('output'):
                    parts.append(f"\n**Output:** {phase['output']}")
                if phase.get('estimated_effort'):
                    parts.append(f"**Effort:** {phase['estimated_effort']}")
        
        # Assumptions
        assumptions = structured_output.get('assumptions', [])
        if assumptions:
            parts.append("\n## Assumptions")
            for a in assumptions:
                parts.append(f"- {a}")
        
        # Risks
        risks = structured_output.get('risks', [])
        if risks:
            parts.append("\n## Risks")
            for r in risks:
                if isinstance(r, dict):
                    parts.append(f"- **{r.get('risk', 'Unknown')}**")
                    if r.get('mitigation'):
                        parts.append(f"  - Mitigation: {r['mitigation']}")
                else:
                    parts.append(f"- {r}")
        
        # Open Questions
        questions = structured_output.get('open_questions', [])
        if questions:
            parts.append("\n## Open Questions")
            for q in questions:
                parts.append(f"- {q}")
        
        return "\n".join(parts)
    
    async def run(self, context) -> AgentResult:
        """
        Execute strategic planning.
        
        Accepts either AgentContext or StrategistAgentContext.
        Returns structured JSON in metadata for downstream processing.
        """
        try:
            print(f"\n=== STRATEGIST AGENT EXECUTING ===")
            print(f"Issue: {context.issue_identifier} - {context.issue_title}")
            print(f"Trigger: {context.trigger_type}")
            
            # Build project-scoped database and agent at runtime
            project_id = getattr(context, 'project_id', None)
            self.db = self._build_db(project_id)
            
            # Build project knowledge base from Qdrant (if project exists)
            knowledge = None
            if project_id:
                try:
                    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
                    collection_name = f"project_{project_id.replace('-', '_').lower()}"
                    
                    vector_db = Qdrant(
                        collection=collection_name,
                        url=qdrant_url,
                    )
                    contents_db = SqliteDb(
                                db_file="data/cortex_memory.db",
                                knowledge_table=f"knowledge_contents_{collection_name}"
                            )
                    
                    knowledge = Knowledge(
                        name=f"Project {project_id} Knowledge",
                        description=f"Knowledge base for project {project_id}",
                        vector_db=vector_db,
                        contents_db=contents_db,
                    )
                    print(f"StrategistAgent: Qdrant KB initialized (collection: {collection_name})")
                except Exception as kb_error:
                    print(f"StrategistAgent: KB initialization failed (will proceed without KB): {kb_error}")
            
            self.agent = self._create_agent(self.db, knowledge=knowledge)
            
            # Build prompt
            prompt = self._build_prompt(context)
            
            # Run the agent
            start_time = datetime.now(timezone.utc)
            response = await self.agent.arun(prompt)
            end_time = datetime.now(timezone.utc)
            
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Extract response text
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse structured output
            structured_output = self._parse_json_output(response_text)
            
            # Format for Linear
            formatted_response = self._format_response_for_linear(structured_output)
            
            print(f"=== STRATEGIST AGENT COMPLETE ({execution_time_ms}ms) ===\n")
            
            return AgentResult(
                success=True,
                response=formatted_response,
                status="completed",
                metadata={
                    "agent": "Strategist",
                    "issue_id": context.issue_id,
                    "trigger_type": context.trigger_type,
                    "execution_time_ms": execution_time_ms,
                    "structured_output": structured_output,
                }
            )
            
        except Exception as e:
            print(f"StrategistAgent error: {str(e)}")
            return AgentResult(
                success=False,
                response=f"Strategy planning failed: {str(e)}",
                status="error",
                metadata={
                    "agent": "Strategist",
                    "error": str(e),
                }
            )


# Auto-register on import
AgentRegistry.register("Strategist", StrategistAgent)
