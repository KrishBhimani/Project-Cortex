# 🧠 Project Cortex

A **collaborative swarm of autonomous AI agents** for Linear that transforms issue tracking into an intelligent project management system — featuring context-aware research, strategic planning, and persistent project memory.

> *"Your Linear workspace, supercharged with specialized AI agents that think, research, and strategize on demand."*

---

## Core Design Philosophy

Cortex orchestrates a **multi-agent system** built around these principles:

| Principle | What it means |
|-----------|---------------|
| **Autonomous agents** | Specialized agents that independently reason and execute tasks |
| **Collaborative intelligence** | Agents share context and build on each other's outputs |
| **Deterministic orchestration** | Same inputs → same agent context → reproducible reasoning |
| **Curated collective memory** | Knowledge is explicitly saved to project-scoped memory |
| **Seamless integration** | Linear remains the only UI — agents work where you work |

> Cortex combines autonomous reasoning with human oversight — agents act intelligently, but only when summoned.

---

## Agent Swarm Architecture

Cortex deploys **specialized autonomous agents** that collaborate through shared context:

### 🔬 ResearcherAgent — *The Intelligence Gatherer*
| Aspect | Details |
|--------|---------|
| **Trigger** | `@researcher` mention in issue/comment |
| **Capability** | Autonomous web research, source synthesis, fact extraction |
| **Intelligence** | Tavily-powered deep search with relevance ranking |
| **Output** | Structured research summaries with citations |

### 🧠 StrategistAgent — *The Strategic Planner*
| Aspect | Details |
|--------|---------|
| **Trigger** | `@strategist` mention |
| **Capability** | Converts ambiguity into actionable strategy and execution plans |
| **Intelligence** | Reasons over issue history, prior agent outputs, and project patterns |
| **Output** | Structured JSON (`intent_summary`, `execution_plan`, `risks`, `assumptions`) |

### 🔍 PerplexityAgent — *The Deep Researcher*
| Aspect | Details |
|--------|---------|
| **Trigger** | `@perplexity` mention |
| **Capability** | Multi-modal research (ask, research, reason) with real-time web access |
| **Intelligence** | Perplexity AI with automatic source citation |
| **Output** | Comprehensive research with inline citations |

> **Collaborative Flow:** ResearcherAgent gathers intelligence → StrategistAgent synthesizes into plans → outputs feed into project memory for future agents.

---

## AgentContext

Each agent receives a frozen, read-only `AgentContext` containing:

- Issue details (title, description, state, labels)
- Trigger information (issue vs comment, body text)
- Project scope (ID, name)
- Retrieved context (related issues, KB snippets)
- Authentication (access token for Linear API)

> Agents never query databases or external systems directly.  
> All retrieval happens **before** agent invocation.

---

## Data & Storage Model

| Layer | Store | Purpose |
|-------|-------|---------|
| **Structured History** | PostgreSQL | Issues, comments, agent executions |
| **Related Issues** | pgvector | Similarity search on issue embeddings |
| **Project KB** | Qdrant (planned) | Curated project knowledge |
| **Active Context** | In-memory | Current agent execution state |

### Database Schema (`cortex` schema)
```
cortex.issues           → Synced Linear issues with embeddings
cortex.issue_comments   → Discussion history
cortex.agent_executions → What agents did, when, and why
```

---

## Event Flow

```mermaid
flowchart TD
    A["🔔 Linear Webhook Event"] --> B{"Event Type?"}
    
    B -->|"Issue/Comment"| C["/sync_webhook"]
    C --> D[("PostgreSQL")]
    D --> E["Generate Embeddings"]
    E --> D
    
    B -->|"Agent Tagged"| F["/webhook"]
    F --> G["Context Assembly"]
    G --> H[("Fetch from DB")]
    H --> I["Find Related Issues"]
    I --> J["Agent Registry"]
    J --> K{"Route to Agent"}
    
    K -->|"@researcher"| L["🔬 ResearcherAgent"]
    K -->|"@strategist"| M["🧠 StrategistAgent"]
    K -->|"@perplexity"| N["🔍 PerplexityAgent"]
    
    L --> O["Agent Execution"]
    M --> O
    N --> O
    
    O --> P["Log to agent_executions"]
    P --> Q["📤 Post to Linear"]
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Framework | FastAPI + Uvicorn |
| AI Framework | Agno |
| Database | PostgreSQL + pgvector |
| Vector DB | Qdrant (planned) |
| LLM | OpenAI GPT-4.1 |
| Embeddings | OpenAI ada-002 |
| Research | Tavily, Perplexity MCP |
| Integration | Linear Webhooks + GraphQL |

---

## What This Project Is NOT

| ❌ Not | Why |
|--------|-----|
| A chatbot | Agents are task-specific, not conversational |
| Autonomous | No background processing without explicit invocation |
| Self-learning | Memory is curated, not emergent |
| A RAG demo | Production-focused with structured outputs |

---

## Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL with pgvector extension
- Linear workspace with agent app configured

### Environment Variables
```env
DB_URL=postgresql://user:pass@host:5432/db
OPENAI_API_KEY=sk-...
LINEAR_API_KEY=lin_api_...
CLIENT_ID=...
CLIENT_SECRET=...
TAVILY_API_KEY=tvly-...
PERPLEXITY_API_KEY=pplx-...
```

### Run
```bash
# Install dependencies
uv sync

# Run database migrations
psql $DB_URL -f migrations/001_create_cortex_schema.sql

# Start server
uv run server.py
```

### Configure Linear Webhooks
1. `/webhook` → Agent session events (agent mentions)
2. `/sync_webhook` → Issue and Comment events (data sync)

---

## Future Roadmap

- [ ] Qdrant KB integration for project knowledge
- [ ] MemoryAgent for explicit knowledge curation
- [ ] Cross-project pattern recognition
- [ ] Agent execution metrics and quality scoring
- [ ] Multi-agent orchestration

---

## Design Principles

> Cortex orchestrates a **swarm of autonomous AI agents** with **collaborative intelligence**, **persistent memory**, and **human-in-the-loop oversight** — delivering the power of AI without sacrificing control.

```
PostgreSQL stores what happened.
Vector DB stores what mattered.
Agents decide what to do next.
```

---

<p align="center">
  <strong>Built with 🧠 by agents, for humans.</strong>
</p>
