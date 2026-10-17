# Multi-Agent Growth Diagnosis System

[中文](README.md) | **English**

A multi-agent e-commerce growth diagnosis system that runs with **zero API keys and no network access**, built for teaching demos and offline development.

A Supervisor and five domain experts (Acquisition, Activation, Retention, Referral, Revenue) are orchestrated as a single graph with **LangGraph**. Each expert has its own **memory** (working memory plus episodic memory) and **self-reflects** after every diagnosis. When the Supervisor writes the final summary, it retrieves domain knowledge through **RAG (ChromaDB)** and injects it into the strategy.

> Core capability: **ChromaDB + RAG**.
> - **Shared team knowledge base:** `src/rag.py` builds a ChromaDB vector store; the Supervisor retrieves domain knowledge from it when summarizing.
> - **Vectorized episodic memory:** the episodic memory in `src/memory.py` uses ChromaDB vector search instead of plain keyword matching, and every agent has its own vector memory store.
> - The default embedding model is English-oriented, so semantic recall on Chinese text is only fair. Both places therefore use **ChromaDB vector search first, with a keyword Jaccard fallback**. If retrieval fails, the system falls back automatically and keeps working offline.

---

## Project Structure

```
multi-agent-growth-diagnosis/
├── run.py                  # Entry point: run a diagnosis in one command
├── requirements.txt        # Dependencies
├── README.md               # Chinese README
├── README.en.md            # This document
└── src/
    ├── config.py           # Config: brain mode / memory dir / RAG dir / scenario
    ├── knowledge.py        # Expert personas, stage order, rating text (step 1)
    ├── llm.py              # Brains: MockBrain / LLMBrain (step 2)
    ├── tools.py            # 5 exact-arithmetic tools (step 3)
    ├── agents.py           # Expert agent = persona + tools + brain + memory + reflection (step 4)
    ├── supervisor.py       # Supervisor: plan + dynamic routing + RAG summary (step 5)
    ├── graph.py            # LangGraph orchestration (step 6)
    ├── state.py            # Agent state definition
    ├── memory.py           # Memory: working + episodic (ChromaDB vector search)
    ├── rag.py              # RAG knowledge base (ChromaDB + keyword fallback)
    ├── data_generator.py   # Multi-scenario data generation
    └── report.py           # Report output (JSON / TXT / PNG)
```

---

## Quick Start

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Run (default scenario: GMV decline)
python run.py

# 3. Pick a scenario
python run.py --scenario retention_crash
```

Each run writes three reports to `output/`: `multi_agent_report.json`, `.txt`, and `.png`.

You can also pass `--session <id>` to set a session ID and demonstrate cross-session memory.

### Does the first run need to download an embedding model?

**No network is required and nothing is forced to download.** `src/rag.py` and `src/memory.py` first try ChromaDB's default embedding (which may trigger a one-time local download on first use). If it is unavailable, they **fall back to keyword Jaccard retrieval automatically** and the whole system still runs end to end. See "RAG and Memory" below.

---

## Core Design: A Team with a Plan

### 1. Team roles (step 1)

| Role | Responsibility | Persona example |
|------|----------------|-----------------|
| Supervisor | Plans, routes dynamically, summarizes and reflects | "You are the Supervisor agent, in charge of orchestration" |
| Acquisition expert | Channel ROI / CAC / quality diagnosis | "You are the acquisition expert, focused on customer-acquisition efficiency" |
| Activation expert | Activation rate / first-order conversion diagnosis | "You are the activation expert, focused on new-user conversion" |
| Retention expert | Retention curve / Aha-moment diagnosis | "You are the retention expert, focused on user stickiness" |
| Referral expert | K-factor / virality diagnosis | "You are the referral expert, focused on viral spread" |
| Revenue expert | LTV / CAC and repurchase diagnosis | "You are the revenue expert, focused on monetization" |

> Cross-agent dependency: the revenue expert needs the CAC computed by the acquisition expert, so revenue is scheduled last.

### 2. Four decision points (step 2)

The brain (`src/llm.py`) handles the four places that need "thinking":

1. **Routing:** which expert the Supervisor dispatches next (`route_next`).
2. **Diagnosis:** an expert produces a diagnosis from tool results, team context, and its own memory (`diagnose`).
3. **Reflection:** an expert checks its own conclusion after diagnosing (`reflect`).
4. **Synthesis:** the Supervisor combines everything into a root-cause chain and a strategy, optionally injecting RAG knowledge (`synthesize`).

There are two implementations:

- **MockBrain** (default): deterministic, reproducible, no key needed.
- **LLMBrain**: connects to DeepSeek or any OpenAI-compatible endpoint. Change `LLM_MODE` in `config.py` to switch.

### 3. Graph orchestration (step 6)

```
START → plan → supervisor ⇄ [expert: recall memory → call tools → reason → reflect → write memory] → reporter → END
```

- **plan:** draws up an explicit plan before work starts and writes it into the conversation, so the whole run is observable.
- **supervisor:** routes dynamically through conditional edges, deciding whom to dispatch or whether to wrap up.
- **expert:** each expert returns to the supervisor after finishing, forming an **agentic loop**.
- **reporter:** summarizes, reflects, and adds RAG knowledge to produce the root-cause chain and strategy.

---

## RAG and Memory

### Shared team RAG knowledge base (`src/rag.py`)

- Builds a **ChromaDB** vector store persisted to `rag_db/`.
- Ships with a set of built-in e-commerce growth knowledge (for example "how to lower CAC" and "three levers to raise LTV").
- During synthesis, the Supervisor builds a query from the **critical and watch-list stages** and injects the top-K retrieved items into the strategy.
- Returns `(knowledge list, retrieval method)`, where the method is `vector` (ChromaDB) or `keyword` (fallback).

### Vectorized episodic memory (`src/memory.py`)

- Every agent has its own `Memory`:
  - **Working memory:** the most recent findings from the current run (short-term).
  - **Episodic memory:** past events, recalled through **ChromaDB vector search** rather than plain keyword matching.
- Persisted under `memory/<agent>/`, so it can be reused across sessions to demonstrate "remembering".

---

## Switching to a Real LLM (Optional)

1. Install `langchain-openai` (see the commented line in `requirements.txt`).
2. Set environment variables, or edit `src/config.py`:

```bash
# DeepSeek
set MUTI_AGENT_LLM_MODE=deepseek
set DEEPSEEK_API_KEY=your_key_here

# Or any OpenAI-compatible endpoint
set MUTI_AGENT_LLM_MODE=openai
set OPENAI_API_KEY=your_key_here
set OPENAI_BASE_URL=https://api.xxx.com/v1
```

(These are Windows `set` commands; on macOS or Linux use `export`.)

3. Run `python run.py` again.

---

## Teaching Notes

- **Why a graph instead of a hard-coded flow?** LangGraph's `add_conditional_edges` lets the Supervisor decide the next step dynamically instead of following fixed if-else logic. That is where the "intelligence" of a multi-agent system comes from.
- **Why memory?** An agent without memory starts from zero every time. With working memory plus episodic memory (ChromaDB vector search), it can remember earlier findings and draw on past experience.
- **Why reflection?** After diagnosing, an agent asks itself whether the conclusion is complete and whether it missed anything across stages. This turns one-shot output into a self-checking agent.
- **Where does RAG fit?** At the synthesis stage, where retrieved domain knowledge is injected into the strategy, giving the multi-agent system external-knowledge augmentation.

---

## FAQ

- **Error about a missing `langgraph`?** Run `pip install -r requirements.txt`.
- **ChromaDB is slow to initialize or fails?** The system falls back to keyword retrieval automatically and the overall run is unaffected.
- **Want a real LLM but have no key?** Keep `LLM_MODE=mock`. MockBrain demonstrates the whole flow.
