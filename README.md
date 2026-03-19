# Slowly

Multi-agent research system that runs **slow, iterative, critique-driven** research. Decomposes problems into tasks, executes them with web search and code tools, synthesizes outputs, evaluates quality, and iterates until the answer is strong enough.

**Different from Perplexity/ChatGPT:** Built for depth over speed. Uses time budgets, adversarial evaluation, and multiple iterations to produce evidence-backed research reports.

## Architecture

### Simple overview

**In plain English:** You ask a research question. Slowly breaks it into sub-questions, answers each (using web search and tools), and merges everything into a report. If an answer raises new questions, it digs deeper. It repeats until the answer is good enough.

```
         "Best investors for voice AI?"
                         │
                         ▼
              ┌─────────────────────┐
              │  Break into tasks   │  ← Orchestrator
              └─────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    "Who are top    "What do they   "Top voice AI
     VCs?"           invest in?"     funds?"
          │              │              │
          │         (needs more?)        │
          │              │              │
          │        ┌─────┴─────┐        │
          │        ▼           ▼        │
          │   "Example deals"  "Typical │
          │                   check"   │
          └──────────┬────────────────┘
                     ▼
              ┌─────────────────────┐
              │  Merge → Report     │  ← Synthesizer
              └─────────────────────┘
                     │
                     ▼
              Score & critique → improve → repeat or done
```

### Iteration Graph (LangGraph)

The main flow is a **state graph** that runs one iteration at a time, conditionally looping until done:

```
                    ┌──────────────────────────┐
                    │                          │
                    ▼                          │
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     ┌───────────┐
    │ decompose │──▶│ execute  │──▶│synthesize│──▶│ evaluate │────▶│  iterate  │
    └──────────┘   └──────────┘   └──────────┘   └──────────┘     └───────────┘
          ▲                │            │              │                  │
          │                │            │              │                  │
          │                │            │              │    (continue?)   │
          │                │            │              ▼                  │
          │                │            │         ┌──────────┐            │
          │                │            │         │finalize  │            │
          │                │            │         └────┬─────┘            │
          │                │            │              │                  │
          └────────────────┴────────────┴──────────────┴──────────────────┘
                           critique feeds back into next iteration
```

- **decompose** – Orchestrator breaks the problem (and prior critique) into tasks
- **execute** – Runs the task tree (see below); produces outputs
- **synthesize** – Merges outputs into one report
- **evaluate** – Scores quality, produces critique; decides: iterate or finalize
- **iterate** – Increment iteration, loop back to decompose
- **finalize** – Produce final synthesis, end

### Task Tree (within execute)

Inside each execution step, tasks form a **dynamic tree**. Here’s the idea in plain terms:

**Example:** *"Who are the best investors for voice AI startups?"*

```
                    YOUR QUESTION
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    "Top 5 VCs?"    "What do they     "Example deals?"
                    invest in?"
         │                │                │
         │           ┌────┴────┐           │
         │           ▼         ▼           │
         │      "Typical   "Check sizes"    │
         │      sectors"                    │
         └────────────┼─────────────────────┘
                      ▼
              All answers merged
                      ▼
              One research report
```

- **Step 1:** Orchestrator breaks the question into 3–10 sub-questions.
- **Step 2:** Workers/Research agents answer each (web search, code, etc.).
- **Step 3:** If an answer suggests new questions (“What sectors?”), those become child tasks.
- **Step 4:** All task outputs are merged into one synthesis.

**Technical details:**

1. **Root tasks** – Orchestrator produces top-level tasks (depth 1)
2. **Child tasks** – If a task’s output has `open_questions`, those become child tasks (depth 2, 3, …)
3. **Parallel execution** – Tasks run in waves; a semaphore limits concurrency
4. **Expansion limits** – Tree stops when: `max_task_depth`, `max_total_tasks`, time budget, or wrap-up buffer is hit

- **Orchestrator** – Breaks the problem into parallel tasks (research / worker)
- **Workers & Research agents** – Use tools: web search, fetch page, run_command, read/write/search_replace files
- **Synthesizer** – Merges outputs into coherent reports
- **Evaluator** – Scores quality, finds weaknesses, drives iteration
- **LangGraph** – Coordinates the flow

## Requirements

- Python 3.10+
- **Ollama** (local) or **Groq** API key
- Optional: `TAVILY_API_KEY` for better search (falls back to DuckDuckGo)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your config
```

For local runs, ensure Ollama is running and pull a model:

```bash
ollama pull qwen2.5:7b
```

## Usage

```bash
python main.py "Your research question here."
```

**Options:**

| Flag | Description |
|------|-------------|
| `--backend ollama \| groq` | LLM backend |
| `--model MODEL` | Override primary model |
| `--hours N` | Time budget in hours |
| `--iterations N` | Max eval iterations |
| `--runs-dir DIR` | Output directory (default: `runs`) |
| `--quiet` | Less console output |

**Examples:**

```bash
# Local Ollama (default)
python main.py "Best investors for voice AI startups - names, contacts, funds."

# Groq backend
python main.py "Market research: who buys deep research reports?" --backend groq

# Shorter run
python main.py "Quick overview of X" --hours 1 --iterations 2
```

## Output

- `runs/{run_id}_output.md` – Final report
- `runs/{run_id}.jsonl` – Event log
- `runs/{run_id}_it{N}_synthesis.md` – Per-iteration synthesis

## Configuration

See `.env.example`. Key vars:

- `ACTIVE_BACKEND` – `ollama` or `groq`
- `PRIMARY_MODEL` – e.g. `qwen2.5:7b` (Ollama) or `llama-3.3-70b-versatile` (Groq)
- `TAVILY_API_KEY` – Optional; improves search quality
- `TIME_BUDGET_MINUTES` – Default 480 (8 hours)
- `MAX_PARALLEL_AGENTS` – 1 = serial (safest for Ollama), increase for Groq

## License

MIT
