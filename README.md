# Slowly

Multi-agent research system that runs **slow, iterative, critique-driven** research. Decomposes problems into tasks, executes them with web search and code tools, synthesizes outputs, evaluates quality, and iterates until the answer is strong enough.

**Different from Perplexity/ChatGPT:** Built for depth over speed. Uses time budgets, adversarial evaluation, and multiple iterations to produce evidence-backed research reports.

## Architecture

```
decompose → execute → synthesize → evaluate → [iterate | finalize]
                    ↑__________________________|
                              (critique)
```

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
