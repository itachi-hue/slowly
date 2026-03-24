# Slowly v2 - Quick Start Guide

## Prerequisites

1. **Python 3.10+**
2. **Ollama** (local) or **Groq API key** (cloud)
3. **Redis** (optional - falls back to in-memory)

## Setup

```bash
cd slowly

# Option 1: Use the start script (recommended)
./start.sh

# Option 2: Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Configuration

Edit `.env`:

```bash
# For local Ollama
ACTIVE_BACKEND=ollama
PRIMARY_MODEL=qwen2.5:7b

# For Groq (faster, cloud)
ACTIVE_BACKEND=groq
GROQ_API_KEY=your-key-here
GROQ_PRIMARY_MODEL=llama-3.3-70b-versatile

# Tree depth (lower = faster for testing)
MAX_TASK_DEPTH=3
```

## Running

### Web Visualization (Recommended)

```bash
./start.sh
# or
python server.py
```

Open http://localhost:8080 in your browser.

### CLI Mode

```bash
./start.sh cli "Your research question"
# or
python main_v2.py "Your research question"
```

Options:
- `--backend ollama|groq` - Override backend
- `--model MODEL` - Override model
- `--max-depth N` - Maximum tree depth (default: 5)
- `--hours N` - Time budget in hours

## Starting Redis (Optional)

Redis enables persistent state across runs. Without it, the system uses in-memory storage.

```bash
# Using Docker
docker run -d -p 6379:6379 redis:alpine

# Using Homebrew (macOS)
brew install redis
brew services start redis
```

## How It Works

1. **You enter a question**
2. **Decomposer** decides if it needs sub-tasks
3. **Children execute** - each sees only its question
4. **Parents assemble** - pull children outputs from Redis, combine
5. **Bubbles up** until root completes

### Example Tree

```
"Find best VCs for voice AI"
├─ "List top 10 VCs in voice AI"
│   ├─ "Seed-stage VCs" → executes
│   └─ "Growth-stage VCs" → executes
│   └─ [assembles into VC list]
├─ "Typical check sizes" → executes
└─ "Recent deals" → executes
└─ [assembles into final answer]
```

## Troubleshooting

### Ollama not running
```bash
ollama serve
```

### Model not found
```bash
ollama pull qwen2.5:7b
```

### Redis connection failed
The system will fall back to in-memory storage. This is fine for testing.

### Slow execution
- Use `--max-depth 2` for faster testing
- Switch to Groq for faster inference
- Reduce `MAX_TOTAL_TASKS` in `.env`
