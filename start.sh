#!/bin/bash

# Slowly - Quick Start Script
# ===========================

set -e

echo "🌳 Slowly - Task Tree Research System"
echo "======================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo "   Edit .env to configure your settings"
    echo ""
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Check if Ollama is running (for local mode)
BACKEND=$(grep ACTIVE_BACKEND .env 2>/dev/null | cut -d'=' -f2 || echo "ollama")
if [ "$BACKEND" = "ollama" ]; then
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo ""
        echo "⚠️  Ollama not running. Start it with: ollama serve"
        echo "   Or switch to Groq: set ACTIVE_BACKEND=groq in .env"
        echo ""
    else
        echo "✅ Ollama is running"
    fi
fi

# Check Redis (optional)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is running"
    else
        echo "ℹ️  Redis not running (will use in-memory fallback)"
        echo "   Start with: docker run -d -p 6379:6379 redis:alpine"
    fi
else
    echo "ℹ️  Redis CLI not found (will use in-memory fallback)"
fi

echo ""
echo "======================================"
echo ""

# Parse command
case "${1:-web}" in
    web|server)
        echo "🚀 Starting web server..."
        echo "   Open: http://localhost:8080"
        echo ""
        python server.py
        ;;
    cli)
        shift
        echo "🚀 Running CLI..."
        python main_v2.py "$@"
        ;;
    *)
        echo "Usage: ./start.sh [command]"
        echo ""
        echo "Commands:"
        echo "  web     Start web visualization server (default)"
        echo "  cli     Run CLI: ./start.sh cli \"Your question here\""
        echo ""
        echo "Examples:"
        echo "  ./start.sh"
        echo "  ./start.sh web"
        echo "  ./start.sh cli \"Find best VCs for voice AI\""
        echo "  ./start.sh cli \"Solve x^2 + 2x + 1 = 0\" --max-depth 2"
        ;;
esac
