"""
Main entry point for tree-based execution with Redis state store.

Usage:
    python main_v2.py "Your research question here"
    python main_v2.py "Your question" --backend groq --max-depth 3
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import string
import time

from agents.core import RunConfig
from agents.llm import load_env, call_llm, call_llm_json
from core_logging import make_logger
from graph_v2 import run_tree_graph
from memory.redis_store import create_state_store
from memory.store import make_run_paths
from orchestration.decomposer import create_decomposer
from orchestration.assembler import create_assembler


def _run_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M", time.localtime())
    suf = "".join(random.choice(string.hexdigits.lower()) for _ in range(4))
    return f"run_{ts}_{suf}"


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None and v != "" else float(default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None and v != "" else int(default)
    except Exception:
        return int(default)


def _write_output_md(paths, problem: str, final_output: str, elapsed_s: float) -> None:
    lines = []
    lines.append(f"# Slowly Run — {paths.run_id}\n")
    lines.append("## Problem\n")
    lines.append(problem.strip() + "\n")
    lines.append("## Answer\n")
    lines.append(final_output.strip() + "\n")
    lines.append("## Run Stats\n")
    lines.append(f"- elapsed_seconds: {elapsed_s:.1f}\n")
    with open(paths.output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def _amain() -> int:
    load_env()

    parser = argparse.ArgumentParser(description="Slowly - Tree-based multi-agent research")
    parser.add_argument("problem", nargs="+", help="Problem statement")
    parser.add_argument("--backend", choices=["ollama", "groq"], default=None, help="Override ACTIVE_BACKEND")
    parser.add_argument("--model", type=str, default=None, help="Override PRIMARY_MODEL")
    parser.add_argument("--hours", type=float, default=None, help="Time budget in hours")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Runs directory")
    parser.add_argument("--quiet", action="store_true", help="Less console output")
    args = parser.parse_args()

    problem = " ".join(args.problem).strip()
    if args.backend:
        os.environ["ACTIVE_BACKEND"] = args.backend
    if args.max_depth:
        os.environ["MAX_TASK_DEPTH"] = str(args.max_depth)

    time_budget_minutes = _env_float("TIME_BUDGET_MINUTES", 480.0)
    if args.hours is not None:
        time_budget_minutes = float(args.hours) * 60.0

    backend = os.getenv("ACTIVE_BACKEND", "ollama")
    primary_model = os.getenv("PRIMARY_MODEL", "qwen2.5:7b")
    fast_model = os.getenv("FAST_MODEL") or primary_model
    if backend == "groq":
        primary_model = os.getenv("GROQ_PRIMARY_MODEL") or "llama-3.1-8b-instant"
        fast_model = os.getenv("GROQ_FAST_MODEL") or primary_model
    if args.model:
        primary_model = args.model

    run_id = _run_id()
    paths = make_run_paths(args.runs_dir, run_id)
    logger = make_logger(args.runs_dir, run_id, quiet=bool(args.quiet))

    # Initialize state store
    store = await create_state_store()

    # Initialize orchestration components
    decomposer = create_decomposer()
    assembler = create_assembler()

    cfg = RunConfig(
        primary_model=primary_model,
        fast_model=fast_model,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        max_parallel_agents=_env_int("MAX_PARALLEL_AGENTS", 1),
        max_total_tasks=_env_int("MAX_TOTAL_TASKS", 500),
        max_task_depth=_env_int("MAX_TASK_DEPTH", 5),
        min_score_improvement=_env_float("MIN_SCORE_IMPROVEMENT", 0.02),
        target_score=_env_float("TARGET_SCORE", 0.92),
    )

    def log_event(e: dict) -> None:
        logger.event(e)

    t0 = time.time()
    log_event({"type": "run_start", "message": "starting", "backend": backend, "mode": "tree"})

    try:
        final_output = await run_tree_graph(
            problem=problem,
            run_id=run_id,
            cfg=cfg,
            store=store,
            decomposer=decomposer,
            assembler=assembler,
            call_llm=call_llm,
            call_llm_json=call_llm_json,
            time_budget_seconds=float(time_budget_minutes) * 60.0,
            log_event=log_event,
        )
    except Exception as e:
        log_event({"type": "run_error", "error": str(e)})
        final_output = f"(Error: {e})"

    elapsed = time.time() - t0

    _write_output_md(paths, problem=problem, final_output=final_output, elapsed_s=elapsed)

    log_event({"type": "run_end", "message": "done", "elapsed_s": round(elapsed, 2)})

    # Cleanup
    await store.close()

    print(f"\n{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Output: {paths.output_md_path}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
