from __future__ import annotations

import argparse
import asyncio
import os
import random
import string
import time

from agents.core import RunConfig
from agents.llm import load_env
from core_logging import make_logger
from graph import build_graph, run_graph
from memory.store import SQLiteStore, TaskOutput, SynthesisVersion, CritiqueReport, make_run_paths


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


def _write_output_md(
    paths, problem: str, final_output: str, score_history: list[float], elapsed_s: float, critique: dict | None = None
) -> None:
    lines = []
    lines.append(f"# Overnight Agent Run — {paths.run_id}\n")
    lines.append("## Problem\n")
    lines.append(problem.strip() + "\n")
    lines.append("## Final Answer\n")
    lines.append(final_output.strip() + "\n")
    lines.append("## Score Progression\n")
    if score_history:
        lines.append("\n".join([f"- iteration {i}: {s:.3f}" for i, s in enumerate(score_history)]) + "\n")
    else:
        lines.append("- (no scores)\n")
    if critique:
        lines.append("## Last Evaluation\n")
        lines.append(f"- **Score**: {critique.get('overall_score', 'N/A')}\n")
        if critique.get("strengths"):
            lines.append("- **Strengths**: " + "; ".join(str(s) for s in critique["strengths"][:5]) + "\n")
        if critique.get("weaknesses"):
            lines.append("- **Weaknesses**:\n")
            for w in (critique.get("weaknesses") or [])[:5]:
                lines.append(f"  - {w.get('dimension', '')}: {w.get('description', '')}\n")
        if critique.get("suggested_fixes"):
            lines.append("- **Suggested fixes**: " + "; ".join(str(f.get("action", "")) for f in critique["suggested_fixes"][:3]) + "\n")
    lines.append("## Run Stats\n")
    lines.append(f"- elapsed_seconds: {elapsed_s:.1f}\n")
    lines.append(f"- iterations_completed: {len(score_history)}\n")
    with open(paths.output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def _amain() -> int:
    load_env()

    parser = argparse.ArgumentParser(description="Overnight multi-agent reasoning runner")
    parser.add_argument("problem", nargs="+", help="Problem statement")
    parser.add_argument("--backend", choices=["ollama", "groq"], default=None, help="Override ACTIVE_BACKEND")
    parser.add_argument("--model", type=str, default=None, help="Override PRIMARY_MODEL (or GROQ_PRIMARY_MODEL if backend=groq)")
    parser.add_argument("--fast-model", type=str, default=None, help="Override FAST_MODEL (or GROQ_FAST_MODEL if backend=groq)")
    parser.add_argument("--hours", type=float, default=None, help="Time budget in hours (overrides env)")
    parser.add_argument("--iterations", type=int, default=None, help="Max iterations (overrides env)")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Runs directory")
    parser.add_argument("--quiet", action="store_true", help="Less console output")
    args = parser.parse_args()

    problem = " ".join(args.problem).strip()
    if args.backend:
        os.environ["ACTIVE_BACKEND"] = args.backend

    time_budget_minutes = _env_float("TIME_BUDGET_MINUTES", 480.0)
    if args.hours is not None:
        time_budget_minutes = float(args.hours) * 60.0

    max_iterations = _env_int("MAX_ITERATIONS", 10)
    if args.iterations is not None:
        max_iterations = int(args.iterations)

    backend = os.getenv("ACTIVE_BACKEND", "ollama")
    primary_model = os.getenv("PRIMARY_MODEL", "qwen2.5:7b")
    fast_model = os.getenv("FAST_MODEL") or primary_model  # default same as primary (avoids 404 if fast not pulled)
    if backend == "groq":
        primary_model = os.getenv("GROQ_PRIMARY_MODEL") or "llama-3.1-8b-instant"
        fast_model = os.getenv("GROQ_FAST_MODEL") or primary_model
    if args.model:
        primary_model = args.model
    if args.fast_model:
        fast_model = args.fast_model

    run_id = _run_id()
    paths = make_run_paths(args.runs_dir, run_id)
    logger = make_logger(args.runs_dir, run_id, quiet=bool(args.quiet))
    store = SQLiteStore(paths.db_path)

    cfg = RunConfig(
        primary_model=primary_model,
        fast_model=fast_model,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        max_parallel_agents=_env_int("MAX_PARALLEL_AGENTS", 1),  # 1 = serial queue; increase for parallel
        # Defaults are intentionally high; time budget + concurrency are the primary limiters.
        max_total_tasks=_env_int("MAX_TOTAL_TASKS", 500),
        max_task_depth=_env_int("MAX_TASK_DEPTH", 12),
        min_score_improvement=_env_float("MIN_SCORE_IMPROVEMENT", 0.02),
        target_score=_env_float("TARGET_SCORE", 0.92),
    )

    def log_event(e: dict) -> None:
        logger.event(e)

    app = build_graph(cfg, log_event=log_event, runs_dir=args.runs_dir)

    state = {
        "run_id": run_id,
        "problem": problem,
        "iteration": 0,
        "max_iterations": max_iterations,
        "time_budget_seconds": float(time_budget_minutes) * 60.0,
        "start_time": time.time(),
        "current_tasks": [],
        "current_outputs": [],
        "current_synthesis": None,
        "current_score": 0.0,
        "current_critique": None,
        "score_history": [],
        "all_syntheses": [],
        "should_continue": True,
        "final_output": None,
        "status_message": "starting",
    }

    t0 = time.time()
    log_event({"type": "run_start", "message": "starting", "backend": os.getenv("ACTIVE_BACKEND", "ollama")})
    final_state = await run_graph(app, state)
    elapsed = time.time() - t0

    # Persist outputs/syntheses/critiques into SQLite (best-effort)
    # Note: graph state only retains the *last* iteration's outputs today.
    try:
        for o in final_state.get("current_outputs", []) or []:
            store.save_task_output(TaskOutput(**o))
    except Exception:
        pass
    try:
        # all_syntheses already includes all versions; attach score if possible
        score_hist = list(final_state.get("score_history", []) or [])
        for idx, v in enumerate(final_state.get("all_syntheses", []) or []):
            sv = SynthesisVersion(**v)
            if idx < len(score_hist):
                sv.score = float(score_hist[idx])
            store.save_synthesis(sv)
    except Exception:
        pass
    try:
        crit = final_state.get("current_critique")
        if isinstance(crit, dict):
            store.save_critique(CritiqueReport(**crit))
    except Exception:
        pass

    final_output = final_state.get("final_output") or final_state.get("current_synthesis") or ""
    score_history = list(final_state.get("score_history", []) or [])
    critique = final_state.get("current_critique")
    critique_d = critique.model_dump() if hasattr(critique, "model_dump") else (critique if isinstance(critique, dict) else None)
    _write_output_md(paths, problem=problem, final_output=final_output, score_history=score_history, elapsed_s=elapsed, critique=critique_d)

    log_event({"type": "run_end", "message": "done", "elapsed_s": round(elapsed, 2)})
    store.close()
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())


