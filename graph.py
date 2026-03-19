from __future__ import annotations

import asyncio
import os
import time
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph  # type: ignore

from agents.core import (
    EvaluatorAgent,
    OrchestratorAgent,
    RunConfig,
    SynthesizerAgent,
    run_task_tree,
)
from memory.store import CritiqueReport, SynthesisVersion, TaskOutput


class AgentState(TypedDict):
    run_id: str
    problem: str
    iteration: int
    max_iterations: int
    time_budget_seconds: float
    start_time: float
    current_tasks: list[dict]
    current_outputs: list[dict]
    current_synthesis: Optional[str]
    current_score: float
    current_critique: Optional[dict]
    score_history: list[float]
    all_syntheses: list[dict]
    should_continue: bool
    final_output: Optional[str]
    status_message: str


ITERATION_BUDGETS_MIN = {0: 90, 1: 60, 2: 45, 3: 30, 4: 20}


def _iter_budget_seconds(iteration: int) -> float:
    if iteration in ITERATION_BUDGETS_MIN:
        return ITERATION_BUDGETS_MIN[iteration] * 60.0
    return 20 * 60.0


def _iter_budget_seconds_dynamic(state: AgentState) -> float:
    """
    Dynamically allocate per-iteration time as a function of the remaining total run time.
    Goal: use more time early, but scale naturally for long runs (e.g. 10 hours).

    Strategy:
    - Allocate a share of remaining time based on decreasing weights.
    - Enforce a min and max per iteration to keep iterations bounded.
    """
    remaining = _time_remaining(state)
    # Decreasing weights, length = max_iterations
    max_iters = max(1, int(state["max_iterations"]))
    # Example weights: 10,9,8,...,1 (front-loaded)
    weights = list(range(max_iters, 0, -1))
    idx = int(state["iteration"])
    if idx >= max_iters:
        return 0.0
    remaining_weights = sum(weights[idx:]) or 1
    share = weights[idx] / remaining_weights

    min_s = float(os.getenv("MIN_ITERATION_SECONDS", "900") or 900)  # 15 min
    max_s = float(os.getenv("MAX_ITERATION_SECONDS", "5400") or 5400)  # 90 min
    budget = remaining * share
    return max(0.0, min(max_s, max(min_s, budget)))


def _time_remaining(state: AgentState) -> float:
    return max(0.0, state["time_budget_seconds"] - (time.time() - state["start_time"]))


def _mk_logger(state: AgentState, log_event):
    def _log(e: dict):
        if log_event:
            log_event({"run_id": state["run_id"], "iteration": state["iteration"], **e})

    return _log


def build_graph(cfg: RunConfig, log_event=None, runs_dir: str = "runs"):
    orchestrator = OrchestratorAgent()
    evaluator = EvaluatorAgent(cfg)
    synthesizer = SynthesizerAgent()

    async def decompose(state: AgentState) -> AgentState:
        log = _mk_logger(state, log_event)
        log({"type": "node_start", "node": "decompose"})
        critique = state["current_critique"]
        tasks = await orchestrator.decompose(
            problem=state["problem"],
            iteration=state["iteration"],
            critique=critique,
            model=cfg.primary_model,
        )
        state["current_tasks"] = tasks
        state["status_message"] = f"Decomposed into {len(tasks)} tasks."
        log({"type": "node_end", "node": "decompose", "tasks": len(tasks)})
        return state

    async def execute(state: AgentState) -> AgentState:
        log = _mk_logger(state, log_event)
        log({"type": "node_start", "node": "execute", "tasks": len(state["current_tasks"])})
        # Budget this execute step so spawning is primarily limited by time remaining.
        remaining = _time_remaining(state)
        mode = (os.getenv("ITERATION_BUDGET_MODE", "dynamic") or "dynamic").lower().strip()
        if mode == "fixed":
            iter_budget = _iter_budget_seconds(state["iteration"])
        else:
            iter_budget = _iter_budget_seconds_dynamic(state)
        budget_s = min(remaining, iter_budget)
        deadline_ts = time.time() + max(0.0, budget_s)
        # Stop spawning/starting new tasks shortly before the deadline to guarantee wrap-up.
        wrapup_buffer_s = float(os.getenv("WRAPUP_BUFFER_SECONDS", "120") or 120)
        spawn_cutoff_ts = max(time.time(), deadline_ts - max(0.0, wrapup_buffer_s))
        outputs: list[TaskOutput] = await run_task_tree(
            cfg=cfg,
            tasks=state["current_tasks"],
            problem=state["problem"],
            iteration=state["iteration"],
            critique=state["current_critique"],
            log_event=log_event,
            deadline_ts=deadline_ts,
            spawn_cutoff_ts=spawn_cutoff_ts,
        )
        state["current_outputs"] = [o.model_dump() for o in outputs]
        state["status_message"] = f"Executed {len(outputs)} task outputs."
        log({"type": "node_end", "node": "execute", "outputs": len(outputs)})
        return state

    async def synthesize(state: AgentState) -> AgentState:
        log = _mk_logger(state, log_event)
        log({"type": "node_start", "node": "synthesize"})
        outputs = [TaskOutput(**o) for o in state["current_outputs"]]
        versions = [SynthesisVersion(**v) for v in state["all_syntheses"]]
        v = await synthesizer.synthesize(
            problem=state["problem"],
            task_outputs=outputs,
            iteration=state["iteration"],
            previous_synthesis=state["current_synthesis"],
            critique=state["current_critique"],
            all_previous_versions=versions,
            model=cfg.primary_model,
        )
        state["current_synthesis"] = v.content
        state["all_syntheses"].append(v.model_dump())
        state["status_message"] = f"Synthesized version {v.version_id}."
        log({"type": "node_end", "node": "synthesize", "version_id": v.version_id})
        # Write per-iteration synthesis so it's inspectable before run completes
        try:
            it_path = os.path.join(runs_dir, f"{state['run_id']}_it{state['iteration']}_synthesis.md")
            with open(it_path, "w", encoding="utf-8") as f:
                f.write(f"# Iteration {state['iteration']} Synthesis\n\n{v.content}\n")
        except Exception:
            pass
        return state

    async def evaluate(state: AgentState) -> AgentState:
        log = _mk_logger(state, log_event)
        log({"type": "node_start", "node": "evaluate"})
        critique: CritiqueReport = await evaluator.evaluate(
            problem=state["problem"],
            synthesis=state["current_synthesis"] or "",
            iteration=state["iteration"],
            previous_scores=state["score_history"],
            model=cfg.primary_model,
        )
        state["current_score"] = float(critique.overall_score)
        state["score_history"].append(float(critique.overall_score))
        state["current_critique"] = critique.model_dump()

        # Determine continuation
        remaining = _time_remaining(state)
        enough_time = remaining > (20 * 60.0)
        under_iter = state["iteration"] + 1 < state["max_iterations"]
        should = (not critique.diminishing_returns_signal) and under_iter and enough_time
        state["should_continue"] = bool(should)
        state["status_message"] = f"Score {critique.overall_score:.3f}. Continue={state['should_continue']}."
        log(
            {
                "type": "node_end",
                "node": "evaluate",
                "score": critique.overall_score,
                "diminishing": critique.diminishing_returns_signal,
                "time_remaining_s": round(remaining, 1),
                "should_continue": state["should_continue"],
            }
        )
        return state

    async def iterate(state: AgentState) -> AgentState:
        log = _mk_logger(state, log_event)
        log({"type": "node_start", "node": "iterate"})
        state["iteration"] += 1
        state["current_tasks"] = []
        state["current_outputs"] = []
        # keep synthesis/critique for targeted improvements
        state["status_message"] = f"Iteration -> {state['iteration']}"
        log({"type": "node_end", "node": "iterate", "iteration": state["iteration"]})
        return state

    async def finalize(state: AgentState) -> AgentState:
        log = _mk_logger(state, log_event)
        log({"type": "node_start", "node": "finalize"})
        versions = [SynthesisVersion(**v) for v in state["all_syntheses"]]
        # attach scores from history if not present (best effort)
        # (scores are stored in critique table by the runner; graph keeps score_history)
        final = await synthesizer.final_synthesis(problem=state["problem"], all_versions=versions, model=cfg.primary_model)
        state["final_output"] = final
        state["status_message"] = "Finalized."
        log({"type": "node_end", "node": "finalize", "chars": len(final)})
        return state

    def should_iterate(state: AgentState) -> str:
        return "iterate" if state["should_continue"] else "finalize"

    g = StateGraph(AgentState)
    g.add_node("decompose", decompose)
    g.add_node("execute", execute)
    g.add_node("synthesize", synthesize)
    g.add_node("evaluate", evaluate)
    g.add_node("iterate", iterate)
    g.add_node("finalize", finalize)

    g.set_entry_point("decompose")
    g.add_edge("decompose", "execute")
    g.add_edge("execute", "synthesize")
    g.add_edge("synthesize", "evaluate")
    g.add_conditional_edges("evaluate", should_iterate, {"iterate": "iterate", "finalize": "finalize"})
    g.add_edge("iterate", "decompose")
    g.add_edge("finalize", END)
    return g.compile()


async def run_graph(app, state: AgentState) -> AgentState:
    # Use async API because our nodes are async.
    if hasattr(app, "ainvoke"):
        return await app.ainvoke(state)
    # Fallback (older langgraph): run sync invoke in a thread.
    return await asyncio.to_thread(app.invoke, state)


