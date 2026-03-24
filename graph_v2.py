"""
Simplified graph for tree-based execution with Redis state store.

Most state now lives in Redis. The graph is minimal:
- Single execution node that runs the task tree
- State just tracks run_id, problem, and status
"""

from __future__ import annotations

import time
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.core import RunConfig
from agents.tree_executor import execute_tree, create_root_task
from memory.redis_store import RedisStateStore, InMemoryStateStore
from orchestration.decomposer import Decomposer
from orchestration.assembler import Assembler


class TreeState(TypedDict):
    """Minimal state - most data lives in Redis."""
    run_id: str
    problem: str
    root_task_id: str
    start_time: float
    time_budget_seconds: float
    final_output: Optional[str]
    status: str  # "pending" | "running" | "completed" | "error"


def build_tree_graph(
    cfg: RunConfig,
    store: RedisStateStore | InMemoryStateStore,
    decomposer: Decomposer,
    assembler: Assembler,
    call_llm,
    call_llm_json,
    log_event=None,
):
    """
    Build a simplified graph for tree execution.

    Args:
        cfg: Run configuration
        store: State store (Redis or in-memory)
        decomposer: Task decomposer
        assembler: Slot output assembler
        call_llm: Function for text LLM calls
        call_llm_json: Function for JSON LLM calls
        log_event: Optional logging callback
    """

    async def execute(state: TreeState) -> TreeState:
        """Execute the task tree until completion."""
        if log_event:
            log_event({"type": "graph_execute_start", "run_id": state["run_id"]})

        state["status"] = "running"

        try:
            # Execute the tree
            final_output = await execute_tree(
                run_id=state["run_id"],
                root_task_id=state["root_task_id"],
                store=store,
                decomposer=decomposer,
                assembler=assembler,
                config=cfg,
                call_llm=call_llm,
                call_llm_json=call_llm_json,
                log_event=log_event,
            )

            state["final_output"] = final_output
            state["status"] = "completed"

        except Exception as e:
            if log_event:
                log_event({"type": "graph_execute_error", "error": str(e)})
            state["final_output"] = f"(Error: {e})"
            state["status"] = "error"

        if log_event:
            log_event({
                "type": "graph_execute_complete",
                "run_id": state["run_id"],
                "status": state["status"],
                "output_len": len(state["final_output"] or ""),
            })

        return state

    # Build simple graph with single node
    g = StateGraph(TreeState)
    g.add_node("execute", execute)
    g.set_entry_point("execute")
    g.add_edge("execute", END)

    return g.compile()


async def run_tree_graph(
    problem: str,
    run_id: str,
    cfg: RunConfig,
    store: RedisStateStore | InMemoryStateStore,
    decomposer: Decomposer,
    assembler: Assembler,
    call_llm,
    call_llm_json,
    time_budget_seconds: float = 28800,  # 8 hours default
    log_event=None,
) -> str:
    """
    High-level function to run the tree graph.

    Args:
        problem: The problem to solve
        run_id: Unique run identifier
        cfg: Run configuration
        store: State store
        decomposer: Task decomposer
        assembler: Slot assembler
        call_llm: Function for text LLM calls
        call_llm_json: Function for JSON LLM calls
        time_budget_seconds: Time budget for execution
        log_event: Optional logging callback

    Returns:
        The final output string
    """
    # Create root task
    root_task_id = await create_root_task(
        run_id=run_id,
        problem=problem,
        store=store,
        decomposer=decomposer,
        config=cfg,
        call_llm_json=call_llm_json,
    )

    if log_event:
        log_event({
            "type": "root_task_created",
            "run_id": run_id,
            "root_task_id": root_task_id,
        })

    # Build and run graph
    app = build_tree_graph(
        cfg=cfg,
        store=store,
        decomposer=decomposer,
        assembler=assembler,
        call_llm=call_llm,
        call_llm_json=call_llm_json,
        log_event=log_event,
    )

    initial_state: TreeState = {
        "run_id": run_id,
        "problem": problem,
        "root_task_id": root_task_id,
        "start_time": time.time(),
        "time_budget_seconds": time_budget_seconds,
        "final_output": None,
        "status": "pending",
    }

    # Run the graph
    final_state = await app.ainvoke(initial_state)

    return final_state.get("final_output") or "(No output)"
