"""
Tree Executor: Bottom-up task tree execution with Redis state store.

Key architecture:
- Leaves see only their question (no lineage, no context accumulation)
- Parents spawn children with ordered templates (slots can reference earlier slots)
- Assembly is bottom-up: each parent only sees direct children's outputs
- Context stays bounded at every level
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional, Callable, Any

from memory.redis_store import (
    TaskDefinition,
    TaskOutput,
    SlotDef,
    RedisStateStore,
    InMemoryStateStore,
    extract_slot_references,
)
from orchestration.decomposer import Decomposer
from orchestration.assembler import Assembler
from agents.core import RunConfig, WorkerAgent, ResearchAgent


def generate_task_id() -> str:
    """Generate a unique task ID."""
    return f"task_{uuid.uuid4().hex[:8]}"


async def execute_tree(
    run_id: str,
    root_task_id: str,
    store: RedisStateStore | InMemoryStateStore,
    decomposer: Decomposer,
    assembler: Assembler,
    config: RunConfig,
    call_llm: Callable,
    call_llm_json: Callable,
    log_event: Optional[Callable] = None,
) -> str:
    """
    Execute a task tree until root is completed.

    Args:
        run_id: Unique identifier for this run
        root_task_id: ID of the root task
        store: State store (Redis or in-memory)
        decomposer: Decomposer for deciding leaf vs decompose
        assembler: Assembler for combining slot outputs
        config: Run configuration
        call_llm: Function to call LLM for text responses
        call_llm_json: Function to call LLM for JSON responses
        log_event: Optional logging callback

    Returns:
        The root task's output string
    """
    worker = WorkerAgent(config)
    researcher = ResearchAgent(config)

    def _log(event: dict) -> None:
        if log_event:
            log_event({"run_id": run_id, **event})

    async def process_task(task_id: str) -> None:
        """Process a single task - either execute or assemble."""
        task = await store.get_definition(run_id, task_id)
        if not task:
            _log({"type": "error", "message": f"Task {task_id} not found"})
            return

        if task.status == "completed":
            return  # Already done

        if task.is_leaf:
            await execute_leaf(task)
        else:
            # Check if we need to spawn children first
            children_ids = await store.get_children_ids(run_id, task_id)
            if not children_ids:
                # Spawn children from template
                await spawn_children(task)
            elif await store.all_children_completed(run_id, task_id):
                # All children done - time to assemble
                await assemble_parent(task)

    async def execute_leaf(task: TaskDefinition) -> None:
        """Execute a leaf task - it only sees its question."""
        _log({"type": "leaf_start", "task_id": task.task_id, "question": task.question[:100]})

        await store.set_status(run_id, task.task_id, "running")

        # Worker sees ONLY the question - no lineage, no context accumulation
        try:
            # Use research agent if question suggests web research needed
            needs_research = any(kw in task.question.lower() for kw in [
                "find", "search", "list", "what are", "who are", "recent", "latest"
            ])

            if needs_research:
                result = await researcher.run(
                    task_id=task.task_id,
                    parent_task_id=task.parent_task_id,
                    topic=task.question,
                    context="",  # No context - leaf sees only its question
                    iteration=0,
                    log_event=lambda e: _log({"task_id": task.task_id, **e}),
                )
            else:
                result = await worker.run(
                    task_id=task.task_id,
                    parent_task_id=task.parent_task_id,
                    question=task.question,
                    context="",  # No context - leaf sees only its question
                    iteration=0,
                    use_tools=True,
                    temperature=0.7,
                    log_event=lambda e: _log({"task_id": task.task_id, **e}),
                )

            output = TaskOutput(
                task_id=task.task_id,
                output=result.output,
                sources=result.sources,
                confidence=result.confidence,
            )

        except Exception as e:
            import traceback
            _log({"type": "leaf_error", "task_id": task.task_id, "error": str(e), "traceback": traceback.format_exc()})
            output = TaskOutput(
                task_id=task.task_id,
                output=f"(Error executing task: {e})",
                sources=[],
                confidence=0.0,
            )

        # Always mark task as completed (even on error) to prevent infinite retries
        try:
            await store.save_output(run_id, output)
            await store.set_status(run_id, task.task_id, "completed")
        except Exception as store_err:
            _log({"type": "store_error", "task_id": task.task_id, "error": str(store_err)})

        _log({
            "type": "leaf_complete",
            "task_id": task.task_id,
            "slot": task.slot_name,
            "input": task.question[:500],
            "output": output.output[:1000],
            "output_len": len(output.output),
        })

        # Unblock siblings that depend on this task's output
        await unblock_dependents(task, output.output)

        # Check if parent can now assemble
        if task.parent_task_id:
            await maybe_trigger_parent(task.parent_task_id)

    async def spawn_children(parent: TaskDefinition) -> None:
        """Spawn children from parent's template."""
        if not parent.template:
            _log({"type": "error", "task_id": parent.task_id, "message": "No template to spawn children from"})
            return

        _log({"type": "spawn_children", "task_id": parent.task_id, "slots": len(parent.template)})

        await store.set_status(run_id, parent.task_id, "waiting")

        for slot_def in parent.template:
            refs = extract_slot_references(slot_def.question)

            if refs:
                # Has dependencies - start blocked
                status = "blocked"
                is_leaf = True  # Will be determined when unblocked
            else:
                # No dependencies - analyze if this should decompose further
                status = "pending"
                try:
                    is_leaf = not await should_child_decompose(slot_def.question, parent)
                except Exception as e:
                    _log({"type": "decompose_error", "slot": slot_def.slot, "error": str(e)})
                    is_leaf = True  # Default to leaf on error

            child = TaskDefinition(
                task_id=generate_task_id(),
                parent_task_id=parent.task_id,
                slot_name=slot_def.slot,
                question=slot_def.question,
                is_leaf=is_leaf,
                template=None,  # Will be set if child decomposes later
                status=status,
            )

            await store.save_definition(run_id, child)
            _log({
                "type": "child_spawned",
                "parent_id": parent.task_id,
                "child_id": child.task_id,
                "slot": slot_def.slot,
                "question": slot_def.question[:200],
                "template_question": slot_def.question,  # Full question with {slot} placeholders
                "dependencies": refs,  # List of slot names this depends on
                "status": status,
            })

    async def should_child_decompose(question: str, parent: TaskDefinition) -> bool:
        """Decide if a child should decompose further."""
        depth = get_depth(parent) + 1
        result = await decomposer.analyze(
            question=question,
            depth=depth,
            model=config.primary_model,
            call_llm_json=call_llm_json,
        )
        return result.should_decompose

    def get_depth(task: TaskDefinition) -> int:
        """Get depth of a task (0 = root)."""
        # Simple heuristic - count underscores in task_id prefix or trace parent chain
        # For now, return a reasonable default
        return 0  # TODO: Track depth in TaskDefinition

    async def unblock_dependents(completed_task: TaskDefinition, output: str) -> None:
        """When a task completes, resolve references in blocked siblings."""
        if not completed_task.parent_task_id:
            return

        siblings = await store.get_siblings(run_id, completed_task.task_id)

        for sibling in siblings:
            if sibling.status != "blocked":
                continue

            placeholder = f"{{{completed_task.slot_name}}}"
            if placeholder not in sibling.question:
                continue

            # Replace reference with actual output
            resolved_question = sibling.question.replace(placeholder, output)
            sibling.question = resolved_question

            # Check if all references now resolved
            remaining_refs = extract_slot_references(resolved_question)
            if not remaining_refs:
                sibling.status = "pending"
                _log({
                    "type": "sibling_unblocked",
                    "task_id": sibling.task_id,
                    "slot": sibling.slot_name,
                    "resolved_question": resolved_question[:200],
                    "unblocked_by": completed_task.slot_name,
                })

            await store.update_definition(run_id, sibling)

    async def assemble_parent(parent: TaskDefinition) -> None:
        """Assemble children outputs into parent answer."""
        _log({"type": "assemble_start", "task_id": parent.task_id})

        await store.set_status(run_id, parent.task_id, "running")

        # Pull only direct children's outputs - bounded context
        slot_outputs = await store.get_children_outputs(run_id, parent.task_id)
        slot_texts = {slot: out.output for slot, out in slot_outputs.items()}

        # Collect sources from children
        all_sources = []
        for out in slot_outputs.values():
            all_sources.extend(out.sources)

        # Assemble using LLM
        assembled = await assembler.assemble(
            parent_question=parent.question,
            slot_outputs=slot_texts,
            model=config.primary_model,
            call_llm=call_llm,
            sources=all_sources[:10] if all_sources else None,
        )

        # Calculate average confidence
        confidences = [out.confidence for out in slot_outputs.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        output = TaskOutput(
            task_id=parent.task_id,
            output=assembled,
            sources=all_sources[:20],
            confidence=avg_confidence,
        )

        await store.save_output(run_id, output)
        await store.set_status(run_id, parent.task_id, "completed")

        _log({
            "type": "assemble_complete",
            "task_id": parent.task_id,
            "slot": parent.slot_name,
            "input_slots": list(slot_texts.keys()),
            "inputs": {k: v[:300] for k, v in slot_texts.items()},
            "output": assembled[:1000],
            "output_len": len(assembled),
        })

        # Check if grandparent can now assemble
        if parent.parent_task_id:
            await maybe_trigger_parent(parent.parent_task_id)

    async def maybe_trigger_parent(parent_id: str) -> None:
        """Check if parent is ready to assemble after a child completes."""
        if await store.all_children_completed(run_id, parent_id):
            parent = await store.get_definition(run_id, parent_id)
            if parent and parent.status == "waiting":
                await assemble_parent(parent)

    # Main execution loop
    _log({"type": "tree_start", "root_task_id": root_task_id})

    max_iterations = 1000  # Safety limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Check if root is done
        root_status = await store.get_status(run_id, root_task_id)
        if root_status == "completed":
            break

        # Find pending tasks
        pending = await store.get_pending_tasks(run_id)
        waiting = await store.get_waiting_tasks(run_id)

        if not pending and not waiting:
            # Nothing to do - might be blocked or error
            _log({"type": "tree_stalled", "iteration": iteration})
            break

        # Process pending tasks
        if pending:
            # Process one task at a time for now (could parallelize)
            await process_task(pending[0])
        else:
            # Check waiting tasks to see if any are ready
            for task_id in waiting:
                if await store.all_children_completed(run_id, task_id):
                    await process_task(task_id)
                    break

    # Get final output
    root_output = await store.get_output(run_id, root_task_id)
    if root_output:
        _log({"type": "tree_complete", "output_len": len(root_output.output)})
        return root_output.output

    _log({"type": "tree_error", "message": "Root task has no output"})
    return "(Error: Root task did not complete)"


async def create_root_task(
    run_id: str,
    problem: str,
    store: RedisStateStore | InMemoryStateStore,
    decomposer: Decomposer,
    config: RunConfig,
    call_llm_json: Callable,
) -> str:
    """
    Create the root task and decompose if needed.

    Returns the root task ID.
    """
    root_id = generate_task_id()

    # Decide if root should decompose
    result = await decomposer.analyze(
        question=problem,
        depth=0,
        model=config.primary_model,
        call_llm_json=call_llm_json,
    )

    root = TaskDefinition(
        task_id=root_id,
        parent_task_id=None,
        slot_name="root",
        question=problem,
        is_leaf=not result.should_decompose,
        template=result.slots if result.should_decompose else None,
        status="pending",
    )

    await store.save_definition(run_id, root)

    return root_id
