from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import requests
from config.prompts import (
    EVALUATOR_PROMPT,
    METADATA_EXTRACTION_PROMPT,
    ORCHESTRATOR_DECOMPOSE_PROMPT,
    RESEARCH_PROMPT,
    SYNTHESIZER_PROMPT,
    WORKER_PROMPT,
)
from memory.store import CritiqueReport, SynthesisVersion, TaskOutput
from tools.file_ops import read_file, write_file, search_replace
from tools.run_command import run_command
from tools.search import search_and_fetch, web_search, fetch_page

from .llm import call_agent_with_tools, call_llm, call_llm_json


def _now() -> float:
    return time.time()


def _mkid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class RunConfig:
    primary_model: str
    fast_model: str
    tavily_api_key: str | None
    max_parallel_agents: int
    max_total_tasks: int
    max_task_depth: int
    min_score_improvement: float
    target_score: float


class OrchestratorAgent:
    async def decompose(self, problem: str, iteration: int, critique: Optional[dict], model: str) -> list[dict]:
        critique_txt = ""
        if critique:
            critique_txt = f"\n\nCritique from previous iteration:\n{critique}"
        user = f"Problem:\n{problem}\n\nIteration: {iteration}{critique_txt}\n\nReturn tasks JSON now."
        data = await call_llm_json(
            messages=[{"role": "system", "content": ORCHESTRATOR_DECOMPOSE_PROMPT}, {"role": "user", "content": user}],
            model=model,
            temperature=0.3,
            max_tokens=900,
        )
        tasks = []
        if isinstance(data, dict) and isinstance(data.get("tasks"), list):
            tasks = data["tasks"]
        elif isinstance(data, list):
            tasks = data
        # Normalize + cap
        max_top = int(os.getenv("MAX_TOP_LEVEL_TASKS", "10") or 10)
        max_top = max(3, min(50, max_top))
        out: list[dict] = []
        for i, t in enumerate(tasks[:max_top]):
            if not isinstance(t, dict):
                continue
            out.append(
                {
                    "id": t.get("id") or f"task_{iteration}_{i+1}",
                    "question": t.get("question") or "",
                    "agent_type": t.get("agent_type") or "worker",
                    "rationale": t.get("rationale") or "",
                    "requires_web_search": bool(t.get("requires_web_search", False)),
                }
            )
        if not out:
            out = [
                {
                    "id": f"task_{iteration}_1",
                    "question": problem,
                    "agent_type": "worker",
                    "rationale": "Fallback: solve directly.",
                    "requires_web_search": False,
                }
            ]
        return out

    async def plan_synthesis(self, problem: str, task_outputs: list[TaskOutput], model: str) -> str:
        bullets = "\n".join([f"- {o.task_id}: {o.output[:600]}" for o in task_outputs])
        user = f"Problem:\n{problem}\n\nTask outputs:\n{bullets}\n\nWrite synthesis plan/instructions."
        return await call_llm(
            messages=[{"role": "system", "content": "Write synthesis instructions."}, {"role": "user", "content": user}],
            model=model,
            temperature=0.4,
            max_tokens=500,
        )


class WorkerAgent:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    async def run(
        self,
        task_id: str,
        parent_task_id: Optional[str],
        question: str,
        context: str,
        iteration: int,
        use_tools: bool,
        temperature: float,
        llm_timeout_s: int = 1800,  # Increased for complex tasks
        log_event=None,
    ) -> TaskOutput:
        tools = {}
        if use_tools:
            tools = {
                "web_search": lambda inp: web_search(
                    query=str(inp.get("query", "")),
                    max_results=int(inp.get("max_results", 5)),
                    tavily_api_key=self.cfg.tavily_api_key,
                ),
                "fetch_page": lambda inp: fetch_page(url=str(inp.get("url", ""))),
                "search_and_fetch": lambda inp: search_and_fetch(
                    query=str(inp.get("query", "")),
                    max_results=int(inp.get("max_results", 5)),
                    tavily_api_key=self.cfg.tavily_api_key,
                ),
                "run_command": lambda inp: run_command(
                    command=str(inp.get("command", "")),
                    timeout_s=int(inp.get("timeout_s", 60)),
                    cwd=inp.get("cwd") or None,
                ),
                "read_file": lambda inp: read_file(path=str(inp.get("path", ""))),
                "write_file": lambda inp: write_file(
                    path=str(inp.get("path", "")),
                    content=str(inp.get("content", "")),
                ),
                "search_replace": lambda inp: search_replace(
                    path=str(inp.get("path", "")),
                    old_string=str(inp.get("old_string", "")),
                    new_string=str(inp.get("new_string", "")),
                ),
            }

        user_prompt = f"Question:\n{question}\n\nContext:\n{context}\n\nReturn tool JSON actions or final_answer."
        answer = await call_agent_with_tools(
            system_prompt=WORKER_PROMPT,
            user_prompt=user_prompt,
            model=self.cfg.primary_model,
            tools=tools,
            temperature=temperature,
            max_tool_iterations=6,
            timeout_s=llm_timeout_s,
            log_event=log_event,
        )

        meta = await call_llm_json(
            messages=[
                {"role": "system", "content": METADATA_EXTRACTION_PROMPT},
                {"role": "user", "content": answer},
            ],
            model=self.cfg.fast_model,
            temperature=0.2,
            max_tokens=500,
            timeout_s=llm_timeout_s,
        )
        meta = meta if isinstance(meta, dict) else {}
        return TaskOutput(
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_role=f"worker_iteration_{iteration}",
            output=answer,
            confidence=float(meta.get("confidence", 0.5) or 0.5),
            sources=list(meta.get("sources", []) or []),
            assumptions_made=list(meta.get("assumptions_made", []) or []),
            open_questions=list(meta.get("open_questions", []) or []),
            suggested_followups=list(meta.get("suggested_followups", []) or []),
            iteration=iteration,
            timestamp=_now(),
        )


class ResearchAgent:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    async def run(
        self,
        task_id: str,
        parent_task_id: Optional[str],
        topic: str,
        context: str,
        iteration: int,
        llm_timeout_s: int = 1800,  # Increased for complex tasks
        log_event=None,
    ) -> TaskOutput:
        tools = {
            "web_search": lambda inp: web_search(
                query=str(inp.get("query", "")),
                max_results=int(inp.get("max_results", 5)),
                tavily_api_key=self.cfg.tavily_api_key,
            ),
            "fetch_page": lambda inp: fetch_page(url=str(inp.get("url", ""))),
            "search_and_fetch": lambda inp: search_and_fetch(
                query=str(inp.get("query", "")),
                max_results=int(inp.get("max_results", 5)),
                tavily_api_key=self.cfg.tavily_api_key,
                max_concurrency=6,
            ),
            "run_command": lambda inp: run_command(
                command=str(inp.get("command", "")),
                timeout_s=int(inp.get("timeout_s", 60)),
                cwd=inp.get("cwd") or None,
            ),
            "read_file": lambda inp: read_file(path=str(inp.get("path", ""))),
            "write_file": lambda inp: write_file(
                path=str(inp.get("path", "")),
                content=str(inp.get("content", "")),
            ),
            "search_replace": lambda inp: search_replace(
                path=str(inp.get("path", "")),
                old_string=str(inp.get("old_string", "")),
                new_string=str(inp.get("new_string", "")),
            ),
        }

        user_prompt = f"Research topic:\n{topic}\n\nContext:\n{context}\n\nReturn tool JSON actions or final_answer."
        answer = await call_agent_with_tools(
            system_prompt=RESEARCH_PROMPT,
            user_prompt=user_prompt,
            model=self.cfg.primary_model,
            tools=tools,
            temperature=0.4,
            max_tool_iterations=8,
            timeout_s=llm_timeout_s,
            log_event=log_event,
        )

        meta: dict | None = None
        for model in (self.cfg.fast_model, self.cfg.primary_model):
            try:
                meta = await call_llm_json(
                    messages=[
                        {"role": "system", "content": METADATA_EXTRACTION_PROMPT},
                        {"role": "user", "content": answer},
                    ],
                    model=model,
                    temperature=0.2,
                    max_tokens=500,
                    timeout_s=llm_timeout_s,
                )
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    continue  # model not found, try primary
                raise
        meta = meta if isinstance(meta, dict) else {}
        return TaskOutput(
            task_id=task_id,
            parent_task_id=parent_task_id,
            agent_role=f"research_iteration_{iteration}",
            output=answer,
            confidence=float(meta.get("confidence", 0.6) or 0.6),
            sources=list(meta.get("sources", []) or []),
            assumptions_made=list(meta.get("assumptions_made", []) or []),
            open_questions=list(meta.get("open_questions", []) or []),
            suggested_followups=list(meta.get("suggested_followups", []) or []),
            iteration=iteration,
            timestamp=_now(),
        )


class SynthesizerAgent:
    async def synthesize(
        self,
        problem: str,
        task_outputs: list[TaskOutput],
        iteration: int,
        previous_synthesis: Optional[str],
        critique: Optional[dict],
        all_previous_versions: list[SynthesisVersion],
        model: str,
    ) -> SynthesisVersion:
        chunks = []
        for o in task_outputs:
            srcs = f"\nSources: {', '.join(o.sources[:8])}" if o.sources else ""
            chunks.append(f"=== {o.task_id} ({o.agent_role}, conf={o.confidence}) ===\n{o.output}{srcs}\n")
        crit = f"\n\nCritique to address:\n{critique}" if critique else ""
        prev = f"\n\nPrevious synthesis:\n{previous_synthesis}" if previous_synthesis else ""
        user = f"Problem:\n{problem}\n\nAgent outputs:\n{''.join(chunks)}{prev}{crit}\n\nWrite the best synthesis."
        content = await call_llm(
            messages=[{"role": "system", "content": SYNTHESIZER_PROMPT}, {"role": "user", "content": user}],
            model=model,
            temperature=0.5,
            max_tokens=3500,
        )
        return SynthesisVersion(version_id=_mkid(f"v{iteration}"), iteration=iteration, content=content, timestamp=_now())

    async def final_synthesis(self, problem: str, all_versions: list[SynthesisVersion], model: str) -> str:
        top = sorted(all_versions, key=lambda v: v.score, reverse=True)[:3]
        payload = "\n\n".join([f"--- {v.version_id} score={v.score} ---\n{v.content}" for v in top])
        user = f"Problem:\n{problem}\n\nTop versions:\n{payload}\n\nCreate the definitive final answer."
        return await call_llm(
            messages=[
                {"role": "system", "content": SYNTHESIZER_PROMPT + "\n\nMerge the top versions into one definitive answer. Preserve all cited sources and numbers. Include a Sources & References section."},
                {"role": "user", "content": user},
            ],
            model=model,
            temperature=0.4,
            max_tokens=4000,
        )


class EvaluatorAgent:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def _check_diminishing(self, score: float, previous_scores: list[float]) -> bool:
        if len(previous_scores) >= 1:
            if score - previous_scores[-1] < self.cfg.min_score_improvement:
                return True
        if score >= self.cfg.target_score:
            return True
        return False

    async def evaluate(
        self, problem: str, synthesis: str, iteration: int, previous_scores: list[float], model: str
    ) -> CritiqueReport:
        user = f"Problem:\n{problem}\n\nCandidate answer:\n{synthesis}\n\nReturn evaluation JSON now."
        data = await call_llm_json(
            messages=[{"role": "system", "content": EVALUATOR_PROMPT}, {"role": "user", "content": user}],
            model=model,
            temperature=0.2,
            max_tokens=900,
        )
        data = data if isinstance(data, dict) else {}
        score = float(data.get("overall_score", 0.0) or 0.0)
        diminishing = self._check_diminishing(score, previous_scores)
        return CritiqueReport(
            version=f"v{iteration}",
            overall_score=score,
            iteration=iteration,
            weaknesses=list(data.get("weaknesses", []) or []),
            strengths=list(data.get("strengths", []) or []),
            suggested_fixes=list(data.get("suggested_fixes", []) or []),
            diminishing_returns_signal=diminishing,
            timestamp=_now(),
        )


async def run_task_tree(
    cfg: RunConfig,
    tasks: list[dict],
    problem: str,
    iteration: int,
    critique: Optional[dict],
    log_event=None,
    deadline_ts: float | None = None,
    spawn_cutoff_ts: float | None = None,
) -> list[TaskOutput]:
    """
    Parallelism strategy:
    - Web I/O runs truly parallel via asyncio.
    - LLM calls are concurrent too; for Ollama they will effectively queue server-side, for Groq they parallelize.
    - A semaphore limits overall concurrency.

    Task tree strategy:
    - Start with orchestrator tasks (depth=1).
    - After each task completes, if it has open_questions, enqueue child tasks (depth+1).
    - Continue until caps hit (max_total_tasks, max_task_depth) or queue drains.
    """
    sem = asyncio.Semaphore(max(1, cfg.max_parallel_agents))
    worker = WorkerAgent(cfg)
    researcher = ResearchAgent(cfg)

    def _normalize_task(t: dict, default_depth: int = 1) -> dict:
        t = dict(t)
        t.setdefault("depth", default_depth)
        t.setdefault("parent_task_id", None)
        t.setdefault("requires_web_search", bool(t.get("requires_web_search", False)))
        t.setdefault("agent_type", t.get("agent_type", "worker"))
        return t

    queue: list[dict] = [_normalize_task(t, 1) for t in tasks]
    outputs: list[TaskOutput] = []
    seen_questions: set[str] = set()

    async def _run_one(t: dict) -> TaskOutput:
        async with sem:
            tid = str(t.get("id", _mkid("task")))
            q = str(t.get("question", "")).strip()
            atype = str(t.get("agent_type", "worker"))
            use_tools = bool(t.get("requires_web_search", False))
            parent_task_id = t.get("parent_task_id")
            depth = int(t.get("depth", 1))
            ctx = f"Overall problem:\n{problem}\n\nIteration: {iteration}\nCritique: {critique}"

            def _log(e: dict) -> None:
                if log_event:
                    log_event({"task_id": tid, "agent_type": atype, "depth": depth, "parent_task_id": parent_task_id, **e})

            if atype == "research":
                return await researcher.run(
                    task_id=tid,
                    parent_task_id=parent_task_id,
                    topic=q,
                    context=ctx,
                    iteration=iteration,
                    llm_timeout_s=_llm_timeout_from_deadline(deadline_ts),
                    log_event=_log,
                )
            return await worker.run(
                task_id=tid,
                parent_task_id=parent_task_id,
                question=q,
                context=ctx,
                iteration=iteration,
                use_tools=use_tools,
                temperature=0.7,
                llm_timeout_s=_llm_timeout_from_deadline(deadline_ts),
                log_event=_log,
            )

    def _llm_timeout_from_deadline(dl: float | None) -> int:
        """
        Prevent a single LLM call from running past the iteration budget.
        We keep a small buffer so synthesis/eval can still happen.
        """
        if dl is None:
            return 1800  # Increased default timeout to 30 minutes for complex tasks
        remaining = max(0.0, float(dl) - time.time())
        # Reserve ~30s for graph overhead; allow up to 30 minutes for complex tasks.
        return int(max(45.0, min(1800.0, remaining - 30.0)))

    async def _maybe_enqueue_children(parent: TaskOutput, depth: int) -> None:
        if depth >= cfg.max_task_depth:
            return
        if len(outputs) + len(queue) >= cfg.max_total_tasks:
            return
        # Spawn up to 2 per parent, prioritize first items (model already ordered)
        for oq in parent.open_questions[:2]:
            q = str(oq).strip()
            if not q:
                continue
            key = q.lower()
            if key in seen_questions:
                continue
            seen_questions.add(key)
            queue.append(
                _normalize_task(
                    {
                        "id": _mkid("sub"),
                        "question": q,
                        "agent_type": "worker",
                        "rationale": f"Follow-up from {parent.task_id}",
                        "requires_web_search": True,
                        "parent_task_id": parent.task_id,
                        "depth": depth + 1,
                    },
                    depth + 1,
                )
            )
            if len(outputs) + len(queue) >= cfg.max_total_tasks:
                return

    # Run tasks in waves up to concurrency until queue drains or caps reached
    def _time_up() -> bool:
        return bool(deadline_ts is not None and time.time() >= float(deadline_ts))

    def _spawn_cutoff() -> bool:
        return bool(spawn_cutoff_ts is not None and time.time() >= float(spawn_cutoff_ts))

    while queue and len(outputs) < cfg.max_total_tasks and not _time_up() and not _spawn_cutoff():
        wave: list[dict] = []
        while (
            queue
            and len(wave) < cfg.max_parallel_agents
            and (len(outputs) + len(wave) < cfg.max_total_tasks)
            and not _time_up()
            and not _spawn_cutoff()
        ):
            wave.append(queue.pop(0))
        if not wave:
            break
        wave_out = await asyncio.gather(*[_run_one(t) for t in wave])
        outputs.extend(wave_out)
        for t, o in zip(wave, wave_out):
            depth = int(t.get("depth", 1))
            # If we're in wrap-up window, don't expand the tree further.
            if not _spawn_cutoff():
                await _maybe_enqueue_children(o, depth)

    if log_event:
        reason = "queue_drained"
        if _time_up():
            reason = "deadline"
        elif _spawn_cutoff():
            reason = "wrapup_buffer"
        elif len(outputs) >= cfg.max_total_tasks:
            reason = "max_total_tasks"
        log_event(
            {
                "type": "task_tree_stop",
                "iteration": iteration,
                "reason": reason,
                "outputs": len(outputs),
                "queued_remaining": len(queue),
                "deadline_ts": deadline_ts,
                "spawn_cutoff_ts": spawn_cutoff_ts,
            }
        )

    return outputs


