"""
Redis-based state store for task tree execution.

Stores task definitions and outputs in Redis, enabling:
- Agents to write outputs without parent context accumulation
- Bottom-up assembly by pulling children outputs from Redis
- Bounded context at every level of the tree

Falls back to in-memory store if Redis unavailable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class SlotDef:
    """Definition of a slot in a parent's template."""
    slot: str
    question: str  # May contain {other_slot} references


@dataclass
class TaskDefinition:
    """Definition of a task in the tree."""
    task_id: str
    parent_task_id: Optional[str]
    slot_name: str  # Which slot this fills in parent's template
    question: str  # The resolved question (references filled in)
    is_leaf: bool  # True = execute directly, False = decompose further
    template: Optional[list[SlotDef]] = None  # If not leaf: ordered slots
    status: Literal["pending", "blocked", "running", "waiting", "completed"] = "pending"


@dataclass
class TaskOutput:
    """Output from a completed task."""
    task_id: str
    output: str  # The answer (executed or assembled)
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.5


def extract_slot_references(question: str) -> list[str]:
    """Extract {slot_name} references from a question string."""
    return re.findall(r'\{(\w+)\}', question)


class RedisStateStore:
    """Async Redis-backed state store for task tree execution."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "slowly",
        ttl_seconds: int = 72 * 3600,  # 72 hours default
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self._client: Optional[Any] = None

    async def connect(self) -> bool:
        """Connect to Redis. Returns True if successful."""
        if not REDIS_AVAILABLE:
            return False
        try:
            self._client = aioredis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
            await self._client.ping()
            return True
        except Exception:
            self._client = None
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    def _key(self, *parts: str) -> str:
        """Build a Redis key with prefix."""
        return f"{self.key_prefix}:{':'.join(parts)}"

    # --- Task Definition Operations ---

    async def save_definition(self, run_id: str, task: TaskDefinition) -> None:
        """Save a task definition to Redis."""
        key = self._key("def", run_id, task.task_id)
        data = {
            "task_id": task.task_id,
            "parent_task_id": task.parent_task_id or "",
            "slot_name": task.slot_name,
            "question": task.question,
            "is_leaf": "1" if task.is_leaf else "0",
            "template": json.dumps([{"slot": s.slot, "question": s.question} for s in task.template]) if task.template else "",
            "status": task.status,
        }
        await self._client.hset(key, mapping=data)
        await self._client.expire(key, self.ttl_seconds)

        # Track in parent's children set
        if task.parent_task_id:
            children_key = self._key("children", run_id, task.parent_task_id)
            await self._client.sadd(children_key, task.task_id)
            await self._client.expire(children_key, self.ttl_seconds)

        # Track task's slot name for assembly lookup
        if task.parent_task_id:
            slot_key = self._key("slot", run_id, task.task_id)
            await self._client.set(slot_key, task.slot_name)
            await self._client.expire(slot_key, self.ttl_seconds)

    async def get_definition(self, run_id: str, task_id: str) -> Optional[TaskDefinition]:
        """Get a task definition from Redis."""
        key = self._key("def", run_id, task_id)
        data = await self._client.hgetall(key)
        if not data:
            return None

        template = None
        if data.get("template"):
            template_data = json.loads(data["template"])
            template = [SlotDef(slot=s["slot"], question=s["question"]) for s in template_data]

        return TaskDefinition(
            task_id=data["task_id"],
            parent_task_id=data["parent_task_id"] or None,
            slot_name=data["slot_name"],
            question=data["question"],
            is_leaf=data["is_leaf"] == "1",
            template=template,
            status=data["status"],
        )

    async def update_definition(self, run_id: str, task: TaskDefinition) -> None:
        """Update an existing task definition."""
        await self.save_definition(run_id, task)

    # --- Task Output Operations ---

    async def save_output(self, run_id: str, output: TaskOutput) -> None:
        """Save a task output to Redis."""
        key = self._key("out", run_id, output.task_id)
        data = {
            "task_id": output.task_id,
            "output": output.output,
            "sources": json.dumps(output.sources),
            "confidence": str(output.confidence),
        }
        await self._client.hset(key, mapping=data)
        await self._client.expire(key, self.ttl_seconds)

    async def get_output(self, run_id: str, task_id: str) -> Optional[TaskOutput]:
        """Get a task output from Redis."""
        key = self._key("out", run_id, task_id)
        data = await self._client.hgetall(key)
        if not data:
            return None

        return TaskOutput(
            task_id=data["task_id"],
            output=data["output"],
            sources=json.loads(data.get("sources", "[]")),
            confidence=float(data.get("confidence", "0.5")),
        )

    # --- Tree Structure Operations ---

    async def get_children_ids(self, run_id: str, parent_id: str) -> list[str]:
        """Get IDs of all children of a task."""
        key = self._key("children", run_id, parent_id)
        children = await self._client.smembers(key)
        return list(children) if children else []

    async def get_children_outputs(self, run_id: str, parent_id: str) -> dict[str, TaskOutput]:
        """Get all children outputs keyed by slot name."""
        children_ids = await self.get_children_ids(run_id, parent_id)
        result = {}

        for child_id in children_ids:
            # Get slot name
            slot_key = self._key("slot", run_id, child_id)
            slot_name = await self._client.get(slot_key)
            if not slot_name:
                continue

            # Get output
            output = await self.get_output(run_id, child_id)
            if output:
                result[slot_name] = output

        return result

    async def get_siblings(self, run_id: str, task_id: str) -> list[TaskDefinition]:
        """Get all sibling tasks (same parent)."""
        task = await self.get_definition(run_id, task_id)
        if not task or not task.parent_task_id:
            return []

        sibling_ids = await self.get_children_ids(run_id, task.parent_task_id)
        siblings = []
        for sid in sibling_ids:
            if sid != task_id:
                sibling = await self.get_definition(run_id, sid)
                if sibling:
                    siblings.append(sibling)
        return siblings

    # --- Status Operations ---

    async def set_status(self, run_id: str, task_id: str, status: str) -> None:
        """Update a task's status."""
        key = self._key("def", run_id, task_id)
        await self._client.hset(key, "status", status)

    async def get_status(self, run_id: str, task_id: str) -> Optional[str]:
        """Get a task's status."""
        key = self._key("def", run_id, task_id)
        return await self._client.hget(key, "status")

    async def all_children_completed(self, run_id: str, parent_id: str) -> bool:
        """Check if all children of a task are completed."""
        children_ids = await self.get_children_ids(run_id, parent_id)
        if not children_ids:
            return False  # No children means not ready to assemble

        for child_id in children_ids:
            status = await self.get_status(run_id, child_id)
            if status != "completed":
                return False
        return True

    async def get_pending_tasks(self, run_id: str) -> list[str]:
        """Get all pending task IDs for a run."""
        # Scan for all task definitions
        pattern = self._key("def", run_id, "*")
        pending = []

        async for key in self._client.scan_iter(match=pattern):
            status = await self._client.hget(key, "status")
            if status == "pending":
                task_id = await self._client.hget(key, "task_id")
                if task_id:
                    pending.append(task_id)

        return pending

    async def get_waiting_tasks(self, run_id: str) -> list[str]:
        """Get all waiting (parent) task IDs for a run."""
        pattern = self._key("def", run_id, "*")
        waiting = []

        async for key in self._client.scan_iter(match=pattern):
            status = await self._client.hget(key, "status")
            if status == "waiting":
                task_id = await self._client.hget(key, "task_id")
                if task_id:
                    waiting.append(task_id)

        return waiting


class InMemoryStateStore:
    """In-memory fallback store with same interface as RedisStateStore."""

    def __init__(self):
        self._definitions: dict[str, dict[str, TaskDefinition]] = {}  # run_id -> task_id -> def
        self._outputs: dict[str, dict[str, TaskOutput]] = {}  # run_id -> task_id -> output
        self._children: dict[str, dict[str, set[str]]] = {}  # run_id -> parent_id -> {child_ids}
        self._slots: dict[str, dict[str, str]] = {}  # run_id -> task_id -> slot_name

    async def connect(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    def _ensure_run(self, run_id: str) -> None:
        if run_id not in self._definitions:
            self._definitions[run_id] = {}
            self._outputs[run_id] = {}
            self._children[run_id] = {}
            self._slots[run_id] = {}

    async def save_definition(self, run_id: str, task: TaskDefinition) -> None:
        self._ensure_run(run_id)
        self._definitions[run_id][task.task_id] = task

        if task.parent_task_id:
            if task.parent_task_id not in self._children[run_id]:
                self._children[run_id][task.parent_task_id] = set()
            self._children[run_id][task.parent_task_id].add(task.task_id)
            self._slots[run_id][task.task_id] = task.slot_name

    async def get_definition(self, run_id: str, task_id: str) -> Optional[TaskDefinition]:
        self._ensure_run(run_id)
        return self._definitions[run_id].get(task_id)

    async def update_definition(self, run_id: str, task: TaskDefinition) -> None:
        await self.save_definition(run_id, task)

    async def save_output(self, run_id: str, output: TaskOutput) -> None:
        self._ensure_run(run_id)
        self._outputs[run_id][output.task_id] = output

    async def get_output(self, run_id: str, task_id: str) -> Optional[TaskOutput]:
        self._ensure_run(run_id)
        return self._outputs[run_id].get(task_id)

    async def get_children_ids(self, run_id: str, parent_id: str) -> list[str]:
        self._ensure_run(run_id)
        children = self._children[run_id].get(parent_id, set())
        return list(children)

    async def get_children_outputs(self, run_id: str, parent_id: str) -> dict[str, TaskOutput]:
        self._ensure_run(run_id)
        children_ids = await self.get_children_ids(run_id, parent_id)
        result = {}

        for child_id in children_ids:
            slot_name = self._slots[run_id].get(child_id)
            if not slot_name:
                continue
            output = self._outputs[run_id].get(child_id)
            if output:
                result[slot_name] = output

        return result

    async def get_siblings(self, run_id: str, task_id: str) -> list[TaskDefinition]:
        task = await self.get_definition(run_id, task_id)
        if not task or not task.parent_task_id:
            return []

        sibling_ids = await self.get_children_ids(run_id, task.parent_task_id)
        siblings = []
        for sid in sibling_ids:
            if sid != task_id:
                sibling = await self.get_definition(run_id, sid)
                if sibling:
                    siblings.append(sibling)
        return siblings

    async def set_status(self, run_id: str, task_id: str, status: str) -> None:
        self._ensure_run(run_id)
        task = self._definitions[run_id].get(task_id)
        if task:
            task.status = status

    async def get_status(self, run_id: str, task_id: str) -> Optional[str]:
        self._ensure_run(run_id)
        task = self._definitions[run_id].get(task_id)
        return task.status if task else None

    async def all_children_completed(self, run_id: str, parent_id: str) -> bool:
        children_ids = await self.get_children_ids(run_id, parent_id)
        if not children_ids:
            return False

        for child_id in children_ids:
            status = await self.get_status(run_id, child_id)
            if status != "completed":
                return False
        return True

    async def get_pending_tasks(self, run_id: str) -> list[str]:
        self._ensure_run(run_id)
        return [
            tid for tid, task in self._definitions[run_id].items()
            if task.status == "pending"
        ]

    async def get_waiting_tasks(self, run_id: str) -> list[str]:
        self._ensure_run(run_id)
        return [
            tid for tid, task in self._definitions[run_id].items()
            if task.status == "waiting"
        ]


async def create_state_store() -> RedisStateStore | InMemoryStateStore:
    """Factory function to create appropriate state store with fallback."""
    if os.getenv("REDIS_ENABLED", "true").lower() == "false":
        return InMemoryStateStore()

    if not REDIS_AVAILABLE:
        print("[WARNING] redis package not installed, using in-memory store")
        return InMemoryStateStore()

    store = RedisStateStore(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        password=os.getenv("REDIS_PASSWORD") or None,
        key_prefix=os.getenv("REDIS_KEY_PREFIX", "slowly"),
    )

    if await store.connect():
        return store

    print("[WARNING] Redis connection failed, using in-memory store")
    return InMemoryStateStore()
