"""
Decomposer: External orchestrator that decides whether to decompose or execute tasks,
and creates ordered templates with slot dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from memory.redis_store import SlotDef


DECOMPOSE_PROMPT = """You are a task decomposer. Given a question, decide if it should be decomposed into sub-tasks or executed directly.

If decomposing, create an ORDERED list of slots. Later slots can reference earlier slots using {slot_name} syntax.

Rules:
1. Output MUST be valid JSON (no markdown fences)
2. Slots execute in order - put independent work first
3. Use {slot_name} to reference earlier slot outputs
4. Keep slot names short and descriptive (snake_case)
5. Each slot should be a focused, answerable question
6. Aim for 2-5 slots (fewer is better if sufficient)

Return JSON:
{
  "should_decompose": true/false,
  "reason": "why decompose or execute directly",
  "slots": [
    {"slot": "slot_name", "question": "The specific question to answer"},
    {"slot": "next_slot", "question": "Question that may reference {slot_name}"}
  ]
}

If should_decompose is false, return empty slots array.
"""


@dataclass
class DecomposeResult:
    should_decompose: bool
    reason: str
    slots: list[SlotDef]


class Decomposer:
    """External orchestrator that decides decomposition strategy."""

    def __init__(self, max_depth: int = 5, min_decompose_depth: int = 0):
        """
        Args:
            max_depth: Maximum tree depth (stop decomposing beyond this)
            min_decompose_depth: Minimum depth before allowing direct execution
        """
        self.max_depth = max_depth
        self.min_decompose_depth = min_decompose_depth

    async def analyze(
        self,
        question: str,
        depth: int,
        model: str,
        call_llm_json,  # Function to call LLM and get JSON response
    ) -> DecomposeResult:
        """
        Analyze whether a question should be decomposed or executed directly.

        Args:
            question: The question to analyze
            depth: Current depth in the tree (0 = root)
            model: Model to use for analysis
            call_llm_json: Async function to call LLM and parse JSON response

        Returns:
            DecomposeResult with decomposition decision and slots
        """
        # Force leaf execution at max depth
        if depth >= self.max_depth:
            return DecomposeResult(
                should_decompose=False,
                reason=f"Max depth {self.max_depth} reached",
                slots=[],
            )

        # Ask LLM whether to decompose
        user_prompt = f"""Question to analyze:
{question}

Current depth: {depth}
Max depth: {self.max_depth}

Should this be decomposed into sub-tasks, or executed directly?
Consider:
- Simple factual questions can be executed directly
- Complex multi-part questions should be decomposed
- If depth is close to max, prefer direct execution
"""

        try:
            data = await call_llm_json(
                messages=[
                    {"role": "system", "content": DECOMPOSE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                temperature=0.3,
                max_tokens=800,
            )
        except Exception as e:
            # On error, default to direct execution
            return DecomposeResult(
                should_decompose=False,
                reason=f"LLM error, defaulting to execute: {e}",
                slots=[],
            )

        if not isinstance(data, dict):
            return DecomposeResult(
                should_decompose=False,
                reason="Invalid LLM response, defaulting to execute",
                slots=[],
            )

        should_decompose = bool(data.get("should_decompose", False))
        reason = str(data.get("reason", ""))

        slots = []
        if should_decompose and data.get("slots"):
            for s in data["slots"]:
                if isinstance(s, dict) and s.get("slot") and s.get("question"):
                    slots.append(SlotDef(
                        slot=str(s["slot"]),
                        question=str(s["question"]),
                    ))

        # If decompose was requested but no valid slots, default to execute
        if should_decompose and not slots:
            return DecomposeResult(
                should_decompose=False,
                reason="Decompose requested but no valid slots, defaulting to execute",
                slots=[],
            )

        return DecomposeResult(
            should_decompose=should_decompose,
            reason=reason,
            slots=slots,
        )

    async def should_decompose(
        self,
        question: str,
        depth: int,
        model: str,
        call_llm_json,
    ) -> bool:
        """Convenience method that just returns the boolean decision."""
        result = await self.analyze(question, depth, model, call_llm_json)
        return result.should_decompose

    async def create_template(
        self,
        question: str,
        depth: int,
        model: str,
        call_llm_json,
    ) -> list[SlotDef]:
        """Get ordered slots for decomposition."""
        result = await self.analyze(question, depth, model, call_llm_json)
        return result.slots


def create_decomposer() -> Decomposer:
    """Factory function to create decomposer from environment."""
    return Decomposer(
        max_depth=int(os.getenv("MAX_TASK_DEPTH", "5")),
        min_decompose_depth=int(os.getenv("MIN_DECOMPOSE_DEPTH", "0")),
    )
