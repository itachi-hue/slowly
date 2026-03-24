"""
Assembler: Weaves slot outputs into a coherent parent answer.

The assembler takes a parent's question and the outputs from its direct children,
then uses an LLM to combine them into a single coherent answer.

Context is always bounded - assembler only sees direct children, never grandchildren.
"""

from __future__ import annotations

from typing import Optional


ASSEMBLER_PROMPT = """You are an assembler that combines slot outputs into a coherent answer.

You will receive:
1. The parent question that needs answering
2. Named slot outputs from child tasks

Your job:
- Weave the slot outputs into a single, coherent answer to the parent question
- Maintain all factual content from the slots
- Add transitions and connections where needed
- Structure the answer clearly
- Do NOT add information that isn't in the slots
- Do NOT omit important information from the slots

Output a well-structured answer that directly addresses the parent question.
"""


class Assembler:
    """Assembles slot outputs into coherent parent answers."""

    async def assemble(
        self,
        parent_question: str,
        slot_outputs: dict[str, str],  # {slot_name: output_text}
        model: str,
        call_llm,  # Function to call LLM and get text response
        sources: Optional[list[str]] = None,
    ) -> str:
        """
        Assemble slot outputs into a coherent answer.

        Args:
            parent_question: The question the parent task needs to answer
            slot_outputs: Dict mapping slot names to their output text
            model: Model to use for assembly
            call_llm: Async function to call LLM and get response
            sources: Optional list of source URLs to include

        Returns:
            Assembled answer string
        """
        if not slot_outputs:
            return "(No slot outputs to assemble)"

        # Format slot outputs for the prompt
        slots_text = []
        for slot_name, output in slot_outputs.items():
            slots_text.append(f"=== SLOT: {slot_name} ===\n{output}\n")

        user_prompt = f"""Parent Question:
{parent_question}

Slot Outputs:
{''.join(slots_text)}

Assemble these slot outputs into a coherent answer to the parent question.
"""

        try:
            assembled = await call_llm(
                messages=[
                    {"role": "system", "content": ASSEMBLER_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                temperature=0.4,
                max_tokens=3000,
            )
        except Exception as e:
            # Fallback: just concatenate slots
            assembled = f"(Assembly error: {e})\n\n"
            assembled += "\n\n".join(
                f"**{slot}**:\n{output}"
                for slot, output in slot_outputs.items()
            )

        # Append sources if provided
        if sources:
            assembled += "\n\n## Sources\n"
            assembled += "\n".join(f"- {url}" for url in sources[:10])

        return assembled

    async def simple_assemble(
        self,
        parent_question: str,
        slot_outputs: dict[str, str],
    ) -> str:
        """
        Simple non-LLM assembly - just formats and concatenates.

        Use this when LLM assembly isn't needed or available.
        """
        if not slot_outputs:
            return "(No slot outputs)"

        parts = [f"# {parent_question}\n"]

        for slot_name, output in slot_outputs.items():
            # Convert slot name to title case header
            header = slot_name.replace("_", " ").title()
            parts.append(f"## {header}\n{output}\n")

        return "\n".join(parts)


def create_assembler() -> Assembler:
    """Factory function to create assembler."""
    return Assembler()
