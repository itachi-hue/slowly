from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Callable, Optional

import requests
from dotenv import load_dotenv


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None and v != "" else int(default)
    except Exception:
        return int(default)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _best_effort_json(s: str) -> Any:
    s2 = _strip_code_fences(s)
    try:
        return json.loads(s2)
    except Exception:
        # Try to salvage a JSON object/array substring
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", s2)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None


def _get_backend(model: str) -> str:
    if model and model.startswith("groq/"):
        return "groq"
    return os.getenv("ACTIVE_BACKEND", "ollama")


def _call_ollama_chat(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    base_url: str,
    timeout_s: int = 1800,  # Increased default to 30 minutes for complex tasks
) -> str:
    # Minimal HTTP client to avoid requiring specific client APIs.
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    # Ollama can take a while on first token (model load). Use a longer read timeout.
    r = requests.post(url, json=payload, timeout=(10, timeout_s))
    r.raise_for_status()
    data = r.json()
    msg = data.get("message", {}) or {}
    content = msg.get("content", "")
    return (content or "").strip()


# Groq free tier ~6000 TPM; ~4 chars/token -> cap ~18k chars to stay safe
_GROQ_MAX_CHARS = int(os.getenv("GROQ_MAX_REQUEST_CHARS", "18000") or 18000)


def _truncate_messages_for_groq(messages: list[dict], max_chars: int | None = None) -> list[dict]:
    """Trim message content so total stays under Groq TPM limit. Keeps system, first user, and most recent exchanges."""
    cap = max_chars or _GROQ_MAX_CHARS
    total = sum(len(str(m.get("content", ""))) for m in messages)
    if total <= cap:
        return messages
    out = [dict(messages[0]), dict(messages[1])] if len(messages) > 1 else [dict(messages[0])]
    used = sum(len(str(m.get("content", ""))) for m in out)
    remaining = cap - used - 100
    # Add most recent exchanges from the end (chronological order)
    tail = []
    for i in range(len(messages) - 1, 1, -1):
        if remaining <= 200:
            break
        m = dict(messages[i])
        c = str(m.get("content", ""))
        if len(c) > remaining - 100:
            c = c[: max(0, remaining - 150)] + "\n...[truncated]"
        m["content"] = c
        tail.insert(0, m)
        remaining -= len(c)
    return out + tail


def _call_groq_chat(messages: list[dict], model: str, temperature: float, max_tokens: int) -> str:
    from groq import Groq  # type: ignore

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    client = Groq(api_key=api_key)
    model_name = model.replace("groq/", "")
    messages = _truncate_messages_for_groq(messages)
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _is_transient(e: BaseException) -> bool:
    """True for errors worth retrying (404, timeout, connection, 5xx, 413 TPM)."""
    if isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
        return e.response.status_code in (404, 413, 429, 502, 503, 504)
    # Groq APIStatusError for 413 (TPM/request too large)
    if hasattr(e, "status_code") and getattr(e, "status_code") in (404, 413, 429, 502, 503, 504):
        return True
    err_str = str(e).lower()
    if "413" in err_str or "request too large" in err_str or "rate_limit" in err_str:
        return True
    return False


def _is_413_or_tpm(e: BaseException) -> bool:
    """True if error is 413 / request too large / TPM limit."""
    if hasattr(e, "status_code") and getattr(e, "status_code") == 413:
        return True
    err_str = str(e).lower()
    return "413" in err_str or "request too large" in err_str or "tpm" in err_str


async def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout_s: int = 1800,  # Increased default to 30 minutes for complex tasks
) -> str:
    backend = _get_backend(model)
    max_retries = _env_int("LLM_MAX_RETRIES", 3)
    last_err: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            if backend == "groq":
                return await asyncio.to_thread(_call_groq_chat, messages, model, temperature, max_tokens)
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            t = _env_int("OLLAMA_TIMEOUT_S", int(timeout_s))
            return await asyncio.to_thread(_call_ollama_chat, messages, model, temperature, max_tokens, base_url, t)
        except BaseException as e:
            last_err = e
            if attempt < max_retries and _is_transient(e):
                delay = 60.0 if _is_413_or_tpm(e) else (2 ** attempt)  # 60s for TPM, else 1s,2s,4s
                await asyncio.sleep(delay)
                continue
            raise
    assert last_err is not None
    raise last_err


async def call_llm_json(
    messages: list[dict],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout_s: int = 1800,  # Increased for complex tasks
    max_retries: int = 2,
) -> Any:
    last = None
    for _ in range(max_retries + 1):
        txt = await call_llm(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
        last = _best_effort_json(txt)
        if last is not None:
            return last
        messages = messages + [
            {
                "role": "user",
                "content": "Your previous response was not valid JSON. Output ONLY valid JSON, no markdown fences.",
            }
        ]
    return last


ToolFunc = Callable[[dict], Any]


async def call_agent_with_tools(
    system_prompt: str,
    user_prompt: str,
    model: str,
    tools: dict[str, ToolFunc],
    temperature: float = 0.7,
    max_tokens: int = 1200,
    max_tool_iterations: int = 6,
    timeout_s: int = 1800,  # Increased for complex tasks
    log_event: Optional[Callable[[dict], None]] = None,
) -> str:
    """
    Tool protocol (model outputs JSON):
    {"action": "web_search", "input": {...}}
    {"action": "fetch_page", "input": {...}}
    {"action": "search_and_fetch", "input": {...}}
    {"action": "run_command", "input": {"command": "...", "timeout_s": 60}}
    {"action": "read_file", "input": {"path": "..."}}
    {"action": "write_file", "input": {"path": "...", "content": "..."}}
    {"action": "search_replace", "input": {"path": "...", "old_string": "...", "new_string": "..."}}
    {"action": "final_answer", "answer": "..."}
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    used_any_tool = False

    for step in range(max_tool_iterations):
        t0 = time.time()
        txt = await call_llm(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        dt = round(time.time() - t0, 3)
        if log_event:
            log_event({"type": "llm_response", "step": step, "elapsed_s": dt, "chars": len(txt)})

        parsed = _best_effort_json(txt)
        if isinstance(parsed, dict):
            action = parsed.get("action")
            if action == "final_answer":
                ans = parsed.get("answer", "")
                return (ans or "").strip()

            if action in tools:
                tool_input = parsed.get("input", {}) or {}
                if log_event:
                    log_event({"type": "tool_call", "step": step, "action": action, "input": tool_input})
                try:
                    t1 = time.time()
                    tool_out = await tools[action](tool_input)
                    if log_event:
                        log_event({"type": "tool_result", "step": step, "action": action, "elapsed_s": round(time.time() - t1, 3)})
                except Exception as e:
                    tool_out = {"error": str(e)}
                used_any_tool = True
                messages.append({"role": "assistant", "content": txt})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result for {action}:\n{json.dumps(tool_out, ensure_ascii=False)[:20000]}",
                    }
                )
                continue

        # If tools exist but the model didn't output a tool action yet, force the protocol.
        if tools and not used_any_tool:
            messages.append({"role": "assistant", "content": txt})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You MUST use the tool protocol. Output ONLY JSON.\n"
                        "Choose one action: web_search, fetch_page, search_and_fetch, run_command, read_file, write_file, search_replace.\n"
                        "Format: {\"action\":\"search_replace\",\"input\":{\"path\":\"...\",\"old_string\":\"...\",\"new_string\":\"...\"}}"
                    ),
                }
            )
            continue

        # Otherwise treat as final answer fallback.
        return txt.strip()

    # Max iterations reached: ask for final answer
    messages.append({"role": "user", "content": "Return your best final answer now."})
    return (
        await call_llm(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, timeout_s=timeout_s)
    ).strip()


def load_env() -> None:
    # .env may be blocked; load if present.
    load_dotenv(override=False)


