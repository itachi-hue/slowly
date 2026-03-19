"""Sandboxed shell command execution for agents."""
from __future__ import annotations

import asyncio
import os
import re
from typing import Optional

# Commands/patterns to block for safety
_BLOCKED_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\brm\s+-r\s+-f\b",
    r"\brd\s+/s\s+/q\b",
    r"\bdel\s+/s\s+/q\b",
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bsudo\b",
    r"\bsu\s+-c\b",
    r"chmod\s+777",
    r">\s*/dev/sd",  # overwrite block device
    r":(){",  # fork bomb
]
_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS), re.IGNORECASE)


def _is_blocked(cmd: str) -> bool:
    return bool(_BLOCKED_RE.search(cmd))


async def run_command(
    command: str,
    timeout_s: int = 60,
    cwd: Optional[str] = None,
) -> dict:
    """
    Run a shell command and return stdout, stderr, returncode.

    Sandboxed: blocks obviously dangerous patterns.
    Input: {"command": "pytest", "timeout_s": 30, "cwd": "/optional/path"}
    """
    cmd = str(command).strip()
    if not cmd:
        return {"error": "command is empty", "stdout": "", "stderr": "", "returncode": -1}

    if _is_blocked(cmd):
        return {
            "error": "command blocked for safety",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }

    work_dir = cwd or os.getcwd()
    if not os.path.isdir(work_dir):
        return {
            "error": f"cwd does not exist: {work_dir}",
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=os.environ.copy(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(timeout_s),
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "error": f"timeout after {timeout_s}s",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }

        return {
            "stdout": stdout.decode("utf-8", errors="replace").strip(),
            "stderr": stderr.decode("utf-8", errors="replace").strip(),
            "returncode": proc.returncode or 0,
        }
    except Exception as e:
        return {
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "returncode": -1,
        }
