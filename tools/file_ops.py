"""File read/write tools for agents. Sandboxed to workspace."""
from __future__ import annotations

import os
from typing import Optional

# Paths/patterns that are never writable
_WRITE_BLOCKED = {
    ".git",
    "node_modules",
    "__pycache__",
    ".env",
    ".venv",
    "venv",
    ".cursor",
    ".cursorrules",
}

_MAX_READ_BYTES = 1_000_000  # 1MB


def _resolve_path(path: str, workspace_root: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Resolve path to absolute. Returns (abs_path, error_msg or None)."""
    root = (workspace_root or os.getcwd()).replace("\\", "/")
    root = os.path.normpath(root)
    path = str(path).strip().replace("\\", "/")
    if not path:
        return "", "path is empty"

    # Resolve to absolute within workspace
    if os.path.isabs(path):
        abs_path = os.path.normpath(path)
    else:
        abs_path = os.path.normpath(os.path.join(root, path))

    # Must stay under workspace
    try:
        abs_path = os.path.abspath(abs_path)
        root_abs = os.path.abspath(root)
        if not abs_path.startswith(root_abs):
            return abs_path, "path is outside workspace"
    except Exception:
        return path, "invalid path"

    return abs_path, None


def _is_write_blocked(abs_path: str, root: str) -> bool:
    """Check if path is in a blocked directory."""
    rel = os.path.relpath(abs_path, root)
    parts = rel.replace("\\", "/").split("/")
    for part in parts:
        if part in _WRITE_BLOCKED:
            return True
    return False


async def read_file(path: str, workspace_root: Optional[str] = None) -> dict:
    """
    Read file contents. Path is relative to workspace or absolute within it.

    Input: {"path": "src/foo.py"}
    Returns: {"content": "...", "path": "abs_path"} or {"error": "..."}
    """
    root = workspace_root or os.getcwd()
    abs_path, err = _resolve_path(path, root)
    if err:
        return {"error": err, "content": "", "path": path}

    if not os.path.isfile(abs_path):
        return {"error": "file not found", "content": "", "path": abs_path}

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(_MAX_READ_BYTES)
            if f.read(1):
                return {"error": "file too large (max 1MB)", "content": content, "path": abs_path}
    except PermissionError:
        return {"error": "permission denied", "content": "", "path": abs_path}
    except Exception as e:
        return {"error": str(e), "content": "", "path": abs_path}

    return {"content": content, "path": abs_path}


async def write_file(path: str, content: str, workspace_root: Optional[str] = None) -> dict:
    """
    Create or overwrite a file. Path must be within workspace.
    Blocked: .git, node_modules, __pycache__, .env, venv, etc.

    Input: {"path": "src/foo.py", "content": "..."}
    Returns: {"path": "abs_path", "wrote_bytes": N} or {"error": "..."}
    """
    root = workspace_root or os.getcwd()
    abs_path, err = _resolve_path(path, root)
    if err:
        return {"error": err, "path": path, "wrote_bytes": 0}

    if _is_write_blocked(abs_path, root):
        return {"error": "path is in a protected directory", "path": abs_path, "wrote_bytes": 0}

    try:
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        data = content.encode("utf-8") if isinstance(content, str) else content
        with open(abs_path, "wb") as f:
            f.write(data)
        return {"path": abs_path, "wrote_bytes": len(data)}
    except PermissionError:
        return {"error": "permission denied", "path": abs_path, "wrote_bytes": 0}
    except Exception as e:
        return {"error": str(e), "path": abs_path, "wrote_bytes": 0}


async def search_replace(
    path: str,
    old_string: str,
    new_string: str,
    workspace_root: Optional[str] = None,
) -> dict:
    """
    Replace first occurrence of old_string with new_string in a file.
    Best for small, surgical edits. old_string must match exactly.

    Input: {"path": "src/foo.py", "old_string": "x = 1", "new_string": "x = 2"}
    Returns: {"path": "abs_path", "replaced": true} or {"error": "..."}
    """
    root = workspace_root or os.getcwd()
    abs_path, err = _resolve_path(path, root)
    if err:
        return {"error": err, "path": path, "replaced": False}

    if _is_write_blocked(abs_path, root):
        return {"error": "path is in a protected directory", "path": abs_path, "replaced": False}

    if not os.path.isfile(abs_path):
        return {"error": "file not found", "path": abs_path, "replaced": False}

    old_str = str(old_string)
    new_str = str(new_string)

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(_MAX_READ_BYTES)
            if f.read(1):
                return {"error": "file too large (max 1MB)", "path": abs_path, "replaced": False}
    except Exception as e:
        return {"error": str(e), "path": abs_path, "replaced": False}

    if old_str not in content:
        return {"error": "old_string not found in file (exact match required)", "path": abs_path, "replaced": False}

    new_content = content.replace(old_str, new_str, 1)
    try:
        with open(abs_path, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)
        return {"path": abs_path, "replaced": True}
    except Exception as e:
        return {"error": str(e), "path": abs_path, "replaced": False}
