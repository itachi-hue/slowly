from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Logger:
    run_id: str
    log_path: str
    quiet: bool = False

    def event(self, e: dict) -> None:
        e = dict(e)
        e.setdefault("ts", time.time())
        e.setdefault("run_id", self.run_id)
        line = json.dumps(e, ensure_ascii=False)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        if not self.quiet:
            # compact console output
            t = e.get("type", "event")
            it = e.get("iteration")
            msg = e.get("message") or e.get("node") or e.get("action") or ""
            prefix = f"[{self.run_id}]"
            if it is not None:
                prefix += f"[it={it}]"
            print(f"{prefix} {t} {msg}".rstrip(), file=sys.stdout)


def make_logger(runs_dir: str, run_id: str, quiet: bool = False) -> Logger:
    os.makedirs(runs_dir, exist_ok=True)
    log_path = os.path.join(runs_dir, f"{run_id}.jsonl")
    return Logger(run_id=run_id, log_path=log_path, quiet=quiet)


