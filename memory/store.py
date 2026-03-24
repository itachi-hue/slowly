from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field

# Re-export new task tree models for convenience
from memory.redis_store import (
    SlotDef,
    TaskDefinition,
    TaskOutput as TreeTaskOutput,
    extract_slot_references,
)


class TaskOutput(BaseModel):
    task_id: str
    parent_task_id: Optional[str] = None
    agent_role: str
    output: str
    confidence: float = 0.5
    sources: list[str] = Field(default_factory=list)
    assumptions_made: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)
    iteration: int
    timestamp: float = Field(default_factory=lambda: time.time())


class CritiqueReport(BaseModel):
    version: str
    overall_score: float
    iteration: int
    weaknesses: list[dict] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    suggested_fixes: list[dict] = Field(default_factory=list)
    diminishing_returns_signal: bool = False
    timestamp: float = Field(default_factory=lambda: time.time())


class SynthesisVersion(BaseModel):
    version_id: str
    iteration: int
    content: str
    score: float = 0.0
    critique: Optional[str] = None
    timestamp: float = Field(default_factory=lambda: time.time())


@dataclass
class RunPaths:
    run_id: str
    runs_dir: str
    db_path: str
    output_md_path: str


def make_run_paths(runs_dir: str, run_id: str) -> RunPaths:
    os.makedirs(runs_dir, exist_ok=True)
    db_path = os.path.join(runs_dir, f"{run_id}.db")
    output_md_path = os.path.join(runs_dir, f"{run_id}_output.md")
    return RunPaths(run_id=run_id, runs_dir=runs_dir, db_path=db_path, output_md_path=output_md_path)


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task_outputs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              task_id TEXT NOT NULL,
              parent_task_id TEXT,
              agent_role TEXT NOT NULL,
              output TEXT NOT NULL,
              confidence REAL NOT NULL,
              sources_json TEXT NOT NULL,
              assumptions_json TEXT NOT NULL,
              open_questions_json TEXT NOT NULL,
              suggested_followups_json TEXT NOT NULL,
              iteration INTEGER NOT NULL,
              ts REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS syntheses (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              version_id TEXT NOT NULL,
              iteration INTEGER NOT NULL,
              content TEXT NOT NULL,
              score REAL NOT NULL,
              critique TEXT,
              ts REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS critiques (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              version TEXT NOT NULL,
              overall_score REAL NOT NULL,
              iteration INTEGER NOT NULL,
              weaknesses_json TEXT NOT NULL,
              strengths_json TEXT NOT NULL,
              suggested_fixes_json TEXT NOT NULL,
              diminishing_returns INTEGER NOT NULL,
              ts REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def save_task_output(self, o: TaskOutput) -> None:
        self._conn.execute(
            """
            INSERT INTO task_outputs (
              task_id, parent_task_id, agent_role, output, confidence,
              sources_json, assumptions_json, open_questions_json, suggested_followups_json,
              iteration, ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                o.task_id,
                o.parent_task_id,
                o.agent_role,
                o.output,
                float(o.confidence),
                json.dumps(o.sources),
                json.dumps(o.assumptions_made),
                json.dumps(o.open_questions),
                json.dumps(o.suggested_followups),
                int(o.iteration),
                float(o.timestamp),
            ),
        )
        self._conn.commit()

    def get_task_outputs(self, iteration: int) -> list[TaskOutput]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT task_id, parent_task_id, agent_role, output, confidence,
                   sources_json, assumptions_json, open_questions_json, suggested_followups_json,
                   iteration, ts
            FROM task_outputs
            WHERE iteration = ?
            ORDER BY id ASC
            """,
            (int(iteration),),
        )
        rows = cur.fetchall()
        out: list[TaskOutput] = []
        for r in rows:
            out.append(
                TaskOutput(
                    task_id=r[0],
                    parent_task_id=r[1],
                    agent_role=r[2],
                    output=r[3],
                    confidence=float(r[4]),
                    sources=json.loads(r[5]),
                    assumptions_made=json.loads(r[6]),
                    open_questions=json.loads(r[7]),
                    suggested_followups=json.loads(r[8]),
                    iteration=int(r[9]),
                    timestamp=float(r[10]),
                )
            )
        return out

    def save_synthesis(self, s: SynthesisVersion) -> None:
        self._conn.execute(
            """
            INSERT INTO syntheses (version_id, iteration, content, score, critique, ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (s.version_id, int(s.iteration), s.content, float(s.score), s.critique, float(s.timestamp)),
        )
        self._conn.commit()

    def update_synthesis_score(self, version_id: str, score: float, critique: Optional[str] = None) -> None:
        self._conn.execute(
            "UPDATE syntheses SET score = ?, critique = COALESCE(?, critique) WHERE version_id = ?",
            (float(score), critique, version_id),
        )
        self._conn.commit()

    def get_all_syntheses(self) -> list[SynthesisVersion]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT version_id, iteration, content, score, critique, ts
            FROM syntheses
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()
        return [
            SynthesisVersion(
                version_id=r[0],
                iteration=int(r[1]),
                content=r[2],
                score=float(r[3]),
                critique=r[4],
                timestamp=float(r[5]),
            )
            for r in rows
        ]

    def get_best_synthesis(self) -> Optional[SynthesisVersion]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT version_id, iteration, content, score, critique, ts
            FROM syntheses
            ORDER BY score DESC, id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            return None
        return SynthesisVersion(
            version_id=row[0],
            iteration=int(row[1]),
            content=row[2],
            score=float(row[3]),
            critique=row[4],
            timestamp=float(row[5]),
        )

    def save_critique(self, c: CritiqueReport) -> None:
        self._conn.execute(
            """
            INSERT INTO critiques (
              version, overall_score, iteration,
              weaknesses_json, strengths_json, suggested_fixes_json,
              diminishing_returns, ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                c.version,
                float(c.overall_score),
                int(c.iteration),
                json.dumps(c.weaknesses),
                json.dumps(c.strengths),
                json.dumps(c.suggested_fixes),
                1 if c.diminishing_returns_signal else 0,
                float(c.timestamp),
            ),
        )
        self._conn.commit()

    def get_all_critiques(self) -> list[CritiqueReport]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT version, overall_score, iteration,
                   weaknesses_json, strengths_json, suggested_fixes_json,
                   diminishing_returns, ts
            FROM critiques
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()
        out: list[CritiqueReport] = []
        for r in rows:
            out.append(
                CritiqueReport(
                    version=r[0],
                    overall_score=float(r[1]),
                    iteration=int(r[2]),
                    weaknesses=json.loads(r[3]),
                    strengths=json.loads(r[4]),
                    suggested_fixes=json.loads(r[5]),
                    diminishing_returns_signal=bool(int(r[6])),
                    timestamp=float(r[7]),
                )
            )
        return out


def safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


