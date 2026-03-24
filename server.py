"""
Simple web server for visualizing task tree execution.

Run with: python server.py
Then open: http://localhost:8080
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict
from typing import Optional

from aiohttp import web
import aiohttp_cors

from agents.core import RunConfig
from agents.llm import load_env, call_llm, call_llm_json
from memory.redis_store import create_state_store, TaskDefinition, TaskOutput
from orchestration.decomposer import create_decomposer
from orchestration.assembler import create_assembler
from graph_v2 import run_tree_graph


# Global state for tracking runs
active_runs: dict[str, dict] = {}
event_queues: dict[str, asyncio.Queue] = {}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None and v != "" else int(default)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None and v != "" else float(default)
    except Exception:
        return float(default)


async def index_handler(request: web.Request) -> web.Response:
    """Serve the main HTML page."""
    return web.Response(text=HTML_PAGE, content_type='text/html')


async def start_run_handler(request: web.Request) -> web.Response:
    """Start a new research run."""
    data = await request.json()
    problem = data.get('problem', '').strip()

    if not problem:
        return web.json_response({'error': 'Problem is required'}, status=400)

    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:4]}"

    # Create event queue for this run
    event_queues[run_id] = asyncio.Queue()

    # Initialize run state
    active_runs[run_id] = {
        'run_id': run_id,
        'problem': problem,
        'status': 'starting',
        'start_time': time.time(),
        'events': [],
        'tree': {},
        'final_output': None,
    }

    # Start execution in background
    asyncio.create_task(execute_run(run_id, problem))

    return web.json_response({'run_id': run_id})


async def execute_run(run_id: str, problem: str) -> None:
    """Execute a run in the background."""
    load_env()

    store = await create_state_store()
    decomposer = create_decomposer()
    assembler = create_assembler()

    backend = os.getenv("ACTIVE_BACKEND", "ollama")
    primary_model = os.getenv("PRIMARY_MODEL", "qwen2.5:7b")
    fast_model = os.getenv("FAST_MODEL") or primary_model
    if backend == "groq":
        primary_model = os.getenv("GROQ_PRIMARY_MODEL") or "llama-3.1-8b-instant"
        fast_model = os.getenv("GROQ_FAST_MODEL") or primary_model
    elif backend == "openai":
        primary_model = os.getenv("OPENAI_PRIMARY_MODEL") or "gpt-4o"
        fast_model = os.getenv("OPENAI_FAST_MODEL") or "gpt-4o-mini"

    cfg = RunConfig(
        primary_model=primary_model,
        fast_model=fast_model,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        max_parallel_agents=_env_int("MAX_PARALLEL_AGENTS", 1),
        max_total_tasks=_env_int("MAX_TOTAL_TASKS", 100),
        max_task_depth=_env_int("MAX_TASK_DEPTH", 3),
        min_score_improvement=_env_float("MIN_SCORE_IMPROVEMENT", 0.02),
        target_score=_env_float("TARGET_SCORE", 0.92),
    )

    def log_event(e: dict) -> None:
        # Add to run state
        if run_id in active_runs:
            active_runs[run_id]['events'].append(e)

            # Update tree structure based on events
            update_tree_from_event(run_id, e)

        # Push to SSE queue
        if run_id in event_queues:
            event_queues[run_id].put_nowait(e)

    active_runs[run_id]['status'] = 'running'
    log_event({'type': 'run_start', 'problem': problem})

    try:
        final_output = await run_tree_graph(
            problem=problem,
            run_id=run_id,
            cfg=cfg,
            store=store,
            decomposer=decomposer,
            assembler=assembler,
            call_llm=call_llm,
            call_llm_json=call_llm_json,
            time_budget_seconds=3600,
            log_event=log_event,
        )

        active_runs[run_id]['final_output'] = final_output
        active_runs[run_id]['status'] = 'completed'
        log_event({'type': 'run_complete', 'output': final_output[:500]})

    except Exception as e:
        active_runs[run_id]['status'] = 'error'
        active_runs[run_id]['error'] = str(e)
        log_event({'type': 'run_error', 'error': str(e)})

    await store.close()


def update_tree_from_event(run_id: str, event: dict) -> None:
    """Update tree visualization from event."""
    tree = active_runs[run_id].get('tree', {})

    event_type = event.get('type', '')
    task_id = event.get('task_id')

    if event_type == 'root_task_created':
        task_id = event.get('root_task_id')
        tree[task_id] = {
            'id': task_id,
            'parent_id': None,
            'question': active_runs[run_id]['problem'][:100],
            'status': 'pending',
            'slot': 'root',
            'children': [],
            'output': None,
        }

    elif event_type == 'child_spawned':
        parent_id = event.get('parent_id')
        child_id = event.get('child_id')
        slot = event.get('slot', '')

        tree[child_id] = {
            'id': child_id,
            'parent_id': parent_id,
            'question': '',
            'status': event.get('status', 'pending'),
            'slot': slot,
            'children': [],
            'output': None,
        }

        if parent_id in tree:
            tree[parent_id]['children'].append(child_id)

    elif event_type == 'leaf_start' and task_id:
        if task_id in tree:
            tree[task_id]['status'] = 'running'
            tree[task_id]['question'] = event.get('question', '')[:100]

    elif event_type == 'leaf_complete' and task_id:
        if task_id in tree:
            tree[task_id]['status'] = 'completed'

    elif event_type == 'assemble_start' and task_id:
        if task_id in tree:
            tree[task_id]['status'] = 'assembling'

    elif event_type == 'assemble_complete' and task_id:
        if task_id in tree:
            tree[task_id]['status'] = 'completed'

    elif event_type == 'leaf_error' and task_id:
        if task_id in tree:
            tree[task_id]['status'] = 'error'
            tree[task_id]['error'] = event.get('error', 'Unknown error')

    active_runs[run_id]['tree'] = tree


async def events_handler(request: web.Request) -> web.StreamResponse:
    """Server-Sent Events endpoint for real-time updates."""
    run_id = request.match_info.get('run_id')

    if run_id not in event_queues:
        return web.json_response({'error': 'Run not found'}, status=404)

    response = web.StreamResponse()
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    await response.prepare(request)

    queue = event_queues[run_id]

    # Send initial state
    if run_id in active_runs:
        initial = {
            'type': 'init',
            'run': active_runs[run_id],
        }
        await response.write(f"data: {json.dumps(initial)}\n\n".encode())

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
                await response.write(f"data: {json.dumps(event)}\n\n".encode())

                # Check if run is done
                if event.get('type') in ('run_complete', 'run_error'):
                    break

            except asyncio.TimeoutError:
                # Send keepalive
                await response.write(b": keepalive\n\n")

    except ConnectionResetError:
        pass

    return response


async def status_handler(request: web.Request) -> web.Response:
    """Get current status of a run."""
    run_id = request.match_info.get('run_id')

    if run_id not in active_runs:
        return web.json_response({'error': 'Run not found'}, status=404)

    return web.json_response(active_runs[run_id])


async def list_runs_handler(request: web.Request) -> web.Response:
    """List all runs."""
    runs = [
        {
            'run_id': r['run_id'],
            'problem': r['problem'][:100],
            'status': r['status'],
            'start_time': r['start_time'],
        }
        for r in active_runs.values()
    ]
    return web.json_response(runs)


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slowly - Task Tree Visualizer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f1a;
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 {
            font-size: 2rem;
            margin-bottom: 20px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .input-section {
            background: #1a1a2e;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .input-row { display: flex; gap: 10px; }
        input[type="text"] {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #334;
            border-radius: 8px;
            background: #0f0f1a;
            color: #fff;
            font-size: 16px;
        }
        input[type="text"]:focus { outline: none; border-color: #667eea; }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .main-content { display: grid; grid-template-columns: 1fr 450px; gap: 20px; }
        .tree-section {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            min-height: 600px;
            overflow-y: auto;
        }
        .events-section {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            max-height: 700px;
            overflow-y: auto;
        }
        h2 { font-size: 1.1rem; margin-bottom: 15px; color: #888; }

        /* Tree Node Styles */
        .tree-node {
            background: #12121f;
            border: 2px solid #334;
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
        }
        .tree-node.pending { border-color: #555; }
        .tree-node.running { border-color: #f39c12; box-shadow: 0 0 15px rgba(243, 156, 18, 0.3); }
        .tree-node.assembling { border-color: #9b59b6; box-shadow: 0 0 15px rgba(155, 89, 182, 0.3); }
        .tree-node.completed { border-color: #27ae60; }
        .tree-node.blocked { border-color: #e74c3c; opacity: 0.7; }

        .node-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 14px;
            background: rgba(255,255,255,0.03);
            border-bottom: 1px solid #333;
        }
        .node-slot {
            font-size: 0.85rem;
            font-weight: 600;
            color: #667eea;
        }
        .node-status {
            font-size: 0.7rem;
            padding: 3px 10px;
            border-radius: 12px;
            text-transform: uppercase;
            font-weight: 600;
        }
        .node-status.pending { background: #444; color: #aaa; }
        .node-status.running { background: #f39c12; color: #000; }
        .node-status.assembling { background: #9b59b6; color: #fff; }
        .node-status.completed { background: #27ae60; color: #fff; }
        .node-status.blocked { background: #e74c3c; color: #fff; }

        .node-body { padding: 12px 14px; }

        .node-section {
            margin-bottom: 10px;
        }
        .node-section:last-child { margin-bottom: 0; }

        .section-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 4px;
            font-weight: 600;
        }
        .section-content {
            font-size: 0.85rem;
            color: #ccc;
            background: #0a0a12;
            padding: 8px 10px;
            border-radius: 6px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 150px;
            overflow-y: auto;
        }
        .section-content.input { border-left: 3px solid #3498db; }
        .section-content.output { border-left: 3px solid #27ae60; }
        .section-content.slots { border-left: 3px solid #9b59b6; }

        .node-children {
            margin-left: 20px;
            padding-left: 15px;
            border-left: 2px dashed #333;
        }

        /* Events */
        .event-item {
            background: #12121f;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 0.8rem;
            font-family: monospace;
            border-left: 3px solid #334;
        }
        .event-item.leaf_complete { border-left-color: #27ae60; }
        .event-item.assemble_complete { border-left-color: #9b59b6; }
        .event-item.sibling_unblocked { border-left-color: #3498db; }
        .event-item.child_spawned { border-left-color: #f39c12; }

        .event-type { color: #667eea; font-weight: bold; }
        .event-detail { color: #888; margin-top: 4px; font-size: 0.75rem; }

        /* Output section */
        .output-section {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        .output-content {
            background: #0a0a12;
            padding: 16px;
            border-radius: 8px;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 0.9rem;
            max-height: 500px;
            overflow-y: auto;
            border-left: 4px solid #27ae60;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            margin-left: 10px;
        }
        .status-badge.running { background: #f39c12; color: #000; }
        .status-badge.completed { background: #27ae60; }
        .status-badge.error { background: #e74c3c; }

        .collapsed .node-body { display: none; }
        .collapse-btn {
            background: none;
            border: none;
            color: #666;
            cursor: pointer;
            font-size: 0.8rem;
            padding: 0 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌳 Slowly - Task Tree Visualizer</h1>

        <div class="input-section">
            <div class="input-row">
                <input type="text" id="problem-input" placeholder="Enter your research question..."
                       value="Solve the system: 2x + 3y - z = 1, x - y + 2z = 4, 3x + y + z = 7">
                <button id="start-btn" onclick="startRun()">Start</button>
            </div>
        </div>

        <div id="run-status"></div>

        <div class="main-content">
            <div class="tree-section">
                <h2>Task Tree</h2>
                <div id="tree-container">
                    <p style="color: #666;">Enter a question and click Start...</p>
                </div>
            </div>

            <div class="events-section">
                <h2>Events</h2>
                <div id="events-container"></div>
            </div>
        </div>

        <div id="output-section" class="output-section" style="display: none;">
            <h2>Final Output</h2>
            <div id="output-content" class="output-content"></div>
        </div>
    </div>

    <script>
        let currentRunId = null;
        let eventSource = null;
        let treeData = {};

        function startRun() {
            const problem = document.getElementById('problem-input').value.trim();
            if (!problem) return;

            document.getElementById('start-btn').disabled = true;
            document.getElementById('tree-container').innerHTML = '<p style="color: #f39c12;">Starting...</p>';
            document.getElementById('events-container').innerHTML = '';
            document.getElementById('output-section').style.display = 'none';
            treeData = {};

            fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ problem })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    document.getElementById('start-btn').disabled = false;
                    return;
                }
                currentRunId = data.run_id;
                connectSSE(data.run_id);
                document.getElementById('run-status').innerHTML =
                    `<p>Run ID: <code>${data.run_id}</code> <span class="status-badge running">Running</span></p>`;
            })
            .catch(err => {
                alert('Error: ' + err);
                document.getElementById('start-btn').disabled = false;
            });
        }

        function connectSSE(runId) {
            if (eventSource) eventSource.close();
            eventSource = new EventSource(`/api/events/${runId}`);
            eventSource.onmessage = (e) => handleEvent(JSON.parse(e.data));
            eventSource.onerror = () => {
                console.log('SSE closed');
                document.getElementById('start-btn').disabled = false;
            };
        }

        function handleEvent(event) {
            // Add to events list
            const eventsContainer = document.getElementById('events-container');
            const eventDiv = document.createElement('div');
            eventDiv.className = `event-item ${event.type || ''}`;

            let detail = '';
            if (event.type === 'leaf_complete') {
                detail = `<div class="event-detail">Output: ${(event.output || '').slice(0, 100)}...</div>`;
            } else if (event.type === 'assemble_complete') {
                detail = `<div class="event-detail">Assembled ${Object.keys(event.inputs || {}).length} slots</div>`;
            } else if (event.type === 'sibling_unblocked') {
                detail = `<div class="event-detail">Unblocked by: ${event.unblocked_by}</div>`;
            } else if (event.type === 'child_spawned') {
                detail = `<div class="event-detail">${event.slot}: ${(event.question || '').slice(0, 60)}...</div>`;
            }

            eventDiv.innerHTML = `<span class="event-type">${event.type || 'unknown'}</span> ${event.task_id || ''} ${detail}`;
            eventsContainer.insertBefore(eventDiv, eventsContainer.firstChild);

            // Handle init
            if (event.type === 'init' && event.run) {
                treeData = event.run.tree || {};
                renderTree();
                return;
            }

            // Update tree
            if (event.type === 'root_task_created') {
                const taskId = event.root_task_id;
                treeData[taskId] = {
                    id: taskId,
                    parent_id: null,
                    question: document.getElementById('problem-input').value,
                    status: 'pending',
                    slot: 'root',
                    children: [],
                    input: document.getElementById('problem-input').value,
                    output: null
                };
            }
            else if (event.type === 'child_spawned') {
                treeData[event.child_id] = {
                    id: event.child_id,
                    parent_id: event.parent_id,
                    question: event.question || '',
                    status: event.status || 'pending',
                    slot: event.slot || '',
                    children: [],
                    input: event.question || '',
                    output: null
                };
                if (treeData[event.parent_id]) {
                    treeData[event.parent_id].children.push(event.child_id);
                }
            }
            else if (event.type === 'sibling_unblocked' && event.task_id) {
                if (treeData[event.task_id]) {
                    treeData[event.task_id].status = 'pending';
                    treeData[event.task_id].question = event.resolved_question || treeData[event.task_id].question;
                    treeData[event.task_id].input = event.resolved_question || treeData[event.task_id].input;
                }
            }
            else if (event.type === 'leaf_start' && event.task_id) {
                if (treeData[event.task_id]) {
                    treeData[event.task_id].status = 'running';
                    treeData[event.task_id].question = event.question || treeData[event.task_id].question;
                    treeData[event.task_id].input = event.question || treeData[event.task_id].input;
                }
            }
            else if (event.type === 'leaf_complete' && event.task_id) {
                if (treeData[event.task_id]) {
                    treeData[event.task_id].status = 'completed';
                    treeData[event.task_id].output = event.output || '';
                    treeData[event.task_id].input = event.input || treeData[event.task_id].input;
                }
            }
            else if (event.type === 'assemble_start' && event.task_id) {
                if (treeData[event.task_id]) {
                    treeData[event.task_id].status = 'assembling';
                }
            }
            else if (event.type === 'assemble_complete' && event.task_id) {
                if (treeData[event.task_id]) {
                    treeData[event.task_id].status = 'completed';
                    treeData[event.task_id].output = event.output || '';
                    treeData[event.task_id].inputs = event.inputs || {};
                }
            }
            else if (event.type === 'run_complete') {
                document.getElementById('run-status').innerHTML =
                    `<p>Run ID: <code>${currentRunId}</code> <span class="status-badge completed">Completed</span></p>`;
                document.getElementById('start-btn').disabled = false;

                fetch(`/api/status/${currentRunId}`)
                    .then(r => r.json())
                    .then(data => {
                        if (data.final_output) {
                            document.getElementById('output-section').style.display = 'block';
                            document.getElementById('output-content').textContent = data.final_output;
                        }
                    });
            }
            else if (event.type === 'run_error') {
                document.getElementById('run-status').innerHTML =
                    `<p>Run ID: <code>${currentRunId}</code> <span class="status-badge error">Error</span></p>`;
                document.getElementById('start-btn').disabled = false;
            }

            renderTree();
        }

        function renderTree() {
            const container = document.getElementById('tree-container');
            const roots = Object.values(treeData).filter(n => !n.parent_id);

            if (roots.length === 0) {
                container.innerHTML = '<p style="color: #666;">Waiting for tasks...</p>';
                return;
            }
            container.innerHTML = roots.map(renderNode).join('');
        }

        function renderNode(node) {
            const childrenHtml = node.children
                .map(cid => treeData[cid])
                .filter(c => c)
                .map(renderNode)
                .join('');

            const inputHtml = node.input ? `
                <div class="node-section">
                    <div class="section-label">Input (Question)</div>
                    <div class="section-content input">${escapeHtml(node.input)}</div>
                </div>
            ` : '';

            const outputHtml = node.output ? `
                <div class="node-section">
                    <div class="section-label">Output</div>
                    <div class="section-content output">${escapeHtml(node.output)}</div>
                </div>
            ` : '';

            // Show assembled inputs for parent nodes
            const inputsHtml = node.inputs ? `
                <div class="node-section">
                    <div class="section-label">Slot Inputs (from children)</div>
                    <div class="section-content slots">${Object.entries(node.inputs).map(([k,v]) =>
                        `<strong>${k}:</strong> ${escapeHtml(v)}`
                    ).join('\\n\\n')}</div>
                </div>
            ` : '';

            return `
                <div class="tree-node ${node.status}">
                    <div class="node-header">
                        <span class="node-slot">${node.slot}</span>
                        <span class="node-status ${node.status}">${node.status}</span>
                    </div>
                    <div class="node-body">
                        ${inputHtml}
                        ${inputsHtml}
                        ${outputHtml}
                    </div>
                    ${childrenHtml ? `<div class="node-children">${childrenHtml}</div>` : ''}
                </div>
            `;
        }

        function escapeHtml(str) {
            if (!str) return '';
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        }

        document.getElementById('problem-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') startRun();
        });
    </script>
</body>
</html>
"""


def create_app() -> web.Application:
    """Create the web application."""
    app = web.Application()

    # Setup CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # Routes
    app.router.add_get('/', index_handler)
    app.router.add_post('/api/start', start_run_handler)
    app.router.add_get('/api/events/{run_id}', events_handler)
    app.router.add_get('/api/status/{run_id}', status_handler)
    app.router.add_get('/api/runs', list_runs_handler)

    # Apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)

    return app


if __name__ == '__main__':
    import sys

    # Load environment
    load_env()

    port = int(os.getenv('PORT', '8080'))

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║           🌳 Slowly - Task Tree Visualizer                ║
╠═══════════════════════════════════════════════════════════╣
║  Open in browser: http://localhost:{port:<5}                 ║
║                                                           ║
║  Backend: {os.getenv('ACTIVE_BACKEND', 'ollama'):<10}                                ║
║  Model:   {os.getenv('PRIMARY_MODEL', 'qwen2.5:7b'):<20}                   ║
╚═══════════════════════════════════════════════════════════╝
""")

    app = create_app()
    web.run_app(app, host='0.0.0.0', port=port)
