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
        .container { max-width: 1800px; margin: 0 auto; }
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

        .main-content { display: grid; grid-template-columns: 1fr 400px; gap: 20px; }
        .graph-section {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            min-height: 600px;
            overflow: auto;
        }
        .detail-section {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            max-height: 800px;
            overflow-y: auto;
        }
        h2 { font-size: 1.1rem; margin-bottom: 15px; color: #888; }

        /* SVG Graph Styles */
        #graph-svg {
            width: 100%;
            min-height: 500px;
        }
        .graph-node {
            cursor: pointer;
            transition: transform 0.2s;
        }
        .graph-node:hover { filter: brightness(1.2); }
        .graph-node rect {
            stroke-width: 2;
            rx: 8;
            ry: 8;
        }
        .graph-node.pending rect { fill: #1a1a2e; stroke: #555; }
        .graph-node.running rect { fill: #2a2a1e; stroke: #f39c12; filter: drop-shadow(0 0 8px rgba(243, 156, 18, 0.5)); }
        .graph-node.assembling rect { fill: #2a1a2e; stroke: #9b59b6; filter: drop-shadow(0 0 8px rgba(155, 89, 182, 0.5)); }
        .graph-node.completed rect { fill: #1a2a1e; stroke: #27ae60; }
        .graph-node.blocked rect { fill: #2a1a1a; stroke: #e74c3c; opacity: 0.8; }
        .graph-node.error rect { fill: #2a1a1a; stroke: #e74c3c; }

        .node-label { fill: #667eea; font-size: 12px; font-weight: 600; }
        .node-status-text { font-size: 9px; text-transform: uppercase; }
        .node-preview { fill: #888; font-size: 10px; }
        .node-deps { fill: #e74c3c; font-size: 9px; font-style: italic; }

        .edge { fill: none; stroke: #444; stroke-width: 2; }
        .edge.completed { stroke: #27ae60; }
        .edge-arrow { fill: #444; }
        .edge-arrow.completed { fill: #27ae60; }

        .dep-edge { fill: none; stroke: #e74c3c; stroke-width: 1.5; stroke-dasharray: 5,5; opacity: 0.6; }
        .dep-edge.resolved { stroke: #27ae60; stroke-dasharray: none; opacity: 0.4; }

        /* Detail Panel */
        .detail-panel { display: none; }
        .detail-panel.active { display: block; }
        .detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        .detail-slot { font-size: 1.2rem; font-weight: 600; color: #667eea; }
        .detail-status {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
        }
        .detail-status.pending { background: #444; color: #aaa; }
        .detail-status.running { background: #f39c12; color: #000; }
        .detail-status.assembling { background: #9b59b6; color: #fff; }
        .detail-status.completed { background: #27ae60; color: #fff; }
        .detail-status.blocked { background: #e74c3c; color: #fff; }

        .detail-section-block { margin-bottom: 15px; }
        .detail-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 6px;
            font-weight: 600;
        }
        .detail-content {
            background: #0a0a12;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #ccc;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 200px;
            overflow-y: auto;
        }
        .detail-content.input { border-left: 3px solid #3498db; }
        .detail-content.output { border-left: 3px solid #27ae60; }
        .detail-content.template { border-left: 3px solid #f39c12; }
        .detail-content.deps { border-left: 3px solid #e74c3c; }

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

        .no-selection { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌳 Slowly - Task Tree Visualizer</h1>

        <div class="input-section">
            <div class="input-row">
                <input type="text" id="problem-input" placeholder="Enter your research question..."
                       value="A store sells apples at $2 each and oranges at $3 each. If I buy 15 fruits for $38, how many of each?">
                <button id="start-btn" onclick="startRun()">Start</button>
            </div>
        </div>

        <div id="run-status"></div>

        <div class="main-content">
            <div class="graph-section">
                <h2>Task Graph</h2>
                <svg id="graph-svg"></svg>
            </div>

            <div class="detail-section">
                <h2>Node Details</h2>
                <div id="detail-placeholder" class="no-selection">Click a node to view details...</div>
                <div id="detail-panel" class="detail-panel"></div>
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
        let selectedNodeId = null;

        // Graph layout constants
        const NODE_WIDTH = 160;
        const NODE_HEIGHT = 70;
        const LEVEL_HEIGHT = 120;
        const NODE_GAP = 30;

        function startRun() {
            const problem = document.getElementById('problem-input').value.trim();
            if (!problem) return;

            document.getElementById('start-btn').disabled = true;
            document.getElementById('graph-svg').innerHTML = '';
            document.getElementById('output-section').style.display = 'none';
            document.getElementById('detail-panel').className = 'detail-panel';
            document.getElementById('detail-placeholder').style.display = 'block';
            treeData = {};
            selectedNodeId = null;

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
            // Handle init
            if (event.type === 'init' && event.run) {
                treeData = event.run.tree || {};
                renderGraph();
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
                    template_question: null,
                    dependencies: [],
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
                    template_question: event.template_question || event.question || '',
                    dependencies: event.dependencies || [],
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
            else if (event.type === 'leaf_error' && event.task_id) {
                if (treeData[event.task_id]) {
                    treeData[event.task_id].status = 'error';
                    treeData[event.task_id].error = event.error || 'Unknown error';
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

            renderGraph();
            if (selectedNodeId && treeData[selectedNodeId]) {
                showNodeDetail(selectedNodeId);
            }
        }

        function calculateLayout() {
            const nodes = Object.values(treeData);
            if (nodes.length === 0) return { positions: {}, width: 0, height: 0 };

            const positions = {};
            const levels = {};

            // Find root
            const root = nodes.find(n => !n.parent_id);
            if (!root) return { positions: {}, width: 0, height: 0 };

            // BFS to assign levels
            const queue = [{ node: root, level: 0 }];
            while (queue.length > 0) {
                const { node, level } = queue.shift();
                if (!levels[level]) levels[level] = [];
                levels[level].push(node);

                for (const childId of node.children) {
                    if (treeData[childId]) {
                        queue.push({ node: treeData[childId], level: level + 1 });
                    }
                }
            }

            // Calculate positions
            const maxLevel = Math.max(...Object.keys(levels).map(Number));
            let maxWidth = 0;

            for (let level = 0; level <= maxLevel; level++) {
                const nodesAtLevel = levels[level] || [];
                const levelWidth = nodesAtLevel.length * (NODE_WIDTH + NODE_GAP) - NODE_GAP;
                maxWidth = Math.max(maxWidth, levelWidth);
            }

            for (let level = 0; level <= maxLevel; level++) {
                const nodesAtLevel = levels[level] || [];
                const levelWidth = nodesAtLevel.length * (NODE_WIDTH + NODE_GAP) - NODE_GAP;
                const startX = (maxWidth - levelWidth) / 2 + 50;

                nodesAtLevel.forEach((node, idx) => {
                    positions[node.id] = {
                        x: startX + idx * (NODE_WIDTH + NODE_GAP),
                        y: 50 + level * LEVEL_HEIGHT
                    };
                });
            }

            return {
                positions,
                width: maxWidth + 100,
                height: 100 + (maxLevel + 1) * LEVEL_HEIGHT
            };
        }

        function renderGraph() {
            const svg = document.getElementById('graph-svg');
            const { positions, width, height } = calculateLayout();

            if (Object.keys(positions).length === 0) {
                svg.innerHTML = '<text x="50" y="50" fill="#666">Waiting for tasks...</text>';
                return;
            }

            svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
            svg.style.minHeight = `${height}px`;

            let html = '';

            // Draw edges first (so they're behind nodes)
            for (const node of Object.values(treeData)) {
                if (node.parent_id && positions[node.parent_id] && positions[node.id]) {
                    const parent = positions[node.parent_id];
                    const child = positions[node.id];
                    const parentCompleted = treeData[node.parent_id]?.status === 'completed';

                    const startX = parent.x + NODE_WIDTH / 2;
                    const startY = parent.y + NODE_HEIGHT;
                    const endX = child.x + NODE_WIDTH / 2;
                    const endY = child.y;
                    const midY = (startY + endY) / 2;

                    html += `<path class="edge ${parentCompleted ? 'completed' : ''}"
                        d="M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}" />`;

                    // Arrow
                    html += `<polygon class="edge-arrow ${parentCompleted ? 'completed' : ''}"
                        points="${endX-5},${endY-8} ${endX+5},${endY-8} ${endX},${endY}" />`;
                }
            }

            // Draw dependency edges between siblings
            for (const node of Object.values(treeData)) {
                if (node.dependencies && node.dependencies.length > 0 && node.parent_id) {
                    // Find siblings that match dependency names
                    const siblings = Object.values(treeData).filter(s =>
                        s.parent_id === node.parent_id && s.id !== node.id
                    );

                    for (const depSlot of node.dependencies) {
                        const depNode = siblings.find(s => s.slot === depSlot);
                        if (depNode && positions[depNode.id] && positions[node.id]) {
                            const from = positions[depNode.id];
                            const to = positions[node.id];
                            const resolved = depNode.status === 'completed';

                            // Draw curved dependency edge
                            const fromX = from.x + NODE_WIDTH;
                            const fromY = from.y + NODE_HEIGHT / 2;
                            const toX = to.x;
                            const toY = to.y + NODE_HEIGHT / 2;

                            html += `<path class="dep-edge ${resolved ? 'resolved' : ''}"
                                d="M ${fromX} ${fromY} Q ${(fromX + toX) / 2} ${fromY - 30}, ${toX} ${toY}" />`;
                        }
                    }
                }
            }

            // Draw nodes
            for (const node of Object.values(treeData)) {
                const pos = positions[node.id];
                if (!pos) continue;

                const isSelected = node.id === selectedNodeId;
                const preview = (node.input || '').slice(0, 25) + ((node.input || '').length > 25 ? '...' : '');
                const deps = node.dependencies?.length > 0 ? `{${node.dependencies.join(', ')}}` : '';

                html += `
                    <g class="graph-node ${node.status}" transform="translate(${pos.x}, ${pos.y})"
                       onclick="selectNode('${node.id}')" ${isSelected ? 'style="filter: brightness(1.3);"' : ''}>
                        <rect x="0" y="0" width="${NODE_WIDTH}" height="${NODE_HEIGHT}"
                              ${isSelected ? 'stroke-width="3"' : ''} />
                        <text class="node-label" x="${NODE_WIDTH/2}" y="18" text-anchor="middle">${escapeHtml(node.slot)}</text>
                        <text class="node-status-text" x="${NODE_WIDTH/2}" y="32" text-anchor="middle" fill="${getStatusColor(node.status)}">${node.status}</text>
                        <text class="node-preview" x="${NODE_WIDTH/2}" y="48" text-anchor="middle">${escapeHtml(preview)}</text>
                        ${deps ? `<text class="node-deps" x="${NODE_WIDTH/2}" y="62" text-anchor="middle">${escapeHtml(deps)}</text>` : ''}
                    </g>
                `;
            }

            svg.innerHTML = html;
        }

        function getStatusColor(status) {
            const colors = {
                pending: '#888',
                running: '#f39c12',
                assembling: '#9b59b6',
                completed: '#27ae60',
                blocked: '#e74c3c',
                error: '#e74c3c'
            };
            return colors[status] || '#888';
        }

        function selectNode(nodeId) {
            selectedNodeId = nodeId;
            renderGraph();
            showNodeDetail(nodeId);
        }

        function showNodeDetail(nodeId) {
            const node = treeData[nodeId];
            if (!node) return;

            document.getElementById('detail-placeholder').style.display = 'none';
            const panel = document.getElementById('detail-panel');
            panel.className = 'detail-panel active';

            let html = `
                <div class="detail-header">
                    <span class="detail-slot">${escapeHtml(node.slot)}</span>
                    <span class="detail-status ${node.status}">${node.status}</span>
                </div>
            `;

            // Template question (shows {slot} placeholders)
            if (node.template_question && node.template_question !== node.input) {
                html += `
                    <div class="detail-section-block">
                        <div class="detail-label">Template (with dependencies)</div>
                        <div class="detail-content template">${escapeHtml(node.template_question)}</div>
                    </div>
                `;
            }

            // Dependencies
            if (node.dependencies && node.dependencies.length > 0) {
                html += `
                    <div class="detail-section-block">
                        <div class="detail-label">Depends On</div>
                        <div class="detail-content deps">${node.dependencies.map(d => `{${d}}`).join(', ')}</div>
                    </div>
                `;
            }

            // Input (resolved question)
            if (node.input) {
                html += `
                    <div class="detail-section-block">
                        <div class="detail-label">Input (Resolved Question)</div>
                        <div class="detail-content input">${escapeHtml(node.input)}</div>
                    </div>
                `;
            }

            // Output
            if (node.output) {
                html += `
                    <div class="detail-section-block">
                        <div class="detail-label">Output</div>
                        <div class="detail-content output">${escapeHtml(node.output)}</div>
                    </div>
                `;
            }

            // Assembled inputs (for parent nodes)
            if (node.inputs && Object.keys(node.inputs).length > 0) {
                const inputsHtml = Object.entries(node.inputs)
                    .map(([k, v]) => `<strong>${k}:</strong>\\n${escapeHtml(v)}`)
                    .join('\\n\\n---\\n\\n');
                html += `
                    <div class="detail-section-block">
                        <div class="detail-label">Slot Inputs (from children)</div>
                        <div class="detail-content">${inputsHtml}</div>
                    </div>
                `;
            }

            // Error
            if (node.error) {
                html += `
                    <div class="detail-section-block">
                        <div class="detail-label">Error</div>
                        <div class="detail-content" style="border-left-color: #e74c3c;">${escapeHtml(node.error)}</div>
                    </div>
                `;
            }

            panel.innerHTML = html;
        }

        function escapeHtml(str) {
            if (!str) return '';
            return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
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
