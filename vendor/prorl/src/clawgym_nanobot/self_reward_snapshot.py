"""Collect task-generated answer files without importing solver runtime or reward assets."""
from pathlib import Path

RUNTIME = {'sessions', 'memory', 'skills', 'tools', '__pycache__', 'node_modules'}
INPUTS = {'input', 'input_files', 'paper_lib', 'repo_lib', 'sources'}
PRIVATE = {'reward', 'judge', 'reference', 'references_private'}
SCAFFOLD = {'AGENTS.md', 'SOUL.md', 'USER.md', 'HEARTBEAT.md', 'IDENTITY.md',
            'TOOLS.md', 'WORKSPACE.md', 'task.md', 'task_context.json'}

def collect(workspace, initial_files, required=()):
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise ValueError('Original answer workspace is unavailable')
    initial = set(initial_files)
    required = [str(p) for p in required]
    mapping = {}
    counts = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        reason = None
        if any(x.startswith('.') or x in RUNTIME | PRIVATE for x in parts):
            reason = 'runtime_or_private'
        elif parts[0] in INPUTS:
            reason = 'input_available_under_sources'
        elif len(parts) == 1 and rel.name in SCAFFOLD:
            reason = 'solver_instructions'
        elif not p.resolve().is_relative_to(root):
            reason = 'external_link'
        elif any(x in PRIVATE for x in p.resolve().relative_to(root).parts):
            reason = 'private_link'
        elif str(rel) in initial and str(rel) not in required:
            reason = 'initial_file'
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
            continue
        mapping[str(rel)] = {'path': str(p), 'resolved': str(p.resolve()), 'bytes': p.stat().st_size}
    return mapping, {'included': sorted(mapping), 'excluded_counts': counts,
                     'required_outputs': [{'path': x, 'included': x in mapping} for x in required],
                     'scope': 'Generated answer files with original relative paths; inputs are separately staged under sources. Runtime, private reward assets and solver instructions are excluded.'}
