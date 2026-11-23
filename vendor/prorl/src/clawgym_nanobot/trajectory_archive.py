"""Synchronous evidence capture before cleanup; distinct train/eval sessions."""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import time


def capture(task, result, rollout_id, task_index, evaluation):
    base = Path(os.environ['CLAWGYM_TRAJECTORY_ARCHIVE_DIR'])
    phase = 'eval' if evaluation else 'train'
    # Preserve every task in this short diagnostic run (includes every fifth
    # step). Unique run directory distinguishes retries without overwriting.
    match = re.fullmatch(r'(train|eval)-(\d+)-(\d+)(?:-[A-Za-z0-9]+)?', result.session_id)
    if not match or match.group(1) != phase: raise ValueError('unexpected diagnostic session identity')
    outer_step, batch_index = int(match.group(2)), int(match.group(3))
    dest = base / phase / f'step-{outer_step:03d}' / result.session_id / Path(result.run_dir).name
    dest.mkdir(parents=True, exist_ok=False)
    files = {}
    def save(src, rel):
        data = src.read_bytes()
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[str(rel)] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    for p in Path(result.log_dir).glob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix in {'.json', '.jsonl', '.log', '.txt'}:
            save(p, Path('solver-and-judge') / p.name)
    for name in ['judge_result.json','artifact-validity-gate.json','judge-failure.json']:
        p = Path(result.run_dir) / name
        if p.is_file(): save(p, Path('reward') / name)
    rawroot = Path(os.environ['CLAWGYM_RAW_TURN_EVIDENCE_DIR'])
    raw = rawroot / hashlib.sha256(result.session_id.encode()).hexdigest()
    for p in raw.glob('*.json'):
        save(p, Path('raw-turns') / p.name)
    reward_record = getattr(result, 'evaluation_reward', None) or {}
    native = reward_record.get('native_result') or {}
    for key in ['judge_log_path', 'judge_cleaned_transcript_path']:
        if native.get(key):
            p = Path(native[key])
            if not p.resolve().is_relative_to(Path(result.run_dir).resolve()):
                raise ValueError('judge evidence path outside this task run')
            if not p.is_file(): raise FileNotFoundError(f'missing judge evidence: {key}')
            save(p, Path('judge') / p.name)
    behavior = {'gate_failed': (reward_record.get('artifact_validity_gate') or {}).get('passed') is False,
                'raw_judge_score': reward_record.get('raw_judge_score'), 'gated_score': result.reward,
                'unparsed_native_call_steps': [], 'tool_error_steps': [],
                'source_fabrication_feedback_candidate': False,
                'note': 'Heuristic review flags only; do not alter rewards or prove intent/causality.'}
    ends = list(Path(result.log_dir).glob('*agent-end.json'))
    if ends:
        trace = json.loads(max(ends, key=lambda p:p.name).read_text())
        for i, step in enumerate(trace.get('llm_steps', [])):
            text = str(step.get('response_content') or '')
            if re.match(r'^\s*<\|tool_call>\s*call:', text) and not step.get('response_tool_calls'):
                behavior['unparsed_native_call_steps'].append(i)
        for i, step in enumerate(trace.get('tool_steps', [])):
            text = str(step.get('result') or '')
            if text.startswith('Error:') or re.search(r'Exit code:\s*[1-9]\d*', text):
                behavior['tool_error_steps'].append(i)
    feedback = str(native.get('feedback') or '') + ' ' + str(native.get('missing_or_wrong') or '')
    behavior['source_fabrication_feedback_candidate'] = bool(re.search(r'fabricat|hallucinat|invented|伪造|编造', feedback, re.I))
    behavior['invalid_artifact_positive_raw_reward'] = bool(behavior['gate_failed'] and (behavior['raw_judge_score'] or 0) > 0)
    (dest/'behavior-review.json').write_text(json.dumps(behavior, ensure_ascii=False, indent=2))
    contract_path = Path(task.task_dir) / 'reward/output_contract.json'
    contract = json.loads(contract_path.read_text())
    save(contract_path, Path('output-contract.json'))
    root = Path(result.workspace_dir).resolve()
    outputs = []
    for item in contract['output_files']:
        if not item.get('must_create', True) or not item.get('path'): continue
        rel = Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe contract path')
        p = root / rel
        readable = p.is_file() and p.resolve().is_relative_to(root)
        outputs.append({'path': str(rel), 'present': readable})
        if readable: save(p, Path('outputs') / rel)
    manifest = dict(version=1, at=time.time(), phase=phase, rollout_id=rollout_id,
                    task_index=task_index, optimizer_rollout_id=outer_step, task_index_in_batch=batch_index, task_id=task.task_id, session_id=result.session_id,
                    status=result.status, reward=result.reward, outputs=outputs, files=files,
                    archived_before_cleanup=True)
    (dest / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return str(dest)
