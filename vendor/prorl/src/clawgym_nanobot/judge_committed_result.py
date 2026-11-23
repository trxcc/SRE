"""Verify a committed native score followed only by a local context-limit exit.
No model calls, score changes, retries, or changes to compression behavior.
"""
import hashlib
import json
import math
from pathlib import Path


def verify_committed_result(native):
    try:
        if not isinstance(native, dict): return None
        if not (native.get('judge_result_complete') is True
                and native.get('judge_schema_complete') is True
                and native.get('judge_source_coverage', {}).get('complete') is True
                and native.get('judge_exit_code') == 1
                and native.get('judge_timed_out') is False
                and native.get('judge_permanent_api_error_status') is None
                and native.get('judge_attempts') == 1): return None
        metrics = native.get('judge_model_metrics', {})
        count = metrics.get('request_count', 0)
        if count <= 0 or metrics.get('successful_request_count') != count: return None
        if not math.isfinite(float(native['score'])) or not 0 <= native['score'] <= 1: return None
        path = Path(native['judge_log_path'])
        raw = path.read_bytes()
        events = []
        for line in raw.decode().splitlines():
            try: event = json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(event, dict): events.append(event)
        terminals = [(i, e) for i, e in enumerate(events) if e.get('type') == 'result']
        if len(terminals) != 1: return None
        end, terminal = terminals[0]
        if end != len(events)-1: return None
        if not (terminal.get('is_error') is True and terminal.get('subtype') == 'success'
                and terminal.get('terminal_reason') == 'blocking_limit'
                and terminal.get('result') == 'Prompt is too long'
                and terminal.get('api_error_status') is None
                and 0 < terminal.get('num_turns', 0) <= 30): return None
        sid = terminal.get('session_id')
        if not sid or any(e.get('session_id', sid) != sid for e in events): return None
        writes = {}
        committed = None
        for i, event in enumerate(events[:end]):
            msg = event.get('message') or {}
            parts = msg.get('content') or []
            if not isinstance(parts, list): parts = []
            if event.get('type') == 'assistant':
                for part in parts:
                    if part.get('type') == 'tool_use' and part.get('name') == 'Write':
                        arg = part.get('input') or {}
                        if arg.get('file_path') == 'judge_result.json':
                            writes[part['id']] = (i, arg['content'])
            result = event.get('tool_use_result') or {}
            if event.get('type') == 'user' and result.get('type') == 'create' and result.get('filePath') == 'judge_result.json':
                matches = [p for p in parts if p.get('type') == 'tool_result' and p.get('tool_use_id') in writes and not p.get('is_error', False)]
                if len(matches) != 1 or committed is not None: return None
                ident = matches[0]['tool_use_id']
                wi, content = writes[ident]
                if wi >= i or result.get('content') != content or result.get('userModified') is not False: return None
                written = json.loads(content)
                # Native evaluator derives the scalar from the original rubric;
                # compare every authored field except its rounded scalar score.
                if any(native.get(k) != v for k, v in written.items() if k != 'score'): return None
                scores = written.get('rubric_scores')
                if not isinstance(scores, dict) or not scores: return None
                values = [float(v) for v in scores.values()]
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in values): return None
                if not math.isclose(sum(values)/len(values), native['score'], abs_tol=1e-12): return None
                if not math.isfinite(float(written['score'])) or abs(float(written['score'])-native['score']) > .005: return None
                committed = (i, ident, event.get('timestamp'))
        if committed is None: return None
        ci, ident, timestamp = committed
        following = events[ci+1:end]
        if not any(e.get('type') == 'system' and e.get('compact_result') == 'failed'
                   and e.get('compact_error') == 'summarization produced empty response' for e in following): return None
        for e in following:
            if e.get('type') == 'system': continue
            if e.get('type') != 'assistant' or e.get('error') != 'invalid_request' or e.get('is_api_error_message') is not True: return None
            parts = (e.get('message') or {}).get('content', [])
            if not isinstance(parts, list) or not parts or any(p.get('type') != 'text' or p.get('text') != 'Prompt is too long' for p in parts): return None
        return dict(kind='native_score_committed_before_local_context_limit',
                    original_exit_code=1, score_unchanged=True, judge_log_path=str(path),
                    judge_log_sha256=hashlib.sha256(raw).hexdigest(), tool_use_id=ident,
                    commit_timestamp=timestamp, successful_requests=count)
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None
