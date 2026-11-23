"""Minimal required-deliverable format gate. Never executes submitted content.

The trusted task contract supplies required paths. Content quality remains the
native judge's responsibility. Validation runs with the existing document stack.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

SUPPORTED = {'docx', 'xlsx', 'pptx', 'pdf', 'json', 'yaml', 'yml', 'md', 'txt', 'py', 'html', 'htm'}


def check_files(workspace, required):
    # Import dependencies outside artifact error handling: missing libraries are
    # infrastructure errors, never evidence that a model output is invalid.
    import docx
    import openpyxl
    import pptx
    import pypdf
    import yaml
    root = Path(workspace).resolve(strict=True)
    rows = []
    for item in required:
        rel = PurePosixPath(item['path'])
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('unsafe path in trusted output contract')
        fmt = str(item.get('format') or rel.suffix.lstrip('.')).lower().lstrip('.')
        fmt = {'markdown': 'md'}.get(fmt, fmt)
        if fmt not in SUPPORTED:
            raise ValueError(f'unsupported trusted contract format: {fmt}')
        row = {'path': rel.as_posix(), 'format': fmt, 'valid': False}
        path = root / rel
        if not path.exists():
            row['reason'] = 'required_file_missing'
        elif not path.resolve().is_relative_to(root):
            row['reason'] = 'output_outside_workspace'
        elif not path.is_file():
            row['reason'] = 'required_path_not_file'
        else:
            # Read failures propagate; do not label I/O outages as bad outputs.
            blob = path.read_bytes()
            row.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
            try:
                # CRC detects broken package members; real format readers check
                # relationships/content types, not just ZIP magic or extension.
                if fmt in {'docx', 'xlsx', 'pptx'}:
                    with zipfile.ZipFile(path) as z:
                        bad = z.testzip()
                        if bad: raise ValueError(f'corrupt package member: {bad}')
                    if fmt == 'docx':
                        docx.Document(path)
                    elif fmt == 'xlsx':
                        book = openpyxl.load_workbook(path, read_only=False, data_only=False)
                        book.close()
                    else:
                        pptx.Presentation(path)
                elif fmt == 'pdf':
                    reader = pypdf.PdfReader(path, strict=False)
                    if reader.is_encrypted and not reader.decrypt(''):
                        raise ValueError('required PDF is password locked')
                    # Traverse pages to expose broken structure, without requiring
                    # any minimum number of pages or amount of textual content.
                    list(reader.pages)
                elif fmt == 'json':
                    json.loads(blob.decode('utf-8-sig'))
                elif fmt in {'yaml', 'yml'}:
                    list(yaml.safe_load_all(blob.decode('utf-8-sig')))
                elif fmt == 'py':
                    # Compile bytes to respect Python encoding declarations. Never execute.
                    compile(blob, str(path), 'exec', dont_inherit=True)
                else:
                    # Text/HTML semantic correctness belongs to the judge.
                    blob.decode('utf-8-sig')
                row.update(valid=True, reason='format_readable')
            except (OSError, MemoryError):
                raise
            except Exception as exc:
                row.update(reason='invalid_format', detail=f'{type(exc).__name__}: {exc}'[:600])
        rows.append(row)
    if not rows:
        raise ValueError('trusted task contract has no required outputs')
    return {'version': 2, 'passed': all(r['valid'] for r in rows), 'artifacts': rows,
            'scope': 'required file existence/readable format and Python syntax only; no execution, content-length or style thresholds'}


def apply_gate(task, workspace_dir, run_dir, raw_score, native_result):
    contract = json.loads((Path(task.task_dir) / 'reward/output_contract.json').read_text())
    required = [dict(x) for x in contract['output_files']
                if isinstance(x, dict) and x.get('must_create', True) and x.get('path')]
    request = {'workspace': str(workspace_dir), 'required': required}
    proc = subprocess.run([os.environ['WSBENCH_PYTHON'], str(Path(__file__).resolve()), '--check'],
                          input=json.dumps(request), capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError('artifact validator infrastructure failure: ' + proc.stderr[-2000:])
    gate = json.loads(proc.stdout)
    score = float(raw_score) if gate['passed'] else 0.0
    result = dict(native_result)
    result.update(raw_judge_score=float(raw_score), score=score, gated_score=score,
                  artifact_validity_gate=gate)
    # Keep the original native judge result nested and completely unchanged.
    for name, data in [('artifact-validity-gate.json', gate), ('judge_result.json', result)]:
        path = Path(run_dir) / name
        tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(path)
    return score, result


if __name__ == '__main__':
    if sys.argv[1:] != ['--check']: raise SystemExit('expected --check')
    req = json.load(sys.stdin)
    print(json.dumps(check_files(req['workspace'], req['required']), ensure_ascii=False))
