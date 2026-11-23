"""Prompt-only guidance. Native result acceptance and scores are untouched."""
from pathlib import Path
import hashlib
import json

SYSTEM_PROMPT = Path(__file__).with_name('judge_system_prompt.txt').read_text()

TABLE_GUIDANCE = '''
Judge evidence retrieval and completion:
- --grep is a text search returning matching lines plus nearby context. Searching a sheet name
  does NOT read the sheet, even when truncated=false (that flag describes the search output).
- For Excel use: python tools/inspect_workspace_file.py output:1 --sheet "Exact Sheet Name" --range A1:H20
  Choose only cells needed for the rubric; inspect the returned sheet dimensions and selection flags.
- Do not repeat an identical inspection after it returned the same evidence. If evidence is missing,
  change the sheet/range or search the actual value/label. Keep a short record of established findings.
- Once rubric evidence is sufficient, stop retrieving and write judge_result.json. Within the existing
  turn budget, reserve time to write the result. State unverified points honestly; do not invent evidence,
  force a zero for uncertainty, change rubric weights, or mark uninspected contents as correct.
'''


FORMAT_REPAIR = """The preceding attempt ended with plain text resembling a tool call, not a tool invocation or a complete scoring result. Do not emit native tool-call delimiter text. To inspect a file, invoke an actual tool from the provided tool schema; for the documented Python inspector use the Bash tool with its command argument. A shell command written in a final response is not executed. Continue evidence-based evaluation and write and return the original complete judge_result.json schema. Do not infer scores from this format error or change the rubric."""

def ended_with_unexecuted_tool_text(log_path):
    try:
        events = [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError, TypeError):
        return False
    results = [e for e in events if isinstance(e, dict) and e.get('type') == 'result']
    if not results:
        return False
    event = results[-1]
    result = event.get('result')
    return (event.get('subtype') == 'success' and not event.get('is_error')
            and isinstance(result, str) and result.lstrip().startswith('<|tool_call>'))

def install(native):
    if getattr(native, '_prompt_only_installed', False):
        raise RuntimeError('Prompt-only adapter already installed')
    original_run = native.run_judge
    original_instructions = native.judge_instructions
    def run(workspace, task_dir, case_dir, args, *, attempt_runner=None, **kwargs):
        # Prepared workspaces may contain instructions from an earlier judge protocol.
        # Restore the native instruction files in this NEW independent copy only.
        for name in ('CLAUDE.md', 'AGENTS.md'):
            (Path(workspace)/name).write_text(original_instructions() + '\n' + TABLE_GUIDANCE)
        process = attempt_runner or native.run_process
        previous_format_error = False
        def with_system_prompt(command, **process_kwargs):
            nonlocal previous_format_error
            command = list(command)
            command[1:1] = ['--system-prompt', SYSTEM_PROMPT + '\n\n' + original_instructions() + '\n' + TABLE_GUIDANCE]
            if previous_format_error:
                prompt_index = command.index('-p') + 1
                command[prompt_index] += '\n\n## Tool invocation format repair\n' + FORMAT_REPAIR
            outcome = process(command, **process_kwargs)
            previous_format_error = ended_with_unexecuted_tool_text(process_kwargs['log_path'])
            return outcome
        native.write_json(Path(case_dir)/'prompt-only-provenance.json', {
            'mode': 'dedicated_judge_system_prompt_with_native_instructions',
            'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            'extra_result_validators': False,
            'bounded_retry_format_feedback': True,
            'native_acceptance_unchanged': True,
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })
        # No postprocessing, validation, score replacement, or additional retries.
        return original_run(workspace, task_dir, case_dir, args,
                            attempt_runner=with_system_prompt, **kwargs)
    native.run_judge = run
    native._prompt_only_installed = True
