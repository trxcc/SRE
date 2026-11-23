"""Remove executable-looking text examples from the generated solver contract."""
import re
from pathlib import Path


INSTRUCTION = (
    "Inspect the required source paths listed above using the provided "
    "inspect_workspace_file tool through the tool-calling interface. Supply "
    "the source path in its path argument and wait for the actual tool result. "
    "Writing a tool name or a Python-style function call in ordinary response "
    "text does not execute a tool."
)


def rewrite_source_inspection_examples(text):
    start = text.find("### Source Inspection\n")
    if start < 0:
        raise ValueError("generated task contract has no Source Inspection section")
    stop = text.find("### Tool Call Discipline", start)
    if stop < 0:
        raise ValueError("generated task contract has no Tool Call Discipline boundary")
    section = text[start:stop]
    pattern = r'```text\n(?:inspect_workspace_file\(path="[^\n]*"\)\n)*```'
    section, count = re.subn(pattern, INSTRUCTION, section)
    if count == 0 and INSTRUCTION in section:
        return text
    if count != 1:
        raise ValueError(f"expected one generated source-inspection example block, got {count}")
    return text[:start] + section + text[stop:]


def prepare_gemma_tool_prompt(workspace):
    path = Path(workspace) / "AGENTS.md"
    before = path.read_text(encoding="utf-8")
    after = rewrite_source_inspection_examples(before)
    path.write_text(after, encoding="utf-8")
