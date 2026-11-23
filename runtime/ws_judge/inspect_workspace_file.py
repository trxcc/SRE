#!/usr/bin/env python3
"""Augment the WSClaw inspector with evidence-complete structured previews."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


BASE_INSPECTOR_NAME = "_inspect_workspace_file_base.py"
try:
    DEFAULT_MAX_CHARS = int(os.environ.get("WSCLAW_INSPECT_DEFAULT_MAX_CHARS", "12000"))
except ValueError:
    DEFAULT_MAX_CHARS = 12000
DEFAULT_MAX_CHARS = max(1000, min(DEFAULT_MAX_CHARS, 100000))
MAX_FILESYSTEM_ATTEMPTS = 3
TRANSIENT_FILESYSTEM_MARKERS = (
    "input/output error",
    "cannot read file data",
    "no such file or directory",
    "stale file handle",
    "transport endpoint is not connected",
    "temporarily unavailable",
    "file not found",
)


def bounded_preview(lines: list[str], max_chars: int) -> tuple[str, bool]:
    output: list[str] = []
    used = 0
    for line in lines:
        separator = 1 if output else 0
        remaining = max_chars - used - separator
        if remaining <= 0:
            return "\n".join(output), True
        if len(line) > remaining:
            output.append(line[:remaining])
            return "\n".join(output), True
        output.append(line)
        used += separator + len(line)
    return "\n".join(output), False


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def workbook_preview(path: Path, max_chars: int) -> tuple[str, bool, dict[str, Any]]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True, keep_links=False)
    lines: list[str] = []
    row_counts: dict[str, int] = {}
    try:
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            lines.append(
                f"## sheet {sheet_name!r} ({sheet.max_row or 0}r x {sheet.max_column or 0}c)"
            )
            seen = 0
            for row_index, values in enumerate(sheet.iter_rows(values_only=True), start=1):
                row = [cell_text(value) for value in values]
                while row and row[-1] == "":
                    row.pop()
                lines.append(
                    f"r{row_index}: {json.dumps(row, ensure_ascii=False)}"
                )
                seen += 1
            row_counts[sheet_name] = seen
    finally:
        workbook.close()
    preview, truncated = bounded_preview(lines, max_chars)
    return preview, truncated, {"preview_row_counts": row_counts}


def document_preview(path: Path, max_chars: int) -> tuple[str, bool, dict[str, Any]]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    lines: list[str] = []
    table_count = 0
    table_row_counts: list[int] = []
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            text = (Paragraph(child, document).text or "").strip()
            if text:
                lines.append(text)
            continue
        if not child.tag.endswith("}tbl"):
            continue
        table_count += 1
        table = Table(child, document)
        lines.append(f"## table {table_count} ({len(table.rows)} rows)")
        table_row_counts.append(len(table.rows))
        for row_index, row in enumerate(table.rows, start=1):
            values = [cell_text(cell.text).strip() for cell in row.cells]
            lines.append(
                f"table{table_count}.r{row_index}: "
                f"{json.dumps(values, ensure_ascii=False)}"
            )
    preview, truncated = bounded_preview(lines, max_chars)
    return preview, truncated, {
        "preview_table_count": table_count,
        "preview_table_row_counts": table_row_counts,
    }


def presentation_preview(path: Path, max_chars: int) -> tuple[str, bool, dict[str, Any]]:
    from pptx import Presentation

    presentation = Presentation(str(path))
    lines: list[str] = []
    table_count = 0
    chart_count = 0
    for slide_index, slide in enumerate(presentation.slides, start=1):
        title = ""
        if slide.shapes.title is not None:
            title = (slide.shapes.title.text or "").strip()
        lines.append(f"## slide {slide_index}: {title or '(no title)'}")
        for shape in slide.shapes:
            if getattr(shape, "has_table", False):
                table_count += 1
                table = shape.table
                lines.append(
                    f"### slide{slide_index}.table{table_count} "
                    f"({len(table.rows)} rows x {len(table.columns)} cols)"
                )
                for row_index, row in enumerate(table.rows, start=1):
                    values = [cell_text(cell.text).strip() for cell in row.cells]
                    lines.append(
                        f"slide{slide_index}.table{table_count}.r{row_index}: "
                        f"{json.dumps(values, ensure_ascii=False)}"
                    )
                continue
            if getattr(shape, "has_chart", False):
                chart_count += 1
                chart = shape.chart
                lines.append(f"### slide{slide_index}.chart{chart_count}")
                for plot_index, plot in enumerate(chart.plots, start=1):
                    categories = [
                        cell_text(getattr(category, "label", category))
                        for category in plot.categories
                    ]
                    lines.append(
                        f"chart{chart_count}.plot{plot_index}.categories: "
                        f"{json.dumps(categories, ensure_ascii=False)}"
                    )
                    for series in plot.series:
                        values = [cell_text(value) for value in series.values]
                        lines.append(
                            f"chart{chart_count}.{cell_text(series.name)}: "
                            f"{json.dumps(values, ensure_ascii=False)}"
                        )
                continue
            if getattr(shape, "has_text_frame", False):
                text = (shape.text or "").strip()
                if text and text != title:
                    lines.append(text)
    preview, truncated = bounded_preview(lines, max_chars)
    return preview, truncated, {
        "preview_slide_count": len(presentation.slides),
        "preview_table_count": table_count,
        "preview_chart_count": chart_count,
    }


def tabular_preview(path: Path, max_chars: int) -> tuple[str, bool, dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    lines: list[str] = []
    row_count = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row_count, row in enumerate(reader, start=1):
            lines.append(
                f"r{row_count}: {json.dumps(row, ensure_ascii=False)}"
            )
    preview, truncated = bounded_preview(lines, max_chars)
    return preview, truncated, {"preview_row_count": row_count}


def json_preview(path: Path, max_chars: int) -> tuple[str, bool, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(value, ensure_ascii=False, indent=2)
    preview, truncated = bounded_preview(text.splitlines(), max_chars)
    return preview, truncated, {"preview_complete_json": not truncated}


def text_preview(path: Path, max_chars: int) -> tuple[str, bool, dict[str, Any]]:
    raw = path.read_bytes()
    encoding = "utf-8-sig"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        try:
            import chardet

            encoding = chardet.detect(raw[:65536]).get("encoding") or "utf-8"
        except Exception:
            encoding = "utf-8"
        text = raw.decode(encoding, errors="replace")
    lines = text.splitlines()
    preview, truncated = bounded_preview(lines, max_chars)
    return preview, truncated, {
        "n_lines": len(lines),
        "preview_line_count": preview.count("\n") + int(bool(preview)),
        "text_encoding": encoding,
    }


PREVIEWERS: dict[str, Callable[[Path, int], tuple[str, bool, dict[str, Any]]]] = {
    ".csv": tabular_preview,
    ".docx": document_preview,
    ".json": json_preview,
    ".pptx": presentation_preview,
    ".tsv": tabular_preview,
    ".xlsm": workbook_preview,
    ".xlsx": workbook_preview,
    ".xltm": workbook_preview,
    ".xltx": workbook_preview,
}


def enhance_result(result: dict[str, Any], path: Path, max_chars: int) -> dict[str, Any]:
    previewer = PREVIEWERS.get(path.suffix.lower())
    if previewer is None and result.get("preview_type") in {"text", "code"}:
        previewer = text_preview
    if previewer is None or result.get("parse_status") not in {"ok", "partial"}:
        return result
    for attempt in range(MAX_FILESYSTEM_ATTEMPTS):
        try:
            preview, truncated, metadata = previewer(path, max_chars)
            break
        except OSError as exc:
            if attempt + 1 < MAX_FILESYSTEM_ATTEMPTS:
                time.sleep(0.25 * (attempt + 1))
                continue
            warnings = result.setdefault("warnings", [])
            warnings.append(f"enhanced preview failed: {type(exc).__name__}: {exc}")
            return result
        except Exception as exc:
            warnings = result.setdefault("warnings", [])
            warnings.append(f"enhanced preview failed: {type(exc).__name__}: {exc}")
            return result
    result["content_preview"] = preview
    result["truncated"] = truncated
    type_specific = result.setdefault("type_specific", {})
    type_specific.update(metadata)
    type_specific["enhanced_preview"] = True
    return result


def alias_file() -> Path | None:
    raw = os.environ.get("WSCLAW_INSPECT_ALIAS_FILE")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def inspection_aliases(path: Path | None = None) -> dict[str, str]:
    path = path or alias_file()
    if path is None:
        return {}
    brief = json.loads(path.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for group in ("sources", "outputs"):
        for item in brief.get(group) or []:
            if not isinstance(item, dict) or not item.get("id") or not item.get("path"):
                continue
            logical = str(item["path"])
            candidate = Path(logical)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"unsafe inspector alias target: {logical!r}")
            aliases[str(item["id"])] = logical
    return aliases


def parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a workspace file by relative path or source:N/output:N alias."
    )
    parser.add_argument("path")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--cwd")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument(
        "--grep",
        dest="grep_pattern",
        help="regular expression used to filter the extracted preview",
    )
    parser.add_argument("--grep-context", type=int, default=2)
    parser.add_argument("--sheet", help="exact Excel sheet name; returns bounded cells, not text matches")
    parser.add_argument("--range", dest="cell_range", help="Excel A1 cell range, used with --sheet")
    args = parser.parse_args(argv)
    if args.cell_range and not args.sheet:
        parser.error('--range requires --sheet')
    if args.sheet and args.grep_pattern:
        parser.error('--sheet and --grep select different operations; use one at a time')
    args.max_chars = max(1, min(args.max_chars, 100000))
    args.grep_context = max(0, min(args.grep_context, 20))
    if args.grep_pattern:
        re.compile(args.grep_pattern)
    aliases = inspection_aliases()
    args.alias = args.path if args.path in aliases else None
    args.logical_path = aliases.get(args.path, args.path)
    logical = Path(args.logical_path)
    args.judge_mode = os.environ.get("WSCLAW_INSPECT_COMPACT") == "1" or bool(
        alias_file()
    )
    if args.judge_mode and (logical.is_absolute() or ".." in logical.parts):
        raise ValueError("judge inspect paths must use a relative path or source:N/output:N alias")
    root = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    args.resolved_path = logical.resolve() if logical.is_absolute() else (root / logical).resolve()
    return args


def grep_result(
    result: dict[str, Any], pattern: str, context: int, max_chars: int | None = None
) -> dict[str, Any]:
    expression = re.compile(pattern, re.IGNORECASE)
    lines = str(result.get("content_preview") or "").splitlines()
    matching = [index for index, line in enumerate(lines) if expression.search(line)]
    included: set[int] = set()
    for index in matching:
        included.update(range(max(0, index - context), min(len(lines), index + context + 1)))
    result["content_preview"] = "\n".join(
        f"L{index + 1}: {lines[index]}" for index in sorted(included)
    )
    result["grep"] = {
        "pattern": pattern,
        "matching_lines": len(matching),
        "returned_lines": len(included),
        "searched_preview_lines": len(lines),
        "scope": "matching text lines plus context; not a whole worksheet",
    }
    if any(lines[index].startswith('## sheet ') for index in matching):
        result['grep']['next_action'] = (
            'A sheet heading matched. Repeating this search or increasing max-chars '
            'does not select the sheet. Use --sheet EXACT_NAME --range A1:H20, '
            'then adjust the range to the rubric-relevant cells.'
        )
    if max_chars is not None:
        # Search the extracted evidence before applying the response budget.
        # A truncated input still remains explicitly incomplete (fallback formats).
        input_truncated = bool(result.get("truncated"))
        preview, output_truncated = bounded_preview(
            result["content_preview"].splitlines(), max_chars
        )
        result["content_preview"] = preview
        result["truncated"] = input_truncated or output_truncated
        result["grep"]["searched_complete_extraction"] = not input_truncated
        result["grep"]["returned_lines"] = len(preview.splitlines())
    return result


COMPACT_STRUCTURE_KEYS = (
    "n_pages",
    "n_slides",
    "n_sheets",
    "all_sheet_names",
    "n_paragraphs",
    "n_tables",
    "n_rows",
    "n_cols",
    "n_lines",
    "preview_line_count",
    "preview_row_count",
    "preview_row_counts",
    "preview_table_count",
    "preview_table_row_counts",
    "preview_slide_count",
    "preview_chart_count",
)


def compact_result(
    result: dict[str, Any], *, logical_path: str, alias: str | None
) -> dict[str, Any]:
    compact = {
        "path": logical_path,
        "format": result.get("ext"),
        "size_bytes": result.get("size_bytes"),
        "preview_type": result.get("preview_type"),
        "parse_status": result.get("parse_status"),
        "summary": result.get("summary"),
        "content_preview": result.get("content_preview"),
        "truncated": bool(result.get("truncated")),
    }
    if alias:
        compact["alias"] = alias
    type_specific = result.get("type_specific")
    if isinstance(type_specific, dict):
        structure = {
            key: type_specific[key]
            for key in COMPACT_STRUCTURE_KEYS
            if key in type_specific
        }
        if structure:
            compact["structure"] = structure
    if result.get("grep"):
        compact["grep"] = result["grep"]
    if result.get("warnings"):
        compact["warnings"] = result["warnings"]
    return compact


def worksheet_selection(args):
    """Read bounded exact cells without executing workbook or submitted code."""
    from openpyxl import load_workbook
    from openpyxl.utils.cell import range_boundaries, get_column_letter

    if args.resolved_path.suffix.lower() not in {'.xlsx', '.xlsm', '.xltx', '.xltm'}:
        raise ValueError('--sheet supports Excel XML workbooks only')
    wb = load_workbook(args.resolved_path, read_only=True, data_only=True, keep_links=False)
    try:
        if args.sheet not in wb.sheetnames:
            raise ValueError('unknown sheet; available names: ' + repr(wb.sheetnames))
        ws = wb[args.sheet]
        selected = args.cell_range or f'A1:{get_column_letter(min(ws.max_column or 1, 20))}{min(ws.max_row or 1, 25)}'
        c1, r1, c2, r2 = range_boundaries(selected)
        if (any(type(x) is not int or x < 1 for x in (c1, r1, c2, r2))
                or c2 < c1 or r2 < r1 or c2 > 16384 or r2 > 1048576
                or (c2-c1+1)*(r2-r1+1) > 10000):
            raise ValueError('use a finite A1 range of at most 10000 cells')
        lines = []
        for row_index, values in enumerate(ws.iter_rows(min_row=r1, max_row=r2,
                min_col=c1, max_col=c2, values_only=True), start=r1):
            lines.append(f'r{row_index}: ' + json.dumps(
                {f'{get_column_letter(c1+i)}{row_index}': cell_text(v) for i, v in enumerate(values)},
                ensure_ascii=False))
        preview, truncated = bounded_preview(lines, args.max_chars)
        return dict(path=args.logical_path, alias=args.alias, format=args.resolved_path.suffix.lstrip('.'),
            size_bytes=args.resolved_path.stat().st_size, parse_status='ok', preview_type='worksheet_range',
            sheet=args.sheet, selected_range=selected, content_preview=preview, truncated=truncated,
            selection_complete=not truncated, full_sheet_selected=(c1 == 1 and r1 == 1
                and c2 >= (ws.max_column or 1) and r2 >= (ws.max_row or 1) and not truncated),
            structure=dict(sheet_rows=ws.max_row, sheet_columns=ws.max_column, all_sheet_names=wb.sheetnames),
            value_mode='cached values; formulas not recalculated',
            next_action='Select a different bounded range for additional evidence; do not repeat unchanged requests.')
    finally:
        wb.close()


def main() -> int:
    try:
        args = parse_cli(sys.argv[1:])
    except (OSError, ValueError, json.JSONDecodeError, re.error) as exc:
        print(json.dumps({"parse_status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    if args.sheet:
        try:
            result = worksheet_selection(args)
        except Exception as exc:
            print(json.dumps({'parse_status': 'failed', 'error': str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=args.indent))
        return 0
    base_inspector = Path(__file__).with_name(BASE_INSPECTOR_NAME)
    forwarded_args = [
        args.logical_path,
        "--max-chars",
        str(args.max_chars),
        "--timeout",
        str(args.timeout),
        "--indent",
        "0",
    ]
    if args.cwd:
        forwarded_args.extend(("--cwd", args.cwd))
    completed = None
    for attempt in range(MAX_FILESYSTEM_ATTEMPTS):
        completed = subprocess.run(
            [sys.executable, str(base_inspector), *forwarded_args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        diagnostic = f"{completed.stderr}\n{completed.stdout}".lower()
        transient = any(marker in diagnostic for marker in TRANSIENT_FILESYSTEM_MARKERS)
        if not transient or attempt + 1 >= MAX_FILESYSTEM_ATTEMPTS:
            break
        time.sleep(0.25 * (attempt + 1))
    assert completed is not None
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        return completed.returncode
    try:
        result = json.loads(completed.stdout)
        extraction_limit = sys.maxsize if args.grep_pattern else args.max_chars
        result = enhance_result(result, args.resolved_path, extraction_limit)
        if args.grep_pattern:
            result = grep_result(result, args.grep_pattern, args.grep_context, args.max_chars)
        if os.environ.get("WSCLAW_INSPECT_COMPACT") == "1":
            result = compact_result(
                result,
                logical_path=args.logical_path,
                alias=args.alias,
            )
    except (json.JSONDecodeError, TypeError, ValueError, re.error):
        sys.stdout.write(completed.stdout)
        return 0
    print(json.dumps(result, ensure_ascii=False, indent=args.indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
