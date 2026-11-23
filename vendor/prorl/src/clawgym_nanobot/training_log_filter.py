"""Bound noisy distributed-training console logs without hiding failures."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator

DEFAULT_MAX_LINE_CHARS = 16_384

_DROP_SUBSTRINGS = (
    "Decode batch,",
    "Prefill batch,",
    '"POST /v1/chat/completions HTTP/1.1" 200 OK',
    '"GET /health HTTP/1.1" 200 OK',
)
_SGLANG_IMPORTANT_MARKERS = (
    " error",
    "error:",
    " exception",
    "exception:",
    " traceback",
    "warning",
    "critical",
    "fatal",
)
_PAYLOAD_OMISSION_MARKERS = (
    "Failed to parse JSON part:",
    "Failed to parse tool call JSON:",
)
_JSON_PARSE_ERROR_MARKER = "JSON parse error:"


def _max_line_chars_from_env() -> int:
    raw = os.environ.get("CLAWGYM_TRAIN_LOG_MAX_LINE_CHARS", str(DEFAULT_MAX_LINE_CHARS))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_LINE_CHARS
    return value if value >= 512 else DEFAULT_MAX_LINE_CHARS


def filter_line(line: str, *, max_line_chars: int = DEFAULT_MAX_LINE_CHARS) -> str | None:
    """Return a bounded useful line, or ``None`` for routine SGLang noise."""

    lowered = line.lower()
    if any(marker in line for marker in _DROP_SUBSTRINGS):
        return None
    if "sglangengine" in lowered and not any(marker in lowered for marker in _SGLANG_IMPORTANT_MARKERS):
        return None

    # Some SGLang tool-parser failures put the complete (occasionally
    # multi-kilobyte) model payload before this suffix. Preserve the useful
    # parser error while omitting generated content from the console log.
    parse_error_index = line.find(_JSON_PARSE_ERROR_MARKER)
    if "sglangengine" in lowered and parse_error_index >= 0:
        engine_prefix_end = line.find(")")
        engine_prefix = line[: engine_prefix_end + 1] if engine_prefix_end >= 0 else "SGLangEngine"
        newline = "\n" if line.endswith("\n") else ""
        line = (
            f"{engine_prefix} {_JSON_PARSE_ERROR_MARKER} "
            f"{line[parse_error_index + len(_JSON_PARSE_ERROR_MARKER):].strip()}"
            f"{newline}"
        )

    for marker in _PAYLOAD_OMISSION_MARKERS:
        marker_index = line.find(marker)
        if marker_index >= 0:
            prefix = line[: marker_index + len(marker)]
            newline = "\n" if line.endswith("\n") else ""
            line = f"{prefix} <payload omitted>{newline}"
            break

    newline = "\n" if line.endswith("\n") else ""
    content = line[:-1] if newline else line
    if len(content) > max_line_chars:
        omitted = len(content) - max_line_chars
        content = f"{content[:max_line_chars]} ... <truncated {omitted} chars>"
    return content + newline


def filter_lines(lines: Iterable[str], *, max_line_chars: int = DEFAULT_MAX_LINE_CHARS) -> Iterator[str]:
    for line in lines:
        filtered = filter_line(line, max_line_chars=max_line_chars)
        if filtered is not None:
            yield filtered


def main() -> int:
    max_line_chars = _max_line_chars_from_env()
    for line in filter_lines(sys.stdin, max_line_chars=max_line_chars):
        sys.stdout.write(line)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
