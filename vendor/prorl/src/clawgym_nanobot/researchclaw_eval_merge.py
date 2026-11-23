from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Candidate:
    task_id: str
    status: str
    score: float | None
    run_root: Path
    reward_path: Path | None
    run_status_path: Path | None
    updated_ns: int

    @property
    def successful(self) -> bool:
        return self.status == "completed" and self.score is not None


def merge_eval_runs(
    expected_tasks_root: str | Path,
    run_roots: Iterable[str | Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_root = Path(expected_tasks_root).resolve()
    expected = _expected_tasks(expected_root)
    candidates: dict[str, list[Candidate]] = {task_id: [] for task_id in expected}
    for raw_root in run_roots:
        root = Path(raw_root).resolve()
        for candidate in _run_candidates(root):
            if candidate.task_id in candidates:
                candidates[candidate.task_id].append(candidate)

    rows: list[dict[str, Any]] = []
    successful_scores: list[float] = []
    trajectory_complete_count = 0
    for task_id, task_dir in expected.items():
        selected = _select_candidate(candidates[task_id])
        if selected is None:
            rows.append(
                {
                    "task_id": task_id,
                    "task_dir": str(task_dir),
                    "status": "missing",
                    "score": 0.0,
                    "score_is_missing_zero": True,
                    "trajectory_complete": False,
                }
            )
            continue

        trajectory_complete = _trajectory_complete(selected)
        trajectory_complete_count += int(trajectory_complete)
        row = {
            "task_id": task_id,
            "task_dir": str(task_dir),
            "status": "completed" if selected.successful else "failed",
            "score": selected.score if selected.score is not None else 0.0,
            "score_is_missing_zero": not selected.successful,
            "source_run": str(selected.run_root),
            "reward_path": str(selected.reward_path) if selected.reward_path else None,
            "run_status_path": str(selected.run_status_path) if selected.run_status_path else None,
            "trajectory_complete": trajectory_complete,
        }
        rows.append(row)
        if selected.successful:
            successful_scores.append(float(selected.score))

    success_count = len(successful_scores)
    failed_count = sum(row["status"] == "failed" for row in rows)
    missing_count = sum(row["status"] == "missing" for row in rows)
    expected_count = len(rows)
    summary = {
        "expected_task_count": expected_count,
        "unique_terminal_task_count": success_count + failed_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "missing_count": missing_count,
        "missing_as_zero_mean": sum(successful_scores) / expected_count if expected_count else 0.0,
        "llmjudge_mean_on_success": (
            sum(successful_scores) / success_count if success_count else 0.0
        ),
        "trajectory_complete_count": trajectory_complete_count,
        "trajectory_complete": trajectory_complete_count == expected_count,
    }
    return rows, summary


def write_merge_outputs(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    write_retry_view: bool = False,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "merged-results.json", rows)
    _atomic_json(root / "summary.json", summary)
    if write_retry_view:
        _write_retry_view(root / "retry-view", rows)


def _expected_tasks(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for entry in sorted(root.iterdir()):
        metadata = entry / "data_entry.json"
        if not metadata.is_file():
            continue
        payload = _read_dict(metadata)
        task_id = str(payload.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"missing task_id in {metadata}")
        if task_id in result:
            raise ValueError(f"duplicate task_id {task_id!r} under {root}")
        result[task_id] = entry.resolve()
    return dict(sorted(result.items()))


def _run_candidates(run_root: Path) -> list[Candidate]:
    eval_root = run_root / "eval"
    statuses = _status_paths(eval_root)
    status_by_task: dict[str, list[Path]] = {}
    for path in statuses:
        task_id = str(_read_dict(path).get("task_id") or "").strip()
        if task_id:
            status_by_task.setdefault(task_id, []).append(path)

    result: list[Candidate] = []
    reward_paths = sorted((eval_root / "nanobot_work").glob("**/researchclaw_reward.json"))
    for path in reward_paths:
        payload = _read_dict(path)
        task_id = str(payload.get("task_id") or "").strip()
        score = _pure_llmjudge_score(payload)
        if not task_id:
            continue
        status_path = _newest(status_by_task.get(task_id, []))
        result.append(
            Candidate(
                task_id=task_id,
                status="completed" if score is not None else "failed",
                score=score,
                run_root=run_root,
                reward_path=path.resolve(),
                run_status_path=status_path.resolve() if status_path else None,
                updated_ns=max(path.stat().st_mtime_ns, _mtime_ns(status_path)),
            )
        )

    tasks_with_reward = {candidate.task_id for candidate in result}
    for task_id, paths in status_by_task.items():
        if task_id in tasks_with_reward:
            continue
        path = _newest(paths)
        if path is None:
            continue
        payload = _read_dict(path)
        result.append(
            Candidate(
                task_id=task_id,
                status=str(payload.get("status") or "failed"),
                score=None,
                run_root=run_root,
                reward_path=None,
                run_status_path=path.resolve(),
                updated_ns=path.stat().st_mtime_ns,
            )
        )
    return result


def _status_paths(eval_root: Path) -> list[Path]:
    root = eval_root / "nanobot_logs"
    return sorted(root.glob("**/run-status.json")) if root.is_dir() else []


def _pure_llmjudge_score(payload: dict[str, Any]) -> float | None:
    if str(payload.get("requested_score_mode") or "").strip().lower() != "hybrid":
        return None
    weights = payload.get("requested_score_weights")
    if isinstance(weights, dict):
        code_weight = _finite(weights.get("code"))
        llm_weight = _finite(weights.get("llmjudge"))
        if code_weight not in (None, 0.0) or llm_weight not in (None, 1.0):
            return None
    final_score = _finite(payload.get("final_score"))
    llm_score = _finite(payload.get("llmjudge_score"))
    if final_score is None or llm_score is None or not math.isclose(final_score, llm_score, abs_tol=1e-12):
        return None
    return final_score


def _select_candidate(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    successful = [candidate for candidate in candidates if candidate.successful]
    return max(successful or candidates, key=lambda candidate: candidate.updated_ns)


def _trajectory_complete(candidate: Candidate) -> bool:
    if not candidate.successful or candidate.run_status_path is None:
        return False
    status = _read_dict(candidate.run_status_path)
    return (
        str(status.get("task_id") or "") == candidate.task_id
        and str(status.get("status") or "") == "completed"
        and int(status.get("turn_count") or 0) >= 1
    )


def _write_retry_view(root: Path, rows: list[dict[str, Any]]) -> None:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"retry view is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row["status"] == "completed":
            continue
        target = Path(str(row["task_dir"]))
        link = root / target.name
        link.symlink_to(os.path.relpath(target, start=root), target_is_directory=True)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _read_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _newest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None


def _mtime_ns(path: Path | None) -> int:
    return path.stat().st_mtime_ns if path is not None else 0


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
