#!/usr/bin/env python3
"""Serve the SRE (1 GT anchor + 7 self-reward) pipeline to training queue requests.

Supports both DEAL-WS (with deterministic deliverable hardgate) and DEAL-RS.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sr_correctness.agent import AgentSpec
from sr_correctness.common import atomic_json, read_json, read_secret
from sr_correctness.group_reward import evaluate_group
from sr_correctness.streaming_group import StreamingGroupEvaluation


EXPECTED_GROUP_SIZE = 8
EXPECTED_MODE = "self_reward_7plus1_train7"


def _endpoint(config: dict[str, Any], model: str) -> dict[str, Any]:
    configured = config["models"][model]
    state = read_json(Path(configured["endpoint_file"]))
    expected_job = configured["job"]
    observed_job = state.get("job") or state.get("job_name")
    if observed_job != expected_job:
        raise ValueError(
            f"{model} endpoint belongs to {observed_job!r}, expected {expected_job!r}"
        )
    if state.get("model") != model or state.get("phase") != "ready":
        raise ValueError(f"{model} endpoint is not a ready matching init policy: {state}")
    url = str(state.get("url") or state.get("router") or "")
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"{model} endpoint has no HTTP(S) URL")
    return dict(state, job=observed_job, url=url)


def _spec(config: dict[str, Any], model: str, stage: str) -> AgentSpec:
    row = config[stage]
    configured = config["models"][model]
    endpoint = _endpoint(config, model)
    return AgentSpec(
        model=configured.get("api_model", model),
        base_url=endpoint["url"],
        api_key_file=Path(configured["api_key_file"]),
        temperature=float(row["temperature"]),
        timeout_sec=int(row["timeout_sec"]),
        max_tool_calls=int(row["max_tool_calls"]),
        max_tokens=int(row["max_tokens"]),
        context_window=int(row["context_window"]),
        enable_thinking=True,
        allow_retries=bool(row.get("allow_retries", False)),
    )


def _response_row(
    candidate: dict[str, Any], *, score: float = 0.0, valid: bool = False,
    trainable: bool = False, reason: str, **extra: Any,
) -> dict[str, Any]:
    return {
        "session_id": candidate["session_id"],
        "score": float(score),
        "valid": bool(valid),
        "trainable": bool(trainable),
        "reason": reason,
        **extra,
    }


class Service:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.queue = Path(config["queue_root"])
        self.queue.mkdir(parents=True, exist_ok=True)
        self.state = Path(config["state_root"])
        self.state.mkdir(parents=True, exist_ok=True)
        limit = int(config.get("agent_concurrency_per_model", 64))
        if limit < 1:
            raise ValueError("agent_concurrency_per_model must be positive")
        self.agent_concurrency_per_model = limit
        self.agent_slots = {
            model: threading.BoundedSemaphore(limit) for model in config["models"]
        }
        self.endpoint_state = {
            model: _endpoint(config, model) for model in config["models"]
        }
        self.stop = threading.Event()
        self.active_lock = threading.Lock()
        self.active: dict[str, dict[str, Any]] = {}

        self.dataset = str(config.get("dataset", "researchclaw"))
        if self.dataset not in {"wsclaw", "researchclaw"}:
            raise ValueError("dataset must be 'wsclaw' or 'researchclaw'")

        self.gt_scope = str(config.get("gt_scope", "anchor_only"))
        if self.gt_scope not in {"anchor_only", "all_candidates"}:
            raise ValueError("gt_scope must be 'anchor_only' or 'all_candidates'")

        self.ws_gt_worker_urls: list[str] | None = None
        self.ws_gt_api_key_file: Path | None = None
        if self.dataset == "wsclaw":
            urls = config.get("ws_gt_worker_urls") or config.get("ws_gt_urls")
            if not isinstance(urls, list) or not urls or any(
                not isinstance(u, str) or not u.startswith(("http://", "https://"))
                for u in urls
            ):
                raise ValueError("DEAL-WS reward service requires non-empty ws_gt_worker_urls")
            self.ws_gt_worker_urls = [u.rstrip("/") for u in urls]
            key_file = config.get("ws_gt_api_key_file")
            if key_file:
                self.ws_gt_api_key_file = Path(key_file)

        rubric_prompt = Path(config["rubric_prompt_path"])
        expected_prompt = (
            "rubric_agent_wsclaw.md"
            if self.dataset == "wsclaw"
            else "rubric_agent_researchclaw_relaxed.md"
        )
        if rubric_prompt.name != expected_prompt:
            raise ValueError(f"{self.dataset} requires rubric prompt {expected_prompt}")
        if not rubric_prompt.is_file():
            raise FileNotFoundError(f"rubric prompt missing: {rubric_prompt}")
        self.rubric_prompt = rubric_prompt

        score_tool_calls = int(config["score_agent"]["max_tool_calls"])
        if not 1 <= score_tool_calls <= 30:
            raise ValueError("score_agent.max_tool_calls must be between 1 and 30")

    @property
    def self_reward_repeats(self) -> int:
        repeats = int(self.config.get("self_reward_repeats", 3))
        if repeats not in (3, 5):
            raise ValueError("self_reward_repeats must be 3 or 5")
        return repeats

    @property
    def gt_all_candidates(self) -> bool:
        if "gt_scope" in self.config:
            return self.gt_scope == "all_candidates"
        return bool(self.config.get("gt_all_candidates_every_step", False))

    def _build_response_rows(
        self,
        candidates: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        gt = {int(row["candidate_id"]): row for row in result["ground_truth"]}
        comparison = {int(row["candidate_id"]): row for row in result["comparison_rows"]}
        rows: dict[str, dict[str, Any]] = {}
        anchor_gt = gt.get(0, {}).get("gt_reward")
        anchor_valid = candidates[0].get("valid") and anchor_gt is not None
        rows["0"] = _response_row(
            candidates[0],
            score=0.0,
            valid=bool(anchor_valid),
            trainable=False,
            reason="anchor_excluded" if anchor_valid else "anchor_or_gt_invalid",
            raw_reward=0.0,
            raw_gt_reward=float(anchor_gt) if anchor_gt is not None else None,
            raw_self_reward=None,
            accepted_repeats=0,
        )
        for candidate_id in range(1, EXPECTED_GROUP_SIZE):
            candidate = candidates[candidate_id]
            row = comparison.get(candidate_id) or {}
            reward = row.get("self_reward")
            eligible = bool(
                result.get("group_mask") == 1
                and candidate.get("valid")
                and reward is not None
            )
            raw_gt = gt.get(candidate_id, {}).get("gt_reward")
            rows[str(candidate_id)] = _response_row(
                candidate,
                score=float(reward or 0.0),
                valid=eligible,
                trainable=eligible,
                reason=(
                    "self_reward" if eligible else
                    "rollout_failed_or_timeout" if not candidate.get("valid") else
                    "invalid_rubric" if result.get("group_mask") != 1 else
                    "all_self_scores_failed"
                ),
                raw_reward=float(reward or 0.0),
                raw_gt_reward=float(raw_gt) if raw_gt is not None else None,
                raw_self_reward=float(reward) if reward is not None else None,
                self_scores=row.get("self_scores") or [],
                accepted_repeats=int(row.get("accepted_repeats") or 0),
                planned_repeats=self.self_reward_repeats,
                aggregation=row.get("aggregation") or "all_failed",
            )
        return rows

    def process(self, directory: Path) -> None:
        request = read_json(directory / "request.json")
        candidates = request["candidates"]
        model = str(request["model"])
        group_size = int(request.get("candidate_count", EXPECTED_GROUP_SIZE))
        if group_size != EXPECTED_GROUP_SIZE or len(candidates) != EXPECTED_GROUP_SIZE:
            raise ValueError(
                f"SRE requires 8 candidates (1 GT anchor + 7 SR), got {len(candidates)} vs {group_size}"
            )
        rollout_step = request.get("rollout_step")
        try:
            rollout_step = int(rollout_step) if rollout_step is not None else None
        except (TypeError, ValueError):
            rollout_step = None
        gt_all = self.gt_all_candidates
        attempt = 1
        while (directory / f"evaluation-attempt-{attempt}").exists():
            attempt += 1
        output = directory / f"evaluation-attempt-{attempt}"
        with self.active_lock:
            self.active[str(directory)] = {
                "model": model, "rollout_step": rollout_step, "started_at": time.time()
            }
        published: list[bool] = []

        def publish_training_result(result: dict[str, Any]) -> None:
            if published:
                return
            rows = self._build_response_rows(candidates, result)
            packet = {
                "group_id": request["group_id"],
                "model": model,
                "rollout_step": rollout_step,
                "gt_diagnostic": "all_candidates_every_step" if gt_all else "anchor_only",
                "mode": EXPECTED_MODE,
                "candidates": rows,
                "evaluation_dir": str(output),
                "metrics": result.get("metrics") or {},
                "scoring_failure_policy": result.get("scoring_failure_policy", "legacy"),
                "diagnostic_pending_candidate_ids": result.get("diagnostic_pending_candidate_ids", []),
                "at": time.time(),
            }
            atomic_json(directory / "response.json", packet)
            with (self.state / "completed.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(packet, ensure_ascii=False) + "\n")
            published.append(True)

        try:
            result = evaluate_group(
                project_root=Path(self.config["project_root"]),
                dataset=self.dataset,
                model=model,
                task_dir=Path(request["task_dir"]),
                candidates=candidates,
                output_dir=output,
                rubric_spec=_spec(self.config, model, "rubric_agent"),
                score_spec=_spec(self.config, model, "score_agent"),
                rubric_prompt_path=self.rubric_prompt,
                rubric_schema_path=ROOT / "schemas" / "rubric.schema.json",
                score_appendix_path=ROOT / "prompts" / "self_score_appendix.md",
                ws_gt_urls=self.ws_gt_worker_urls,
                ws_gt_api_key_file=self.ws_gt_api_key_file,
                ws_gt_workers_per_node=int(self.config.get("ws_gt_workers_per_node", 8)),
                gt_timeout_sec=int(self.config.get("gt_timeout_sec", 2700)),
                gt_concurrency=int(self.config.get("gt_concurrency_per_group", 1 if self.dataset == "wsclaw" else 4)),
                scoring_concurrency=7 * self.self_reward_repeats,
                self_reward_repeats=self.self_reward_repeats,
                gt_all_candidates=gt_all,
                training_ready_callback=(
                    publish_training_result if self.config.get("training_score_fail_fast") else None
                ),
                agent_semaphore=self.agent_slots[model],
            )
            publish_training_result(result)
        finally:
            with self.active_lock:
                self.active.pop(str(directory), None)

    def process_streaming(self, directory: Path) -> None:
        if self.config.get("training_score_fail_fast"):
            raise RuntimeError("single_attempt_mask_v1 requires batch request protocol")
        group = read_json(directory / "group.json")
        if group.get("protocol") != "self_reward_stream_v1":
            raise ValueError(f"unsupported streaming reward protocol: {group.get('protocol')!r}")
        model = str(group["profile"]["model"])
        group_size = int(group.get("candidate_count", EXPECTED_GROUP_SIZE))
        if group_size != EXPECTED_GROUP_SIZE:
            raise ValueError(
                f"SRE requires streaming group size 8 (1 GT anchor + 7 SR), got {group_size}"
            )
        rollout_step = group.get("rollout_step")
        attempt = 1
        while (directory / f"evaluation-attempt-{attempt}").exists():
            attempt += 1
        output = directory / f"evaluation-attempt-{attempt}"
        with self.active_lock:
            self.active[str(directory)] = {
                "model": model,
                "rollout_step": rollout_step,
                "protocol": group["protocol"],
                "started_at": time.time(),
            }
        try:
            evaluator = StreamingGroupEvaluation(
                project_root=Path(self.config["project_root"]),
                group_dir=directory,
                output_dir=output,
                group=group,
                rubric_spec=_spec(self.config, model, "rubric_agent"),
                score_spec=_spec(self.config, model, "score_agent"),
                rubric_prompt_path=self.rubric_prompt,
                rubric_schema_path=ROOT / "schemas" / "rubric.schema.json",
                score_appendix_path=ROOT / "prompts" / "self_score_appendix.md",
                self_reward_repeats=self.self_reward_repeats,
                gt_timeout_sec=int(self.config.get("gt_timeout_sec", 2700)),
                gt_concurrency=int(self.config.get("gt_concurrency_per_group", 1 if self.dataset == "wsclaw" else 4)),
                scoring_concurrency=7 * self.self_reward_repeats,
                agent_semaphore=self.agent_slots[model],
                ws_gt_urls=self.ws_gt_worker_urls,
                ws_gt_api_key_file=self.ws_gt_api_key_file,
                ws_gt_workers_per_node=int(self.config.get("ws_gt_workers_per_node", 8)),
                gt_all_candidates=self.gt_all_candidates,
            )
            result = evaluator.run()
            candidates = [
                read_json(directory / "candidates" / f"{i}.json")
                for i in range(EXPECTED_GROUP_SIZE)
            ]
            rows = self._build_response_rows(candidates, result)
            packet = {
                "group_id": group["group_id"],
                "model": model,
                "rollout_step": rollout_step,
                "gt_diagnostic": "all_candidates_every_step" if self.gt_all_candidates else "anchor_only",
                "mode": EXPECTED_MODE,
                "candidates": rows,
                "evaluation_dir": str(output),
                "metrics": result.get("metrics") or {},
                "stream_protocol": group["protocol"],
                "at": time.time(),
            }
            atomic_json(directory / "response.json", packet)
            with (self.state / "completed.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(packet, ensure_ascii=False) + "\n")
        finally:
            with self.active_lock:
                self.active.pop(str(directory), None)

    def failure(self, directory: Path, exc: BaseException) -> None:
        try:
            if (directory / "response.json").exists():
                atomic_json(directory / "post-response-diagnostic-error.json", {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "at": time.time(),
                    "training_response_preserved": True,
                })
                return
            if (directory / "request.json").exists():
                request = read_json(directory / "request.json")
                candidates = request["candidates"]
                group_id = request["group_id"]
            else:
                group = read_json(directory / "group.json")
                candidate_paths = [
                    directory / "candidates" / f"{index}.json"
                    for index in range(EXPECTED_GROUP_SIZE)
                ]
                if not (directory / "closed.json").exists() or not all(path.exists() for path in candidate_paths):
                    atomic_json(directory / "service-error.json", {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "at": time.time(),
                        "response_deferred_until_group_closed": True,
                    })
                    return
                candidates = [read_json(path) for path in candidate_paths]
                group_id = group["group_id"]
            rows = {
                str(index): _response_row(
                    candidate, reason="reward_service_error", valid=False, trainable=False
                )
                for index, candidate in enumerate(candidates)
            }
            atomic_json(directory / "service-error.json", {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "at": time.time(),
            })
            atomic_json(directory / "response.json", {
                "group_id": group_id,
                "candidates": rows,
                "error": type(exc).__name__,
                "at": time.time(),
            })
        finally:
            with self.active_lock:
                self.active.pop(str(directory), None)

    def run(self) -> None:
        workers = int(self.config.get("group_workers", 32))
        futures: dict[str, cf.Future[None]] = {}
        with cf.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="reward-group") as pool:
            while not self.stop.is_set():
                for group_path in self.queue.glob("*/group.json"):
                    directory = group_path.parent
                    key = str(directory)
                    if (
                        (directory / "response.json").exists()
                        or (directory / "abandoned.json").exists()
                        or key in futures
                    ):
                        continue
                    futures[key] = pool.submit(self.process_streaming, directory)
                request_paths = sorted(
                    self.queue.glob("*/request.json"),
                    key=lambda path: int(
                        read_json(path).get("scheduler", {}).get("group_id", 2**63)
                    ),
                )
                for request_path in request_paths:
                    directory = request_path.parent
                    key = str(directory)
                    if (directory / "response.json").exists() or key in futures:
                        continue
                    futures[key] = pool.submit(self.process, directory)
                for key, future in list(futures.items()):
                    if not future.done():
                        continue
                    try:
                        future.result()
                    except Exception as exc:
                        self.failure(Path(key), exc)
                    del futures[key]
                with self.active_lock:
                    active = dict(self.active)
                atomic_json(self.state / "heartbeat.json", {
                    "phase": "ready",
                    "at": time.time(),
                    "pid": os.getpid(),
                    "dataset": self.dataset,
                    "active_groups": active,
                    "queued_or_active": len(futures),
                    "group_workers": workers,
                    "agent_concurrency_per_model": self.agent_concurrency_per_model,
                    "self_reward_repeats": self.self_reward_repeats,
                    "candidate_count": EXPECTED_GROUP_SIZE,
                    "score_sessions_per_group": 7 * self.self_reward_repeats,
                    "gt_scope": self.gt_scope,
                    "stream_protocol": "self_reward_stream_v1",
                    "rubric_prompt_path": str(self.rubric_prompt),
                    "init_policy_endpoints": {
                        model: {
                            "job": state["job"],
                            "model": state["model"],
                            "url": state["url"],
                            "phase": state["phase"],
                        }
                        for model, state in self.endpoint_state.items()
                    },
                })
                time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = read_json(Path(args.config))
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(name, None)
    os.environ.update(NO_PROXY="*", no_proxy="*", PYTHONDONTWRITEBYTECODE="1")
    if config.get("training_score_fail_fast"):
        os.environ["CLAWGYM_TRAIN_SCORE_FAIL_FAST"] = "1"
        os.environ["RESEARCHCLAW_JUDGE_MAX_RETRIES"] = "0"
    if config.get("dataset", "researchclaw") == "researchclaw":
        gt = config["gt_judge"]
        os.environ.update({
            "RESEARCHCLAW_SCORE_MODE": "hybrid",
            "RESEARCHCLAW_LLMJUDGE_BACKEND": "hybrid_agentic",
            "RESEARCHCLAW_CODE_WEIGHT": "0.0",
            "RESEARCHCLAW_LLMJUDGE_WEIGHT": "1.0",
            "RESEARCHCLAW_JUDGE_BASE_URL": gt["base_url"],
            "RESEARCHCLAW_JUDGE_BASE_URL_OVERRIDE": gt["base_url"],
            "RESEARCHCLAW_JUDGE_MODEL": gt["model"],
            "RESEARCHCLAW_JUDGE_MODEL_OVERRIDE": gt["model"],
            "RESEARCHCLAW_JUDGE_API_KEY": read_secret(Path(gt["api_key_file"])),
            "RESEARCHCLAW_JUDGE_TEMPERATURE": "0.7",
            "RESEARCHCLAW_JUDGE_INCLUDE_GT_ANSWER": "1",
            "RESEARCHCLAW_DIRECT_JUDGE_ENABLE_THINKING": "0",
            "RESEARCHCLAW_AGENTIC_JUDGE_ENABLE_THINKING": "0",
            "NANOBOT_DISABLE_TOOL_LOOP_FINALIZATION": "0",
        })
    service = Service(config)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda _s, _f: service.stop.set())
    service.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
