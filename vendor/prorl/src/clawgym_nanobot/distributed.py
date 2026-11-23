from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
import urllib.parse
from concurrent.futures import as_completed
from typing import Any

import requests
from slime.utils.types import Sample

from . import generate as generate_mod
from . import tasks
from . import training_free
from . import user_agent


logger = logging.getLogger(__name__)

_DISTRIBUTED_ROLLOUT_ACTORS: list[dict[str, Any]] | None = None
_DISTRIBUTED_ROLLOUT_ACTORS_KEY: tuple[Any, ...] | None = None
_DISTRIBUTED_ROLLOUT_ACTORS_LOCK = threading.Lock()


def enabled() -> bool:
    return _env_truthy("CLAWGYM_DISTRIBUTED_ROLLOUT_NANOBOT", False)


def run_eval_jobs(
    args: Any,
    rollout_id: int,
    source: tasks.TaskSource,
    *,
    n_eval_samples: int,
    sampling_params: dict[str, Any],
    task_indices: list[int] | None = None,
    training_free_payload: training_free.BatchPayload | None = None,
) -> list[tuple[int, int, Any]]:
    import ray

    actors = _get_eval_actors(args)
    selected_task_indices = list(range(len(source.tasks))) if task_indices is None else list(task_indices)
    if selected_task_indices != sorted(set(selected_task_indices)):
        raise ValueError("distributed eval task_indices must be sorted and unique")
    if any(index < 0 or index >= len(source.tasks) for index in selected_task_indices):
        raise IndexError("distributed eval task index is out of range")
    eval_jobs = [
        (sample_idx * len(source.tasks) + task_index, task_index, task)
        for sample_idx in range(n_eval_samples)
        for task_index in selected_task_indices
        for task in (source.tasks[task_index],)
    ]
    assignments = _assign_eval_jobs_to_actors(eval_jobs, actors)
    logger.info(
        "[ClawGym-Nanobot] distributed eval dataset %s: %s; total=%d",
        source.root.name,
        ", ".join(
            f"{info['node_host']}={len(jobs)}" for info, jobs in zip(actors, assignments, strict=False)
        ),
        len(eval_jobs),
    )
    start_results = ray.get([info["actor"].start_eval_step.remote(rollout_id) for info in actors])
    for result in start_results:
        cleared = result.get("cleared") or {}
        if any(int(value or 0) for value in cleared.values()):
            logger.info(
                "[ClawGym-Nanobot] rollout-node %s cleared eval adapter state before rollout %s: %s",
                result.get("node_host"),
                rollout_id,
                cleared,
            )

    run_refs = [
        info["actor"].run_eval_jobs.remote(
            rollout_id, jobs, dict(sampling_params), training_free_payload
        )
        for info, jobs in zip(actors, assignments, strict=False)
    ]
    try:
        nested_results = ray.get(run_refs)
    finally:
        finish_results = ray.get([info["actor"].finish_eval_step.remote(rollout_id) for info in actors])
        for result in finish_results:
            cleared = result.get("cleared") or {}
            if any(int(value or 0) for value in cleared.values()):
                logger.info(
                    "[ClawGym-Nanobot] rollout-node %s cleared eval adapter state after rollout %s: %s",
                    result.get("node_host"),
                    rollout_id,
                    cleared,
                )
    results: list[tuple[int, int, Any]] = []
    deferred: dict[Any, tuple[dict[str, Any], Any]] = {}
    for node_results in nested_results:
        for item in node_results:
            flat_index = int(item["flat_index"])
            task_index = int(item["task_index"])
            if item.get("status") == "completed":
                results.append((flat_index, task_index, item["sample"]))
                continue
            if item.get("status") == "needs_user_agent":
                future = user_agent.get_scoring_pool().submit(item["user_agent_job"])
                deferred[future] = (item, item["sample"])
                continue
            logger.warning(
                "[ClawGym-Nanobot] distributed eval task %s skipped on %s (%s: %s)",
                source.tasks[task_index].task_id,
                item.get("node_host"),
                item.get("error_type"),
                item.get("error"),
            )
            results.append(
                (
                    flat_index,
                    task_index,
                    _aborted_eval_sample(
                        rollout_id=rollout_id,
                        flat_index=flat_index,
                        task_index=task_index,
                        task=source.tasks[task_index],
                        reason=f"{item.get('error_type')}: {item.get('error')}",
                    ),
                )
            )
    for future in as_completed(deferred):
        item, sample = deferred[future]
        ua_result = future.result()
        generate_mod.apply_user_agent_result(sample, ua_result)
        flat_index = int(item["flat_index"])
        task_index = int(item["task_index"])
        logger.info(
            "[ClawGym-Nanobot] rank0 eval UA complete flat=%d task=%s code_verifier=%.4f user_agent=%.4f",
            flat_index,
            source.tasks[task_index].task_id,
            float(item["user_agent_job"].code_verifier_score),
            float(ua_result.score),
        )
        results.append((flat_index, task_index, sample))
    return results


def run_auxiliary_chat_completion(
    args: Any,
    messages: list[dict[str, str]],
    sampling: dict[str, Any],
) -> str:
    """Run a non-trainable curator request on one existing rollout actor."""
    import ray

    actors = _get_eval_actors(args)
    if not actors:
        raise RuntimeError("no distributed rollout actor is available for training-free curator")
    return str(ray.get(actors[0]["actor"].auxiliary_chat_completion.remote(list(messages), dict(sampling))))


async def generate_train_sample(
    args: Any,
    base_sample: Any,
    sampling_params: dict[str, Any],
) -> Any:
    actors = _get_train_actors(args)
    actor_idx = _actor_index_for_sample(args, base_sample, len(actors))
    info = actors[actor_idx]
    group_id = _sample_group_id(args, base_sample)
    logger.debug(
        "[ClawGym-Nanobot] route train sample index=%s group=%s rollout_id=%s to rollout-node %s",
        getattr(base_sample, "index", None),
        group_id,
        getattr(base_sample, "rollout_id", None),
        info["node_host"],
    )
    result = await _ray_get_async(
        info["actor"].generate_train_sample.remote(base_sample, dict(sampling_params or {}))
    )
    if not isinstance(result, dict) or "user_agent_job" not in result:
        return result
    job = result["user_agent_job"]
    output = result["output"]
    ua_result = await asyncio.wrap_future(user_agent.get_scoring_pool().submit(job))
    generate_mod.apply_user_agent_result(output, ua_result)
    logger.info(
        "[ClawGym-Nanobot] rank0 train UA complete task=%s rollout-node=%s "
        "code_verifier=%.4f user_agent=%.4f",
        job.task_id,
        info["node_host"],
        float(job.code_verifier_score),
        float(ua_result.score),
    )
    return output


class _NanobotRolloutActor:
    def __init__(self, args: Any, node_host: str, worker_urls: list[str], max_workers: int):
        self.args = args
        self.node_host = node_host
        self.worker_urls = [_normalize_worker_url(url) for url in worker_urls if _normalize_worker_url(url)]
        self.max_workers = max(1, int(max_workers))
        os.environ["CLAWGYM_NANOBOT_DISTRIBUTED_ACTOR"] = "1"
        os.environ["CLAWGYM_NANOBOT_MAX_WORKERS"] = str(self.max_workers)
        os.environ["CLAWGYM_SGLANG_WORKER_URLS"] = ",".join(self.worker_urls)
        if _env_truthy("NANOBOT_DISTRIBUTED_LOCAL_WORKDIR", True):
            workdir_base = os.environ.get(
                "NANOBOT_ROLLOUT_WORKDIR_BASE",
                f"/tmp/{os.environ.get('USER', 'nanobot')}/clawgym-nanobot-rollout",
            )
            safe_node = self.node_host.replace(".", "-").replace(":", "-")
            os.environ["NANOBOT_WORKDIR"] = os.path.join(workdir_base, safe_node)
        # Ray creates this actor with ``max_concurrency``.  Without a preload,
        # the first group can enter ``load_tasks`` from several actor calls at
        # once and repeat the recursive AFS metadata scan.  Populate the
        # process-local LRU once, before concurrent rollout methods are
        # accepted; later training steps then reuse the same TaskSpec tuple.
        preload_started = time.monotonic()
        train_task_count = len(tasks.load_tasks())
        logger.info(
            "[ClawGym-Nanobot] rollout-node actor started: node=%s, max_workers=%d, "
            "sglang_workers=%d, train_tasks=%d, task_preload_sec=%.3f",
            self.node_host,
            self.max_workers,
            len(self.worker_urls),
            train_task_count,
            time.monotonic() - preload_started,
        )

    async def generate_train_sample(self, base_sample: Any, sampling_params: dict[str, Any]) -> Any:
        return await generate_mod.generate(
            self.args,
            base_sample,
            dict(sampling_params or {}),
            evaluation=False,
            _defer_user_agent=True,
        )

    def start_eval_step(self, rollout_id: int) -> dict[str, Any]:
        cleared = generate_mod.reset_adapter_state()
        return {
            "node_host": self.node_host,
            "rollout_id": rollout_id,
            "worker_urls": list(self.worker_urls),
            "cleared": cleared,
        }

    def finish_eval_step(self, rollout_id: int) -> dict[str, Any]:
        cleared = generate_mod.reset_adapter_state()
        return {
            "node_host": self.node_host,
            "rollout_id": rollout_id,
            "cleared": cleared,
        }

    async def run_eval_jobs(
        self,
        rollout_id: int,
        jobs: list[tuple[int, int, tasks.TaskSpec]],
        sampling_params: dict[str, Any],
        training_free_payload: training_free.BatchPayload | None = None,
    ) -> list[dict[str, Any]]:
        if not jobs:
            return []
        return await self._run_eval_jobs_async(
            rollout_id, jobs, sampling_params, training_free_payload
        )

    async def auxiliary_chat_completion(
        self,
        messages: list[dict[str, str]],
        sampling: dict[str, Any],
    ) -> str:
        return await generate_mod.auxiliary_chat_completion(self.args, messages, sampling)

    async def _run_eval_jobs_async(
        self,
        rollout_id: int,
        jobs: list[tuple[int, int, tasks.TaskSpec]],
        sampling_params: dict[str, Any],
        training_free_payload: training_free.BatchPayload | None,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, min(self.max_workers, len(jobs))))
        coroutines = [
            self._run_one(
                rollout_id=rollout_id,
                flat_index=flat_index,
                task_index=task_index,
                task=task,
                sampling_params=dict(sampling_params),
                semaphore=semaphore,
                training_free_payload=training_free_payload,
            )
            for flat_index, task_index, task in jobs
        ]
        results: list[dict[str, Any]] = []
        for future in asyncio.as_completed(coroutines):
            item = await future
            results.append(item)
            if item.get("status") == "completed":
                reward = _sample_reward(item["sample"])
                logger.info(
                    "[ClawGym-Nanobot] rollout-node %s eval progress %d/%d flat=%d task=%s: reward=%.4f",
                    self.node_host,
                    len(results),
                    len(jobs),
                    int(item["flat_index"]),
                    item.get("task_id"),
                    reward,
                )
            elif item.get("status") == "needs_user_agent":
                logger.info(
                    "[ClawGym-Nanobot] rollout-node %s eval progress %d/%d flat=%d task=%s: "
                    "code_verifier=%.4f queued_user_agent=1",
                    self.node_host,
                    len(results),
                    len(jobs),
                    int(item["flat_index"]),
                    item.get("task_id"),
                    float(item["user_agent_job"].code_verifier_score),
                )
            else:
                logger.warning(
                    "[ClawGym-Nanobot] rollout-node %s eval progress %d/%d flat=%d: SKIPPED (%s: %s)",
                    self.node_host,
                    len(results),
                    len(jobs),
                    int(item["flat_index"]),
                    item.get("error_type"),
                    item.get("error"),
                )
        return results

    async def _run_one(
        self,
        *,
        rollout_id: int,
        flat_index: int,
        task_index: int,
        task: tasks.TaskSpec,
        sampling_params: dict[str, Any],
        semaphore: asyncio.Semaphore,
        training_free_payload: training_free.BatchPayload | None,
    ) -> dict[str, Any]:
        async with semaphore:
            sample = Sample(
                index=flat_index,
                group_index=task_index,
                prompt=task.user_query,
                metadata={
                    "task_id": task.task_id,
                    "task_index": task_index,
                    "task_dir": str(task.task_dir),
                },
            )
            sample.rollout_id = rollout_id
            if training_free_payload is not None:
                training_free.attach_sample_context(sample, training_free_payload)
                sample.clawgym_task_override = task
            try:
                result = await generate_mod.generate(
                    self.args,
                    sample,
                    dict(sampling_params),
                    evaluation=True,
                    _defer_user_agent=True,
                )
                if isinstance(result, dict) and "user_agent_job" in result:
                    output = result["output"]
                    if isinstance(output, list):
                        output = output[-1] if output else _aborted_eval_sample(
                            rollout_id=rollout_id,
                            flat_index=flat_index,
                            task_index=task_index,
                            task=task,
                            reason="empty deferred eval result",
                        )
                    return {
                        "node_host": self.node_host,
                        "flat_index": flat_index,
                        "task_index": task_index,
                        "task_id": task.task_id,
                        "status": "needs_user_agent",
                        "sample": output,
                        "user_agent_job": result["user_agent_job"],
                    }
                if isinstance(result, list):
                    result = result[-1] if result else _aborted_eval_sample(
                        rollout_id=rollout_id,
                        flat_index=flat_index,
                        task_index=task_index,
                        task=task,
                        reason="empty eval result",
                    )
                return {
                    "node_host": self.node_host,
                    "flat_index": flat_index,
                    "task_index": task_index,
                    "task_id": task.task_id,
                    "status": "completed",
                    "sample": result,
                }
            except Exception as exc:
                logger.exception("[ClawGym-Nanobot] rollout-node %s eval task %s failed", self.node_host, task.task_id)
                return {
                    "node_host": self.node_host,
                    "flat_index": flat_index,
                    "task_index": task_index,
                    "task_id": task.task_id,
                    "status": "skipped",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }


def _get_eval_actors(args: Any) -> list[dict[str, Any]]:
    actors, key = _get_rollout_actors(args)
    logger.info(
        "[ClawGym-Nanobot] distributed eval actors ready: %s; max_workers_per_node=%d",
        ", ".join(f"{info['node_host']}({len(info['worker_urls'])} engines)" for info in actors),
        int(key[-1]),
    )
    return actors


def _get_train_actors(args: Any) -> list[dict[str, Any]]:
    actors, key = _get_rollout_actors(args)
    logger.info(
        "[ClawGym-Nanobot] distributed train actors ready: %s; max_workers_per_node=%d; group_affinity=%s",
        ", ".join(f"{info['node_host']}({len(info['worker_urls'])} engines)" for info in actors),
        int(key[-1]),
        int(_env_truthy("NANOBOT_ROLLOUT_GROUP_AFFINITY", True)),
    )
    return actors


def _get_rollout_actors(args: Any) -> tuple[list[dict[str, Any]], tuple[Any, ...]]:
    global _DISTRIBUTED_ROLLOUT_ACTORS, _DISTRIBUTED_ROLLOUT_ACTORS_KEY
    specs, key = _rollout_actor_specs(args)
    with _DISTRIBUTED_ROLLOUT_ACTORS_LOCK:
        if _DISTRIBUTED_ROLLOUT_ACTORS is not None and _DISTRIBUTED_ROLLOUT_ACTORS_KEY == key:
            return _DISTRIBUTED_ROLLOUT_ACTORS, key

        if _DISTRIBUTED_ROLLOUT_ACTORS:
            for info in _DISTRIBUTED_ROLLOUT_ACTORS:
                try:
                    _kill_actor(info["actor"])
                except Exception:
                    pass

        actors = _create_rollout_actors(args, specs, max_workers=int(key[-1]))
        _DISTRIBUTED_ROLLOUT_ACTORS = actors
        _DISTRIBUTED_ROLLOUT_ACTORS_KEY = key
        logger.info(
            "[ClawGym-Nanobot] distributed rollout actors: %s; max_workers_per_node=%d",
            ", ".join(f"{info['node_host']}({len(info['worker_urls'])} engines)" for info in actors),
            int(key[-1]),
        )
        return actors, key


def _rollout_actor_specs(args: Any) -> tuple[list[tuple[str, list[str]]], tuple[Any, ...]]:
    worker_urls = _fetch_sglang_worker_urls(args)
    grouped = _group_worker_urls_by_node(worker_urls)
    if not grouped:
        raise RuntimeError("distributed nanobot rollout enabled but no SGLang worker URLs were grouped by node")

    max_workers = _positive_int_env("NANOBOT_ROLLOUT_MAX_WORKERS_PER_NODE", 64)
    specs = [(host, urls) for host, urls in sorted(grouped.items())]
    key = (tuple((host, tuple(urls)) for host, urls in specs), max_workers)
    return specs, key


def _create_rollout_actors(
    args: Any,
    specs: list[tuple[str, list[str]]],
    *,
    max_workers: int,
) -> list[dict[str, Any]]:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    actor_cls = ray.remote(num_cpus=1, num_gpus=0, max_concurrency=max_workers)(_NanobotRolloutActor)
    actors: list[dict[str, Any]] = []
    for node_host, urls in specs:
        node_id = _ray_node_id_for_host(node_host)
        if not node_id:
            raise RuntimeError(f"Ray node for rollout SGLang host {node_host} not found")
        actor = actor_cls.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        ).remote(args, node_host, urls, max_workers)
        actors.append({"node_host": node_host, "worker_urls": urls, "actor": actor})
    return actors


def _kill_actor(actor: Any) -> None:
    import ray

    ray.kill(actor, no_restart=True)


def _fetch_sglang_worker_urls(args: Any) -> list[str]:
    router_base = f"http://{getattr(args, 'sglang_router_ip')}:{getattr(args, 'sglang_router_port')}"
    deadline = time.monotonic() + _positive_int_env("NANOBOT_ROLLOUT_WORKER_DISCOVERY_TIMEOUT", 180)
    last_error = ""
    while time.monotonic() < deadline:
        for path in ("/workers", "/list_workers"):
            url = f"{router_base}{path}"
            try:
                response = requests.get(url, timeout=5)
                if response.status_code != 200:
                    last_error = f"{url} status={response.status_code}"
                    continue
                payload = response.json()
                urls: list[str] = []
                if isinstance(payload, dict) and isinstance(payload.get("workers"), list):
                    for item in payload["workers"]:
                        if isinstance(item, dict):
                            urls.append(_normalize_worker_url(str(item.get("url") or "")))
                        else:
                            urls.append(_normalize_worker_url(str(item)))
                elif isinstance(payload, dict) and isinstance(payload.get("urls"), list):
                    urls = [_normalize_worker_url(str(item)) for item in payload["urls"]]
                urls = [item for item in urls if item]
                if urls:
                    return sorted(set(urls))
                last_error = f"{url} returned no workers"
            except Exception as exc:
                last_error = f"{url}: {type(exc).__name__}: {exc}"
        time.sleep(2)
    raise RuntimeError(f"could not discover SGLang worker URLs from {router_base}: {last_error}")


def _group_worker_urls_by_node(worker_urls: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for url in worker_urls:
        normalized = _normalize_worker_url(url)
        host = _host_from_worker_url(normalized)
        if host:
            grouped.setdefault(host, []).append(normalized)
    return {host: sorted(set(urls)) for host, urls in grouped.items() if urls}


def _assign_eval_jobs_to_actors(
    eval_jobs: list[tuple[int, int, tasks.TaskSpec]],
    actors: list[dict[str, Any]],
) -> list[list[tuple[int, int, tasks.TaskSpec]]]:
    assignments: list[list[tuple[int, int, tasks.TaskSpec]]] = [[] for _ in actors]
    if not actors:
        return assignments
    if _env_truthy("NANOBOT_ROLLOUT_GROUP_AFFINITY", True):
        grouped: dict[int, list[tuple[int, int, tasks.TaskSpec]]] = {}
        for flat_index, task_index, task in eval_jobs:
            grouped.setdefault(int(task_index), []).append((flat_index, int(task_index), task))
        loads = [0 for _ in actors]
        for task_index in sorted(grouped):
            actor_idx = min(range(len(actors)), key=lambda idx: (loads[idx], idx))
            jobs = grouped[task_index]
            assignments[actor_idx].extend(jobs)
            loads[actor_idx] += len(jobs)
        return assignments

    for job_index, job in enumerate(eval_jobs):
        assignments[job_index % len(actors)].append(job)
    return assignments


async def _ray_get_async(object_ref: Any) -> Any:
    import ray

    return await asyncio.to_thread(ray.get, object_ref)


def _actor_index_for_sample(args: Any, sample: Any, actor_count: int) -> int:
    if actor_count <= 0:
        raise RuntimeError("distributed nanobot rollout enabled but no rollout-node actors are available")
    if _env_truthy("NANOBOT_ROLLOUT_GROUP_AFFINITY", True):
        return _sample_group_id(args, sample) % int(actor_count)
    sample_index = _int_or_none(getattr(sample, "index", None))
    if sample_index is None:
        sample_index = _int_or_none(getattr(sample, "rollout_id", None))
    return int(sample_index or 0) % int(actor_count)


def _sample_group_id(args: Any, sample: Any) -> int:
    group_index = _int_or_none(getattr(sample, "group_index", None))
    if group_index is not None:
        return int(group_index)

    sample_index = _int_or_none(getattr(sample, "index", None))
    if sample_index is not None:
        n_samples_per_prompt = _positive_int_attr(args, "n_samples_per_prompt", 1)
        return int(sample_index) // n_samples_per_prompt

    rollout_id = _int_or_none(getattr(sample, "rollout_id", None))
    return int(rollout_id or 0)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int_attr(args: Any, name: str, default: int) -> int:
    value = _int_or_none(getattr(args, name, None))
    return value if value is not None and value > 0 else int(default)


def _ray_node_id_for_host(host: str) -> str | None:
    import ray

    wanted = str(host or "").strip()
    if not wanted:
        return None
    wanted_candidates = {wanted}
    try:
        wanted_candidates.add(socket.gethostbyname(wanted))
    except OSError:
        pass
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        node_address = str(node.get("NodeManagerAddress") or "")
        node_hostname = str(node.get("NodeManagerHostname") or "")
        candidates = {node_address, node_hostname}
        try:
            candidates.add(socket.gethostbyname(node_hostname))
        except OSError:
            pass
        resources = node.get("Resources") or {}
        candidates.update(key[len("node:") :] for key in resources if isinstance(key, str) and key.startswith("node:"))
        if wanted_candidates & candidates:
            return node.get("NodeID")
    return None


def _normalize_worker_url(url: str) -> str:
    stripped = str(url or "").strip().rstrip("/")
    if not stripped:
        return ""
    if "://" not in stripped:
        stripped = f"http://{stripped}"
    return stripped


def _host_from_worker_url(url: str) -> str:
    parsed = urllib.parse.urlparse(_normalize_worker_url(url))
    return parsed.hostname or ""


def _aborted_eval_sample(
    *,
    rollout_id: int,
    flat_index: int,
    task_index: int,
    task: tasks.TaskSpec,
    reason: str,
) -> Sample:
    sample = Sample(
        index=flat_index,
        group_index=task_index,
        prompt=task.user_query,
        tokens=[0, 0],
        response="",
        response_length=1,
        loss_mask=[0],
        rollout_log_probs=[0.0],
        reward={"score": 0.0},
        remove_sample=True,
        metadata={
            "truncated": False,
            "finish_reason": "abort",
            "truncation_reason": None,
            "clawgym_nanobot": {
                "status": "aborted",
                "reason": reason,
                "task_id": task.task_id,
                "task_index": task_index,
            }
        },
    )
    sample.rollout_id = rollout_id
    sample.status = Sample.Status.ABORTED
    return sample


def _sample_reward(sample: Any) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if "score" in reward:
            return float(reward["score"])
        if reward:
            return float(next(iter(reward.values())))
        return 0.0
    if reward is None:
        return 0.0
    return float(reward)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(default)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}
