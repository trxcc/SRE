"""Authenticate deliberate eight-candidate SR masks without changing any loss mask."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

KEY = "researchclaw_intentional_terminal_mask"
RECEIPT = "researchclaw_terminal_mask_receipt"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _group_size():
    from clawgym_nanobot.self_reward import configuration, configured_group_size
    return configured_group_size(configuration())


def _zero(mask):
    if mask is None:
        return False
    return all(float(v) == 0.0 for v in mask)


def _authority():
    value = os.environ.get("CLAWGYM_SKIP_INTENTIONAL_ZERO_BATCHES", "0")
    _require(value in ("0", "1"), "invalid terminal skip opt-in")
    if value == "0":
        return None
    from clawgym_nanobot.self_reward import configuration, configured_group_size
    cfg = configuration()
    _require(isinstance(cfg, dict) and cfg.get("profile", {}).get("dataset") == "researchclaw", "terminal skip requires ResearchClaw profile")
    _require(configured_group_size(cfg) in (4, 8), "terminal skip requires four- or eight-candidate SR")
    filename = os.environ.get("CLAWGYM_INTENTIONAL_ZERO_AUTHORITY")
    expected = os.environ.get("CLAWGYM_INTENTIONAL_ZERO_AUTHORITY_SHA")
    _require(bool(filename and expected), "terminal skip requires sealed run authority")
    _require(hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, "terminal skip authority mismatch")
    return expected


def mark_terminal_group(task_result, samples, group_id):
    authority = _authority()
    if authority is None:
        return
    results = task_result.results
    # Finalization alone is not evidence of an ordinary scoring outcome:
    # failed rollout capture and a missing reward-service delivery are infra.
    if any(
        str((r.metadata or {}).get("polar_failure_stage") or "").lower() == "infrastructure"
        or str((r.metadata or {}).get("self_reward_reason") or "").lower()
        in {"reward_service_timeout", "rollout_failed_or_timeout"}
        for r in results
    ):
        return
    if not (len(results) == _group_size() and all(isinstance(r.metadata, dict) and r.metadata.get("self_reward_finalized") is True and r.metadata.get("self_reward_trainable") is False for r in results)):
        return
    sessions = {str(r.session_id): i for i, r in enumerate(results)}
    _require(len(sessions) == _group_size(), "terminal group session identities are not unique")
    seen = set()
    for sample in samples:
        polar = (getattr(sample, "metadata", None) or {}).get("polar", {})
        sid = str(polar.get("session_id") or "")
        _require(sid in sessions and _zero(sample.loss_mask), "terminal group contains unverified session or trainable token")
        candidate = sessions[sid]
        _require(int(sample.index) == _group_size() * group_id + candidate and int(sample.group_index) == group_id, "terminal group sample identity mismatch")
        sample.metadata[KEY] = {"schema": 1, "authority_SHA": authority, "group_id": group_id, "candidate": candidate, "task_id": str(task_result.task_id), "session_id": sid, "reason": "native_finalized_SR_group_has_no_trainable_candidates"}
        seen.add(candidate)
    _require(seen == set(range(_group_size())), "terminal group is missing a logical candidate")


def build_receipt(args, samples, rollout_id):
    authority = _authority()
    if authority is None:
        return None
    _require(not getattr(args, "use_critic", False), "terminal skip is not defined for critic training")
    _require(args.n_samples_per_prompt == _group_size() and args.rollout_batch_size == 16 and (args.num_rollout == 113 or (getattr(args, "start_rollout_id", None) == 21 and args.num_rollout == 41)), "terminal skip requires sealed finite1800 schedule")
    if not samples or any(not _zero(s.loss_mask) for s in samples):
        return None
    markers = [(getattr(s, "metadata", None) or {}).get(KEY) for s in samples]
    if any(m is None for m in markers):
        return None
    begin, end = rollout_id * 16, min(1800, (rollout_id + 1) * 16)
    _require(begin < end, "terminal receipt rollout outside finite pass")
    seen, group_tasks, group_sessions, indices = {}, {}, {}, []
    for sample, marker in zip(samples, markers):
        _require(isinstance(marker, dict) and marker.get("schema") == 1 and marker.get("authority_SHA") == authority and marker.get("reason") == "native_finalized_SR_group_has_no_trainable_candidates", "unverified terminal marker")
        gid, candidate = marker["group_id"], marker["candidate"]
        _require(type(gid) is int and type(candidate) is int and begin <= gid < end and 0 <= candidate < _group_size(), "terminal marker outside current batch")
        _require(int(sample.index) == _group_size() * gid + candidate and int(sample.group_index) == gid, "terminal marker sample identity changed")
        polar = sample.metadata.get("polar", {})
        _require(polar.get("accepted_rollout_id") == rollout_id and polar.get("scheduler_group_id") == gid and polar.get("session_id") == marker["session_id"], "terminal marker delivery identity changed")
        seen.setdefault(gid, set()).add(candidate)
        group_tasks.setdefault(gid, set()).add(marker["task_id"])
        group_sessions.setdefault(gid, {}).setdefault(candidate, set()).add(marker["session_id"])
        indices.append(int(sample.index))
    _require(set(seen) == set(range(begin, end)) and all(values == set(range(_group_size())) for values in seen.values()), "terminal batch logical coverage incomplete")
    _require(all(len(ids) == 1 for ids in group_tasks.values()) and all(len(ids) == 1 for per in group_sessions.values() for ids in per.values()), "mixed terminal task or session identities")
    _require(all(len({next(iter(ids)) for ids in per.values()}) == _group_size() for per in group_sessions.values()), "duplicate terminal session identity")
    return {"schema": 1, "authority_SHA": authority, "rollout_id": rollout_id, "group_ids": list(range(begin, end)), "logical_candidates": (end - begin) * _group_size(), "native_samples": len(samples), "sample_indices_SHA": _digest(indices), "task_sessions_SHA": _digest({str(g): {str(c): next(iter(ids)) for c, ids in per.items()} for g, per in group_sessions.items()}), "trainable_tokens": 0, "reason": "native_finalized_SR_group_has_no_trainable_candidates"}


def validate_transport(args, shards):
    authority = _authority()
    if authority is None:
        return None
    _require(not getattr(args, "use_critic", False), "terminal skip is not defined for critic training")
    receipts = [s.get(RECEIPT) for s in shards]
    if not any(r is not None for r in receipts):
        return None
    receipt = receipts[0]
    _require(isinstance(receipt, dict) and all(r == receipt for r in receipts), "terminal receipts differ across DP shards")
    _require(receipt.get("schema") == 1 and receipt.get("authority_SHA") == authority and receipt.get("trainable_tokens") == 0 and receipt.get("reason") == "native_finalized_SR_group_has_no_trainable_candidates", "invalid terminal transport receipt")
    indices = []
    # The actor reconstructs shards in their original DP order. Native dynamic
    # scheduling may reorder rows; carry the original partition separately.
    by_partition = {}
    for shard in shards:
        partition = shard.get("researchclaw_original_partition")
        masks, rows = shard["loss_masks"], shard["sample_indices"]
        _require(partition is not None and len(partition) == len(masks) == len(rows), "terminal shard fields differ")
        _require(not any(shard.get("dp_padding_mask", [])), "terminal receipt does not permit fabricated padding rows")
        for position, index, mask in zip(partition, rows, masks):
            _require(type(position) is int and position not in by_partition and _zero(mask), "terminal partition duplicated or contains trainable tokens")
            by_partition[position] = int(index)
    _require(sorted(by_partition) == list(range(receipt["native_samples"])), "terminal transport native sample coverage differs")
    indices = [by_partition[i] for i in range(len(by_partition))]
    _require(_digest(indices) == receipt["sample_indices_SHA"], "terminal transport logical identities changed")
    expected_groups = list(range(receipt["rollout_id"] * 16, min(1800, (receipt["rollout_id"] + 1) * 16)))
    _require(receipt["group_ids"] == expected_groups and receipt["logical_candidates"] == _group_size() * len(expected_groups), "terminal transport batch coverage differs")
    _require(set(indices) == {_group_size() * g + c for g in expected_groups for c in range(_group_size())}, "terminal transport candidates missing")
    return receipt
