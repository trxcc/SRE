"""Coalesce reduced metric events into one complete record per RL iteration."""

from __future__ import annotations

import argparse
import json
import logging
import numbers
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clawgym_nanobot.metrics_jsonl import append_metric_event, validate_metric_event


LOGGER = logging.getLogger("metrics-step-coalescer")
STOP_REQUESTED = False


def _request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOGGER.info("received signal %s; draining source", signum)


def classify_component(event: dict[str, Any]) -> str:
    """Classify one already-reduced event by its place in an RL iteration."""
    event = validate_metric_event(event)
    step_key = event["step_key"]
    metrics = event["metrics"]
    keys = set(metrics)

    if step_key == "eval/step":
        return "eval"
    if step_key == "train/step":
        if any(key.startswith("train/critic-") for key in keys):
            return "critic_train"
        return "actor_train"
    if step_key != "rollout/step":
        raise ValueError(f"unsupported source step key {step_key!r}")

    if "perf/actor_train_time" in keys or "perf/train_time" in keys:
        return "actor_perf"
    if "rollout/advantages" in keys or "rollout/returns" in keys:
        return "actor_rollout"
    return "rollout"


@dataclass(frozen=True)
class CoalescerConfig:
    algorithm: str
    num_critic_only_steps: int = 0
    num_steps_per_rollout: int = 1
    critic_train_epochs: int = 1
    first_rollout_iteration: int = 0
    eval_interval: int | None = None
    initial_eval: bool = True
    compact_policy_events: bool = False

    def required_component_counts(self, iteration: int) -> dict[str, int]:
        if self.algorithm == "eval_only":
            return {"eval": 1}
        if iteration < self.first_rollout_iteration:
            # A resumed run can evaluate the loaded checkpoint immediately
            # before its first new rollout.  Such historical-checkpoint rows
            # have no matching rollout/train/perf events in this process.
            return {"eval": 1}

        required = {"rollout": 1}
        if self.algorithm == "ppo":
            required["critic_train"] = (
                self.num_steps_per_rollout * self.critic_train_epochs
            )
            if iteration >= self.num_critic_only_steps:
                required.update(
                    {
                        "actor_rollout": 1,
                        "actor_train": self.num_steps_per_rollout,
                        "actor_perf": 1,
                    }
                )
        elif self.algorithm in {"grpo", "policy_gradient"}:
            required["actor_train"] = self.num_steps_per_rollout
            if not self.compact_policy_events:
                required.update(
                    {
                        "actor_rollout": 1,
                        "actor_perf": 1,
                    }
                )
        elif self.algorithm == "opsd":
            # OPSD consumes the sampled response and user feedback directly.
            # It intentionally has no advantage/return event, so waiting for
            # ``actor_rollout`` would permanently gate otherwise complete
            # iterations out of canonical JSONL and W&B.
            required.update(
                {
                    "actor_train": self.num_steps_per_rollout,
                    "actor_perf": 1,
                }
            )
        else:
            raise ValueError(f"unsupported algorithm {self.algorithm!r}")

        eval_due = (self.initial_eval and iteration == 0) or (
            self.eval_interval is not None
            and iteration > 0
            and iteration % self.eval_interval == 0
        )
        if eval_due:
            required["eval"] = 1
        return required

    def iteration_for_event(self, event: dict[str, Any]) -> int:
        raw_step = event["step"]
        if isinstance(raw_step, bool) or int(raw_step) != raw_step:
            raise ValueError(f"step must be an integer, got {raw_step!r}")
        step = int(raw_step)
        if event["step_key"] == "train/step":
            return step // self.num_steps_per_rollout
        return step


@dataclass
class PendingIteration:
    events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    event_ids: set[str] = field(default_factory=set)

    def add(self, event: dict[str, Any], component: str) -> None:
        event_id = str(event.get("event_id", ""))
        if event_id and event_id in self.event_ids:
            return
        component_events = self.events.setdefault(component, [])
        if event["step_key"] == "train/step":
            for existing in component_events:
                if existing["step"] == event["step"]:
                    if existing != event:
                        if component == "critic_train":
                            continue
                        raise ValueError(
                            f"conflicting {component} events for optimizer step {event['step']}"
                        )
                    return
        component_events.append(event)
        if event_id:
            self.event_ids.add(event_id)

    def component_counts(self) -> dict[str, int]:
        return {component: len(events) for component, events in self.events.items()}

    @staticmethod
    def _merge_metric(
        metrics: dict[str, Any],
        key: str,
        value: Any,
        *,
        iteration: int,
        conflicts: dict[str, list[Any]] | None = None,
    ) -> None:
        if key in metrics and metrics[key] != value:
            # Native Megatron emits both the ClawGym reward aggregate and the
            # generic actor-rollout aggregate.  They are both named
            # ``rollout/raw_reward`` but use different denominators when
            # masked self-reward samples are present.  Keep the first value
            # (the ClawGym aggregate is emitted first) and retain alternates
            # as diagnostics so the coalescer cannot kill W&B logging.
            if key == "rollout/raw_reward" and conflicts is not None:
                conflicts.setdefault(key, [metrics[key]]).append(value)
                return
            raise ValueError(
                f"conflicting metric {key!r} within iteration {iteration}: "
                f"{metrics[key]!r} != {value!r}"
            )
        metrics[key] = value

    @staticmethod
    def _aggregate_train_events(
        events: list[dict[str, Any]],
        component: str,
    ) -> dict[str, Any]:
        ordered = sorted(events, key=lambda event: int(event["step"]))
        keys = set().union(*(event["metrics"] for event in ordered))
        keys.discard("train/step")
        aggregated: dict[str, Any] = {}
        for key in sorted(keys):
            values = [event["metrics"][key] for event in ordered if key in event["metrics"]]
            if len(values) != len(ordered):
                raise ValueError(f"metric {key!r} is missing from one or more {component} substeps")
            if all(value == values[0] for value in values[1:]):
                aggregated[key] = values[0]
            elif all(isinstance(value, numbers.Real) and not isinstance(value, bool) for value in values):
                aggregated[key] = sum(float(value) for value in values) / len(values)
            else:
                raise ValueError(f"cannot aggregate non-numeric metric {key!r} across {component} substeps")

        prefix = "train/critic-" if component == "critic_train" else "train/"
        steps = [int(event["step"]) for event in ordered]
        aggregated[f"{prefix}optimizer_step_first"] = steps[0]
        aggregated[f"{prefix}optimizer_step_last"] = steps[-1]
        aggregated[f"{prefix}num_optimizer_steps"] = len(steps)
        global_batch_key = f"{prefix}global_batch_size"
        if global_batch_key in aggregated:
            aggregated[f"{prefix}samples_per_rollout"] = sum(
                int(event["metrics"][global_batch_key]) for event in ordered
            )
        return aggregated

    def merged_metrics(self, iteration: int) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        conflicts: dict[str, list[Any]] = {}
        for component, events in self.events.items():
            if component in {"critic_train", "actor_train"}:
                component_metrics = self._aggregate_train_events(events, component)
                for key, value in component_metrics.items():
                    self._merge_metric(
                        metrics,
                        key,
                        value,
                        iteration=iteration,
                        conflicts=conflicts,
                    )
                continue
            for event in events:
                for key, value in event["metrics"].items():
                    self._merge_metric(
                        metrics,
                        key,
                        value,
                        iteration=iteration,
                        conflicts=conflicts,
                    )
        for key, values in conflicts.items():
            # Metric names are deliberately flat and deterministic so W&B can
            # chart the alternate without replacing the canonical value.
            safe_key = key.replace("/", "_")
            metrics[f"coalescer/{safe_key}_conflict_count"] = len(values) - 1
            for index, alternate in enumerate(values[1:], start=1):
                if isinstance(alternate, numbers.Real) and not isinstance(alternate, bool):
                    metrics[f"coalescer/{safe_key}_alternate_{index}"] = float(alternate)
        metrics["train/step"] = iteration
        metrics["iteration"] = iteration
        return metrics


class StepCoalescer:
    """Stateful event coalescer with idempotent output recovery."""

    def __init__(self, output: Path, config: CoalescerConfig):
        self.output = output
        self.config = config
        self.pending: dict[int, PendingIteration] = {}
        self.emitted = self._load_emitted_iterations()

    def _load_emitted_iterations(self) -> set[int]:
        emitted: set[int] = set()
        try:
            handle = self.output.open("r", encoding="utf-8")
        except FileNotFoundError:
            return emitted
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise ValueError(f"incomplete output record at line {line_number}")
                event = validate_metric_event(json.loads(line))
                if event["step_key"] != "iteration":
                    raise ValueError(f"unexpected output step key at line {line_number}: {event['step_key']!r}")
                emitted.add(int(event["step"]))
        return emitted

    def add(self, event: dict[str, Any]) -> bool:
        event = validate_metric_event(event)
        iteration = self.config.iteration_for_event(event)
        if iteration in self.emitted:
            return False

        component = classify_component(event)
        pending = self.pending.setdefault(iteration, PendingIteration())
        pending.add(event, component)
        required = self.config.required_component_counts(iteration)
        component_counts = pending.component_counts()
        if any(component_counts.get(component, 0) < count for component, count in required.items()):
            return False

        metrics = pending.merged_metrics(iteration)
        append_metric_event(self.output, metrics, step_key="iteration")
        self.emitted.add(iteration)
        del self.pending[iteration]
        LOGGER.info(
            "emitted iteration=%d component_counts=%s metrics=%d",
            iteration,
            ",".join(
                f"{component}:{count}"
                for component, count in sorted(component_counts.items())
            ),
            len(metrics),
        )
        return True

    def flush_train_complete_iterations_missing_eval(self) -> list[int]:
        """Emit terminal iterations whose only missing component is eval."""
        flushed: list[int] = []
        for iteration, pending in sorted(list(self.pending.items())):
            if iteration in self.emitted:
                continue
            required = self.config.required_component_counts(iteration)
            if "eval" not in required:
                continue
            component_counts = pending.component_counts()
            missing = {
                component
                for component, count in required.items()
                if component_counts.get(component, 0) < count
            }
            if missing != {"eval"}:
                continue
            metrics = pending.merged_metrics(iteration)
            metrics["coalescer/eval_missing"] = 1
            append_metric_event(self.output, metrics, step_key="iteration")
            self.emitted.add(iteration)
            del self.pending[iteration]
            flushed.append(iteration)
            LOGGER.warning(
                "emitted train-complete iteration=%d without scheduled eval "
                "component_counts=%s metrics=%d",
                iteration,
                ",".join(
                    f"{component}:{count}"
                    for component, count in sorted(component_counts.items())
                ),
                len(metrics),
            )
        return flushed

    def flush_train_complete_iterations_missing_eval(self) -> list[int]:
        """Emit terminal iterations whose only missing component is eval."""
        flushed: list[int] = []
        for iteration, pending in sorted(list(self.pending.items())):
            if iteration in self.emitted:
                continue
            required = self.config.required_component_counts(iteration)
            if "eval" not in required:
                continue
            component_counts = pending.component_counts()
            missing = {
                component
                for component, count in required.items()
                if component_counts.get(component, 0) < count
            }
            if missing != {"eval"}:
                continue
            metrics = pending.merged_metrics(iteration)
            metrics["coalescer/eval_missing"] = 1
            append_metric_event(self.output, metrics, step_key="iteration")
            self.emitted.add(iteration)
            del self.pending[iteration]
            flushed.append(iteration)
            LOGGER.warning(
                "emitted train-complete iteration=%d without scheduled eval "
                "component_counts=%s metrics=%d",
                iteration,
                ",".join(
                    f"{component}:{count}"
                    for component, count in sorted(component_counts.items())
                ),
                len(metrics),
            )
        return flushed


def run(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    completion_marker = args.completion_marker.resolve() if args.completion_marker else None
    config = CoalescerConfig(
        algorithm=args.algorithm,
        num_critic_only_steps=args.num_critic_only_steps,
        num_steps_per_rollout=args.num_steps_per_rollout,
        critic_train_epochs=args.critic_train_epochs,
        first_rollout_iteration=args.first_rollout_iteration,
        eval_interval=args.eval_interval,
        initial_eval=not args.skip_initial_eval,
        compact_policy_events=args.compact_policy_events,
    )
    coalescer = StepCoalescer(output, config)
    LOGGER.info(
        "coalescer ready source=%s output=%s already_emitted=%d algorithm=%s "
        "compact_policy_events=%s",
        source,
        output,
        len(coalescer.emitted),
        config.algorithm,
        config.compact_policy_events,
    )

    with source.open("rb") as handle:
        while True:
            record_offset = handle.tell()
            line = handle.readline()
            if not line:
                done = STOP_REQUESTED or bool(args.done_file and args.done_file.exists())
                if done:
                    break
                time.sleep(args.poll_seconds)
                continue
            if not line.endswith(b"\n"):
                handle.seek(record_offset)
                if STOP_REQUESTED:
                    LOGGER.warning("leaving incomplete source record at byte %d", record_offset)
                    break
                time.sleep(args.poll_seconds)
                continue
            try:
                event = json.loads(line)
                coalescer.add(event)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"invalid metric record at byte {record_offset}: {exc}") from exc

    coalescer.flush_train_complete_iterations_missing_eval()
    coalescer.flush_train_complete_iterations_missing_eval()
    if coalescer.pending:
        detail = {
            iteration: pending.component_counts()
            for iteration, pending in sorted(coalescer.pending.items())
            if iteration not in coalescer.emitted
        }
        LOGGER.warning("source ended with incomplete iterations: %s", detail)
    if completion_marker is not None:
        completion_marker.parent.mkdir(parents=True, exist_ok=True)
        completion_marker.touch()
    LOGGER.info("coalescer stopped emitted=%d", len(coalescer.emitted))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--algorithm",
        choices=["grpo", "policy_gradient", "ppo", "opsd", "eval_only"],
        required=True,
    )
    parser.add_argument("--num-critic-only-steps", type=int, default=0)
    parser.add_argument("--num-steps-per-rollout", type=int, default=1)
    parser.add_argument("--critic-train-epochs", type=int, default=1)
    parser.add_argument("--first-rollout-iteration", type=int, default=0)
    parser.add_argument(
        "--compact-policy-events",
        action="store_true",
        help=(
            "For GRPO/policy-gradient backends that emit one aggregate rollout "
            "event and one train event, do not wait for separate actor-rollout "
            "and actor-perf events."
        ),
    )
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--done-file", type=Path)
    parser.add_argument("--completion-marker", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.num_critic_only_steps < 0:
        parser.error("--num-critic-only-steps must be non-negative")
    if args.num_steps_per_rollout <= 0:
        parser.error("--num-steps-per-rollout must be positive")
    if args.critic_train_epochs <= 0:
        parser.error("--critic-train-epochs must be positive")
    if args.first_rollout_iteration < 0:
        parser.error("--first-rollout-iteration must be non-negative")
    if args.eval_interval is not None and args.eval_interval <= 0:
        parser.error("--eval-interval must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if not args.source.is_file():
        parser.error(f"--source is not a file: {args.source}")
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
