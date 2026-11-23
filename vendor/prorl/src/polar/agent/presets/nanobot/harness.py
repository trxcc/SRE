"""Polar harness for the ClawGym Nanobot agent."""

from __future__ import annotations

import base64
import shlex

from polar.agent.base import BaseHarness
from polar.runtime.models import ExecInput


class NanobotHarness(BaseHarness):
    """Run Nanobot while routing all model calls through the Polar gateway."""

    def run_steps(self, instruction: str) -> list[ExecInput]:
        encoded = base64.b64encode(instruction.encode("utf-8")).decode("ascii")
        turn_type = str(self.settings.get("turn_type", "main"))
        model_id = self.model_name or ""
        command = [
            "python3",
            "-m",
            "clawgym_nanobot.polar_agent",
            "--instruction-b64",
            encoded,
            "--workspace",
            "/polar/session/workspace",
            "--run-dir",
            "/polar/session/nanobot",
            "--log-dir",
            "/polar/session/logs/agent",
            "--session-id",
            "$SESSION_ID",
            "--turn-type",
            turn_type,
        ]
        if model_id:
            command.extend(["--model-id", model_id])
        return [
            ExecInput(
                command=" ".join(
                    item if item == "$SESSION_ID" else shlex.quote(item)
                    for item in command
                ),
                cwd="/polar/session/workspace",
                env={**self.env},
            )
        ]
