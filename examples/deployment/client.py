"""Shared robot-client configuration and entry point. No model imports.

A robot entry point supplies its own robot config and an Embodiment builder;
this module owns the RPC, control-loop and scheduling settings and runs the
common phyai_robot.remote.run_remote lifecycle.
"""

from __future__ import annotations

import sys
from typing import Any, TypeVar
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Sequence

from phyai_robot import StopAndRefill, DeploymentOptions
from phyai_robot.remote import PolicySpec, Embodiment, RemotePolicy, run_remote

from .configuration import parse_config


@dataclass
class RemoteConfig:
    """Server endpoint and per-request waits, independent of control rates."""

    endpoint: str = "tcp://127.0.0.1:5555"
    # Covers info, begin, predict and end. The first warmup may compile.
    timeout_s: float = 2.0
    warmup_timeout_s: float = 120.0

    def create(self) -> RemotePolicy:
        return RemotePolicy(
            self.endpoint,
            timeout_s=self.timeout_s,
            warmup_timeout_s=self.warmup_timeout_s,
        )


@dataclass
class ControlConfig:
    """Control loop and sensor freshness; see phyai_robot.DeploymentOptions."""

    # Commands per second while a chunk executes; sets interpolation density.
    control_hz: float = 200.0
    # Every observation field must have been received within this age.
    max_sample_age_s: float = 0.5
    # Wait for a complete, fresh snapshot before each prediction.
    observation_timeout_s: float = 10.0
    # Wait for in-flight inference after the robot has been stopped.
    shutdown_timeout_s: float = 2.0
    # Allowed lateness of a scheduled send; larger lateness stops the robot.
    max_control_lateness_s: float = 0.01

    def to_options(self) -> DeploymentOptions:
        return DeploymentOptions(
            control_hz=self.control_hz,
            max_sample_age_s=self.max_sample_age_s,
            observation_timeout_s=self.observation_timeout_s,
            shutdown_timeout_s=self.shutdown_timeout_s,
            max_control_lateness_s=self.max_control_lateness_s,
        )


@dataclass
class ScheduleConfig:
    """How predicted chunks are executed."""

    # Only stop_and_refill exists today; RTC would be another strategy.
    strategy: str = "stop_and_refill"
    # Model targets consumed per second; with execution_horizon this sets the
    # duration of each executed chunk (horizon / action_hz seconds).
    action_hz: float = 25.0
    # Number of leading targets executed from each prediction; null = all.
    execution_horizon: int | None = 20

    def create(self) -> StopAndRefill:
        if self.strategy != "stop_and_refill":
            raise ValueError(f"unknown schedule.strategy: {self.strategy!r}")
        return StopAndRefill(
            action_hz=self.action_hz, execution_horizon=self.execution_horizon
        )


ConfigT = TypeVar("ConfigT")


def validate_client(config: Any) -> None:
    """Validate the shared sections and the robot's own section."""
    if config.max_steps is not None and config.max_steps < 1:
        raise ValueError("max_steps must be positive or null")
    if not config.remote.endpoint.strip():
        raise ValueError("remote.endpoint must not be empty")
    config.remote.create()
    config.control.to_options()
    config.schedule.create()
    config.robot.validate()


def main(
    schema: type[ConfigT],
    build: Callable[[Any, PolicySpec], Embodiment],
    *,
    default_config: Path,
    argv: Sequence[str] | None = None,
) -> None:
    """Parse configuration, then run one robot session until Ctrl-C or max_steps."""
    config: Any = parse_config(
        schema, default_config=default_config, argv=argv, validate=validate_client
    )
    try:
        run_remote(
            config.remote.create(),
            build=lambda spec: build(config.robot, spec),
            scheduler=config.schedule.create(),
            options=config.control.to_options(),
            max_steps=config.max_steps,
            report=lambda message: print(message, flush=True),
        )
    except KeyboardInterrupt:
        print("Deployment interrupted; robot cleanup completed.", file=sys.stderr)
        raise SystemExit(130) from None
