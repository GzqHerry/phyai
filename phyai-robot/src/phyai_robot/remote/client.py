"""The robot-side lifecycle for a resident model server.

Every robot client follows the same order: read the model contract, build the
robot and its adapter without I/O, reserve a run, connect, discard one warmup,
check sensors again, enable device commands, and run the control loop. Cleanup
always stops and closes the robot before releasing the model run.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple
from contextlib import ExitStack
from collections.abc import Callable

from ..robot import Robot, wait_for_observation
from ..policy import PolicyAdapter
from ._wire import PolicySpec, schema_to_json
from .policy import RemotePolicy
from ..scheduling import Scheduler, InferenceContext
from ..deployment import RobotDeployment, DeploymentOptions


class Embodiment(NamedTuple):
    """An unconnected robot, its model adapter, and an optional enable step.

    adapter.request_schema must equal the server's input schema. activate runs
    once after warmup and a fresh observation, immediately before control; use
    it to switch a controller to external commands. Robot.stop must undo it.
    """

    robot: Robot
    adapter: PolicyAdapter[Any, Any]
    activate: Callable[[], None] | None = None


def run_remote(
    policy: RemotePolicy,
    *,
    build: Callable[[PolicySpec], Embodiment],
    scheduler: Scheduler,
    options: DeploymentOptions,
    max_steps: int | None = None,
    report: Callable[[str], None] = lambda message: None,
) -> None:
    """Run one robot session against a resident model server.

    build receives the server's spec (for example its image size) and must not
    perform I/O. The warmup result is converted and validated like any other
    chunk, so adapter, horizon and schema errors surface before activation.
    """
    with ExitStack() as resources:
        # ExitStack unwinds in reverse: stop/close the robot, then end the run.
        resources.callback(policy.close)
        spec = policy.info()
        embodiment = build(spec)
        robot, adapter = embodiment.robot, embodiment.adapter
        expected = schema_to_json(spec.inputs)
        actual = schema_to_json(getattr(adapter, "request_schema", {}))
        if actual != expected:
            raise ValueError(
                f"robot adapter produces {actual}, but the model expects {expected}"
            )
        resources.callback(robot.close)
        deployment = RobotDeployment(
            robot=robot,
            policy=policy,
            adapter=adapter,
            scheduler=scheduler,
            options=options,
        )
        policy.begin_run()
        robot.connect()

        def fresh_observation():
            return wait_for_observation(
                robot,
                timeout_s=options.observation_timeout_s,
                max_age_s=options.max_sample_age_s,
            )

        observation = fresh_observation()
        started = time.monotonic()
        result = policy.warmup(adapter.to_request(observation, InferenceContext()))
        warmup = deployment.prepare_chunk(adapter.to_actions(result, observation))
        report(
            f"Warmup took {time.monotonic() - started:.3f}s; "
            f"discarded {len(warmup.actions)} targets"
        )
        # Warmup may outlive the samples it used; never enable on stale sensors.
        fresh_observation()
        if embodiment.activate is not None:
            embodiment.activate()
        report("Robot commands enabled; Ctrl-C stops")
        deployment.run(max_steps=max_steps)
