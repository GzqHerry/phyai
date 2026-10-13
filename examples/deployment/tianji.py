"""Tianji robot-only client. See ROBOT_ENVIRONMENTS.md for installation/startup.

Run the Pi05 server separately in the PhyAI environment. This process uses ROS
Jazzy and the controller's message overlay. After one discarded warmup and fresh
sensors, preserve ready -> mode -> optional speed -> input=3. Shutdown requests
input=0, never homes the robot, and leaves the model server resident.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from contextlib import ExitStack
from dataclasses import field, dataclass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from phyai_robot import RobotDeployment, ObservationAdapter

from .remote import RemotePolicyConfig
from .options import DeploymentConfig
from .configuration import parse_config
from .robots.tianji import (
    ACTION_SCHEMA,
    TianjiControl,
    TianjiRobotConfig,
    make_tianji_robot,
    wait_for_observation,
)
from .layouts.tianji import observation_schema

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "tianji.yaml"


@dataclass
class TianjiDeploymentConfig:
    """Robot connection, RPC endpoint and control scheduling; no model settings."""

    max_steps: int | None = None
    robot: TianjiRobotConfig = field(default_factory=TianjiRobotConfig)
    remote: RemotePolicyConfig = field(default_factory=RemotePolicyConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def validate(self):
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or null")
        self.robot.validate()
        self.remote.validate()
        self.deployment.to_options()


def main(argv=None):
    """Compose the tianji client; configuration printing exits before any I/O."""
    config = parse_config(
        TianjiDeploymentConfig,
        default_config=DEFAULT_CONFIG,
        argv=argv,
        validate=TianjiDeploymentConfig.validate,
    )
    options = config.deployment.to_options(interpolate_keys=ACTION_SCHEMA)
    # ExitStack unwinds in reverse registration order: stop/close the robot
    # before releasing the model run. The server and its engine stay resident.
    with ExitStack() as resources:
        policy = config.remote.create()
        resources.callback(policy.close)
        # The server's processor chooses image_size, which is needed to build
        # the robot schema before reserving a run or opening any ROS resources.
        spec = policy.info()
        if spec.embodiment != "tianji":
            raise ValueError("Tianji client requires a Tianji model server")
        image_size = spec.metadata["image_size"]
        spec = policy.begin_run(
            embodiment="tianji",
            observation_schema=observation_schema(image_size),
            action_schema=ACTION_SCHEMA,
        )
        if options.execution_horizon > spec.metadata["horizon"]:
            raise ValueError("execution_horizon exceeds checkpoint chunk_size")
        control = TianjiControl(
            node_name=f"{config.robot.node_name}_control",
            service_timeout_s=config.robot.service_timeout_s,
        )
        resources.callback(control.close)
        robot = make_tianji_robot(
            control,
            node_name=config.robot.node_name,
            io_timeout_s=config.robot.io_timeout_s,
            image_size=image_size,
        )
        resources.callback(robot.close)
        control.connect()
        robot.connect()
        observation = wait_for_observation(
            robot,
            timeout_s=options.observation_timeout_s,
            max_age_s=options.max_sample_age_s,
        )
        # Warmup consumes the same first noise draw as the original deployment.
        # Discard its chunk, then check sensor freshness again before running.
        started = time.monotonic()
        warmup = policy.warmup(observation)
        print(
            f"Warmup: {time.monotonic() - started:.3f}s; denoise_steps={spec.metadata['num_inference_steps']}; discarded {len(warmup.actions)} actions",
            flush=True,
        )
        wait_for_observation(
            robot,
            timeout_s=options.observation_timeout_s,
            max_age_s=options.max_sample_age_s,
        )
        control.prepare_user_control(
            mode=config.robot.mode, velocity_ratio=config.robot.velocity_ratio
        )
        print(
            f"Tianji input=3, mode={config.robot.mode}; source={options.action_hz:g}Hz, control={options.control_hz:g}Hz, horizon={options.execution_horizon}; Ctrl-C stops",
            flush=True,
        )
        # RemotePolicy already returns physical ActionChunks. The lightweight
        # adapter passes them through; real Pi05 transforms stay on the server.
        RobotDeployment(
            robot=robot, policy=policy, adapter=ObservationAdapter(), options=options
        ).run(max_steps=config.max_steps)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Deployment interrupted; robot cleanup requested.", file=sys.stderr)
        raise SystemExit(130) from None
