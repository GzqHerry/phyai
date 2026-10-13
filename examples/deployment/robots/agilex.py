"""AgileX assembled from camera and joint/command Ros2Backend instances.

One complete write publishes both arms. No interpolation, clipping, services,
homing or motor-enable calls. RobotDeployment is the only action scheduler.
"""

from __future__ import annotations

import os
import time
from functools import partial
from dataclasses import field, dataclass

from phyai_robot import CompositeRobot, ObservationNotReady
from phyai_robot.backends.ros2 import Ros2Backend, RosObservation, RosCommand

from .agilex_codecs import decode_image, decode_state_fields, encode_state_fields
from ..layouts.agilex import (
    SIDES,
    VIEWS,
    JOINT_NAMES,
    ACTION_SCHEMA,
    action_from_target,
    target_from_action,
    state_from_observation,
)
from ..layouts.agilex import (
    observation_schema as layout_schema,
)


@dataclass
class AgilexRobotConfig:
    """Native ROS topic/layout settings; no checkpoint or model dependencies.

    Camera dimensions describe the wire schema rather than requesting a resize.
    They must match the model server's camera_height/camera_width. Both backend
    contexts share domain_id; localhost_only configures the native ROS process.
    """

    node_name: str = "phyai_agilex"
    domain_id: int = 0
    localhost_only: bool = False
    camera_height: int = 480
    camera_width: int = 640
    # Standard sensor_msgs are sufficient; custom driver overlays are not imported.
    image_topics: dict[str, str] = field(
        default_factory=lambda: {
            "front": "/camera_f/color/image_raw",
            "left": "/camera_l/color/image_raw",
            "right": "/camera_r/color/image_raw",
        }
    )
    state_topics: dict[str, str] = field(
        default_factory=lambda: {
            "left": "/puppet/joint_left",
            "right": "/puppet/joint_right",
        }
    )
    command_topics: dict[str, str] = field(
        default_factory=lambda: {
            "left": "/joint_left_states",
            "right": "/joint_right_states",
        }
    )

    def validate(self):
        if self.camera_height < 1 or self.camera_width < 1:
            raise ValueError("Camera dimensions must be positive")
        if not 0 <= self.domain_id <= 232:
            raise ValueError("ROS domain_id must be in [0, 232]")
        if not self.node_name.replace("_", "").isalnum() or self.node_name[0].isdigit():
            raise ValueError("Invalid ROS node_name")
        for attr, keys in (
            ("image_topics", VIEWS),
            ("state_topics", SIDES),
            ("command_topics", SIDES),
        ):
            topics = getattr(self, attr)
            if set(topics) != set(keys) or any(
                not value.startswith("/") for value in topics.values()
            ):
                raise ValueError(
                    f"robot.{attr} must map {keys} to absolute ROS topic names"
                )
            if len(set(topics.values())) != len(topics):
                raise ValueError(f"robot.{attr} cannot contain duplicate topics")
        if set(self.state_topics.values()) & set(self.command_topics.values()):
            raise ValueError("State topics and command topics must differ")


def observation_schema(config):
    return layout_schema(config.camera_height, config.camera_width)


def make_agilex_robot(config):
    """Compose camera and joint/command transports without connecting hardware.

    Imports message classes but does not create a ROS context. Keeping camera
    decoding on its own worker prevents those callbacks from occupying the
    command worker. CompositeRobot merges both caches and routes each complete
    action to the command backend; RobotDeployment alone schedules control ticks.
    """
    config.validate()
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState

    os.environ["ROS_LOCALHOST_ONLY"] = "1" if config.localhost_only else "0"
    node_name = f"{config.node_name}_{os.getpid()}"
    cameras = Ros2Backend(
        observations={
            f"{view}_camera": RosObservation(
                config.image_topics[view],
                Image,
                decode_image,
                qos_profile_sensor_data,
                recover_on_next_message=True,
            )
            for view in VIEWS
        },
        actions={},
        node_name=f"{node_name}_cameras",
        domain_id=config.domain_id,
    )
    states, commands = {}, {}
    for side in SIDES:
        # One seven-position JointState owns two schema fields. A tuple binding
        # keeps their receive timestamps together and emits one command message,
        # preserving left-before-right order without duplicate subscriptions.
        fields = (f"joint_position_{side}", f"gripper_{side}")
        states[fields] = RosObservation(
            config.state_topics[side],
            JointState,
            partial(decode_state_fields, side=side),
            qos_profile_sensor_data,
            recover_on_next_message=True,
        )
        commands[fields] = RosCommand(
            config.command_topics[side],
            JointState,
            partial(encode_state_fields, side=side, message_type=JointState),
            qos=10,
            stamp=True,
        )
    joints = Ros2Backend(
        observations=states,
        actions=commands,
        # Stopping only ends publication; do not home, hold or disable motors.
        on_stop=lambda node: None,
        node_name=node_name,
        domain_id=config.domain_id,
    )
    return CompositeRobot(
        observation_schema=observation_schema(config),
        action_schema=ACTION_SCHEMA,
        backends=[cameras, joints],
    )


def wait_for_observation(robot, *, timeout_s, max_age_s):
    """Wait for all required fields to exist and meet the receive-age budget.

    connect() starts subscribers before their first callback, and warmup may
    outlive the current samples. Only missing/stale observations are retried;
    transport and decode errors propagate. This checks local receive age, not
    synchronized capture time or whether each field is newer than a prior read.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            observation = robot.get_observation()
            now = time.monotonic_ns()
            stale = [
                key
                for key, sample in observation.samples.items()
                if not 0 <= (now - sample.received_at_ns) / 1e9 <= max_age_s
            ]
            if stale:
                raise ObservationNotReady(f"Stale sensors: {stale}")
            return observation
        except ObservationNotReady:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)
