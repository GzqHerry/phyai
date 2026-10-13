"""AgileX dual Piper robot: ROS bindings and model adapter.

One complete write publishes both arms. Stopping only ends publication: no
services, homing or motor-enable calls are made. RobotDeployment is the only
action scheduler.
"""

from __future__ import annotations

from functools import partial
from dataclasses import field, dataclass

from phyai_robot import (
    StateSlot,
    CommandSlot,
    FeatureSpec,
    VectorAdapter,
    CompositeRobot,
)
from phyai_robot.images import resize_with_pad
from phyai_robot.remote import PolicySpec, Embodiment
from phyai_robot.backends.ros2 import (
    RosCommand,
    Ros2Backend,
    RosObservation,
    image_backends,
)
from phyai_robot.backends.ros2.codecs import decode_rgb_image

from .agilex_codecs import decode_arm, encode_arm

SIDES = ("left", "right")


@dataclass
class AgilexConfig:
    """AgileX ROS topics and node settings; no checkpoint or model settings."""

    # null uses ROS_DOMAIN_ID from the environment.
    domain_id: int | None = None
    node_name: str = "phyai_agilex"
    io_timeout_s: float = 0.1
    # Observation field -> RGB image topic.
    camera_topics: dict[str, str] = field(
        default_factory=lambda: {
            "front_camera": "/camera_f/color/image_raw",
            "left_camera": "/camera_l/color/image_raw",
            "right_camera": "/camera_r/color/image_raw",
        }
    )
    # Model camera name -> observation field.
    model_cameras: dict[str, str] = field(
        default_factory=lambda: {
            "front": "front_camera",
            "left": "left_camera",
            "right": "right_camera",
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
    qos: dict[str, str] = field(
        default_factory=lambda: {
            "camera": "sensor_data",
            "state": "sensor_data",
            "command": "reliable",
        }
    )

    def validate(self) -> None:
        if self.domain_id is not None and not 0 <= self.domain_id <= 232:
            raise ValueError("robot.domain_id must be in [0, 232] or null")
        if not self.node_name.replace("_", "").isalnum():
            raise ValueError("robot.node_name must be a valid ROS node name")
        if not self.io_timeout_s > 0:
            raise ValueError("robot.io_timeout_s must be positive")
        if not set(self.model_cameras.values()) <= set(self.camera_topics):
            raise ValueError("robot.model_cameras must name camera_topics fields")
        for attr in ("state_topics", "command_topics"):
            if set(getattr(self, attr)) != set(SIDES):
                raise ValueError(f"robot.{attr} must map left and right")
        if set(self.state_topics.values()) & set(self.command_topics.values()):
            raise ValueError("state topics and command topics must differ")
        if {"camera", "state", "command"} - set(self.qos):
            raise ValueError("robot.qos needs camera, state and command")


ACTION_SCHEMA = {
    **{
        f"joint_position_{side}": FeatureSpec((6,), "float64", unit="rad")
        for side in SIDES
    },
    **{f"gripper_{side}": FeatureSpec((1,), "float64", unit="m") for side in SIDES},
}


def observation_schema(config: AgilexConfig, image_size: int) -> dict[str, FeatureSpec]:
    """Model-size RGB cameras plus the same joint/gripper fields as commands."""
    return {
        **{
            key: FeatureSpec((image_size, image_size, 3), "uint8")
            for key in config.camera_topics
        },
        **ACTION_SCHEMA,
    }


def make_adapter(config: AgilexConfig, observations: dict[str, FeatureSpec]):
    """Dataset layout: [left 6 joints, left gripper, right 6 joints, right gripper]."""
    fields = [
        key for side in SIDES for key in (f"joint_position_{side}", f"gripper_{side}")
    ]
    return VectorAdapter(
        observation_schema=observations,
        action_schema=ACTION_SCHEMA,
        cameras=config.model_cameras,
        state=[StateSlot(key) for key in fields],
        commands=[CommandSlot(key) for key in fields],
    )


def build(config: AgilexConfig, spec: PolicySpec) -> Embodiment:
    """Compose camera and joint backends plus the adapter, without ROS I/O."""
    from sensor_msgs.msg import Image, JointState

    image_size = int(spec.metadata["image_size"])
    ros = dict(domain_id=config.domain_id, io_timeout_s=config.io_timeout_s)
    # One JointState per arm carries two schema fields. A tuple binding keeps
    # their receive timestamps together and publishes one message per arm.
    states, commands = {}, {}
    for side in SIDES:
        fields = (f"joint_position_{side}", f"gripper_{side}")
        states[fields] = RosObservation(
            config.state_topics[side],
            JointState,
            partial(decode_arm, side=side),
            config.qos["state"],
            recover_on_next_message=True,
        )
        commands[fields] = RosCommand(
            config.command_topics[side],
            JointState,
            partial(encode_arm, side=side, message_type=JointState),
            config.qos["command"],
            stamp=True,
        )
    joints = Ros2Backend(
        observations=states,
        actions=commands,
        # Stopping only ends publication; do not home, hold or disable motors.
        on_stop=lambda backend: None,
        node_name=config.node_name,
        **ros,
    )
    cameras = image_backends(
        config.camera_topics,
        message_type=Image,
        decode=lambda message: resize_with_pad(decode_rgb_image(message), image_size),
        node_name=config.node_name,
        qos=config.qos["camera"],
        **ros,
    )
    schema = observation_schema(config, image_size)
    robot = CompositeRobot(
        observation_schema=schema,
        action_schema=ACTION_SCHEMA,
        backends=[*cameras, joints],
    )
    return Embodiment(robot, make_adapter(config, schema))
