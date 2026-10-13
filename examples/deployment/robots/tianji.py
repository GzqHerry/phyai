"""Tianji (Marvin Pro) robot: ROS bindings, model adapter and controller enable.

Data flow: ROS message -> decoder -> cached Sample -> Observation -> adapter ->
model state/images; model action rows -> adapter -> complete Action -> one
publication per arm and gripper. RobotDeployment owns timing and interpolation.

Before commands are routed, activate() calls set_ready, set_mode, optional
set_vel_ratio and finally set_input(3) (User/Custom joint input). Stopping the
robot calls set_input(0). It never homes the arms, restores a previous mode or
sends a gripper target; it is not an emergency stop.
"""

from __future__ import annotations

from typing import Any
from functools import partial
from dataclasses import field, dataclass

import numpy as np
from phyai_robot import (
    Action,
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
from phyai_robot.backends.ros2.backend import SERVICE_POLL_MARGIN_S
from phyai_robot.backends.ros2.codecs import (
    POSE_NAMES,
    WRENCH_NAMES,
    decode_pose,
    decode_wrench,
    decode_nv12_image,
)

from .tianji_codecs import (
    ARM_DOF,
    GRIPPER_FEEDBACK_SIZE,
    encode_arm,
    encode_gripper,
    decode_arm_positions,
    decode_gripper_feedback,
)

SIDES = ("left", "right")
MODE_JOINT_POSITION = 1
MODE_JOINT_IMPEDANCE = 3
INPUT_IDLE = 0
INPUT_USER = 3
JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, ARM_DOF + 1))
GRIPPER_FEEDBACK_NAMES = (
    "position",
    "velocity",
    "torque",
    "mos_temperature",
    "motor_temperature",
)


def _sides(**topics: str) -> Any:
    return field(default_factory=lambda: dict(topics))


@dataclass
class TianjiConfig:
    """Tianji ROS connection, controller and calibration settings.

    Topic maps are keyed by "left"/"right" (or by camera field). QoS values are
    names accepted by phyai_robot.backends.ros2.resolve_qos and must be
    compatible with the actual publishers/subscribers.
    """

    # ROS domain of the Apex stack; null uses ROS_DOMAIN_ID from the environment.
    domain_id: int | None = None
    node_name: str = "phyai_tianji"
    # 1: joint position; 3: joint impedance. Mode 2 does not accept joint targets.
    mode: int = MODE_JOINT_IMPEDANCE
    # Controller velocity percentage [1, 100]; null leaves it unchanged.
    velocity_ratio: int | None = None
    # End-to-end budget of each controller service call, including stop.
    service_timeout_s: float = 3.0
    # Budget for publishing one command.
    io_timeout_s: float = 0.1
    # Gripper feedback [closed, open] in motor radians; maps model state to [0, 1].
    gripper_limits_rad: list[float] = field(default_factory=lambda: [0.0, 1.6])
    # Observation field -> NV12 image topic.
    camera_topics: dict[str, str] = field(
        default_factory=lambda: {
            "head_camera": "/camera/left_eye/image_nv12",
            "left_wrist_camera": "/camera/left_wrist/image_nv12",
            "right_wrist_camera": "/camera/right_wrist/image_nv12",
        }
    )
    # Model camera name -> observation field.
    model_cameras: dict[str, str] = field(
        default_factory=lambda: {
            "head_left": "head_camera",
            "left_wrist": "left_wrist_camera",
            "right_wrist": "right_wrist_camera",
        }
    )
    joint_feedback_topic: str = "/tj/info/joint_feedback"
    gripper_feedback_topics: dict[str, str] = _sides(
        left="/info/gripper_feedback_L", right="/info/gripper_feedback_R"
    )
    # Optional extra observations; empty maps subscribe to nothing.
    eef_topics: dict[str, str] = field(default_factory=dict)
    wrench_topics: dict[str, str] = field(default_factory=dict)
    joint_command_topics: dict[str, str] = _sides(
        left="/tj/control/user/joint_cmd_A", right="/tj/control/user/joint_cmd_B"
    )
    gripper_command_topics: dict[str, str] = _sides(
        left="/control/gripperValueL", right="/control/gripperValueR"
    )
    service_namespace: str = "/tj/control"
    qos: dict[str, str] = field(
        default_factory=lambda: {
            "camera": "sensor_data",
            "joint_feedback": "sensor_data",
            "gripper_feedback": "reliable",
            "eef": "reliable",
            "wrench": "sensor_data",
            "joint_command": "sensor_data",
            "gripper_command": "reliable",
        }
    )

    def validate(self) -> None:
        """Reject incompatible settings before creating ROS resources."""
        if self.domain_id is not None and not 0 <= self.domain_id <= 232:
            raise ValueError("robot.domain_id must be in [0, 232] or null")
        if self.mode not in (MODE_JOINT_POSITION, MODE_JOINT_IMPEDANCE):
            raise ValueError("robot.mode must be 1 (joint position) or 3 (impedance)")
        if self.velocity_ratio is not None and not 1 <= self.velocity_ratio <= 100:
            raise ValueError("robot.velocity_ratio must be in [1, 100] or null")
        if not self.node_name.replace("_", "").isalnum():
            raise ValueError("robot.node_name must be a valid ROS node name")
        for name in ("service_timeout_s", "io_timeout_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"robot.{name} must be positive")
        low, high = self.gripper_limits_rad
        if not np.isfinite((low, high)).all() or low >= high:
            raise ValueError("robot.gripper_limits_rad must be finite and increasing")
        if not set(self.model_cameras.values()) <= set(self.camera_topics):
            raise ValueError("robot.model_cameras must name camera_topics fields")
        for attr in (
            "gripper_feedback_topics",
            "joint_command_topics",
            "gripper_command_topics",
        ):
            if set(getattr(self, attr)) != set(SIDES):
                raise ValueError(f"robot.{attr} must map left and right")
        for attr in ("eef_topics", "wrench_topics"):
            if not set(getattr(self, attr)) <= set(SIDES):
                raise ValueError(f"robot.{attr} keys must be left/right")
        missing = {
            "camera",
            "joint_feedback",
            "gripper_feedback",
            "eef",
            "wrench",
            "joint_command",
            "gripper_command",
        } - set(self.qos)
        if missing:
            raise ValueError(f"robot.qos is missing {sorted(missing)}")


ACTION_SCHEMA = {
    **{
        f"joint_position_{side}": FeatureSpec(
            (ARM_DOF,), "float64", unit="rad", names=JOINT_NAMES
        )
        for side in SIDES
    },
    **{
        f"gripper_{side}": FeatureSpec(
            (1,), "float32", unit="normalized", names=("target_position",)
        )
        for side in SIDES
    },
}


def observation_schema(config: TianjiConfig, image_size: int) -> dict[str, FeatureSpec]:
    """Camera fields use the model's square image size; decoders resize to it."""
    schema = {
        key: FeatureSpec((image_size, image_size, 3), "uint8")
        for key in config.camera_topics
    }
    for side in SIDES:
        schema[f"joint_position_{side}"] = ACTION_SCHEMA[f"joint_position_{side}"]
        schema[f"gripper_feedback_{side}"] = FeatureSpec(
            (GRIPPER_FEEDBACK_SIZE,), "float32", names=GRIPPER_FEEDBACK_NAMES
        )
    for side in config.eef_topics:
        schema[f"eef_{side}"] = FeatureSpec((7,), "float64", names=POSE_NAMES)
    for side in config.wrench_topics:
        schema[f"wrench_{side}"] = FeatureSpec((6,), "float64", names=WRENCH_NAMES)
    return schema


def make_adapter(config: TianjiConfig, observations: dict[str, FeatureSpec]):
    """Dataset layout: [left joints 0:7, left gripper, right joints, right gripper].

    Gripper state is feedback position mapped from gripper_limits_rad to [0, 1],
    unclipped; gripper commands are clamped to [0, 1].
    """
    limits = tuple(config.gripper_limits_rad)
    return VectorAdapter(
        observation_schema=observations,
        action_schema=ACTION_SCHEMA,
        cameras=config.model_cameras,
        state=[
            slot
            for side in SIDES
            for slot in (
                StateSlot(f"joint_position_{side}"),
                StateSlot(f"gripper_feedback_{side}", index=(0,), limits=limits),
            )
        ],
        commands=[
            slot
            for side in SIDES
            for slot in (
                CommandSlot(f"joint_position_{side}"),
                CommandSlot(f"gripper_{side}", clip=(0.0, 1.0)),
            )
        ],
    )


def check_gripper_range(action: Action) -> None:
    """Reject normalized gripper targets outside [0, 1] before publishing."""
    for side in SIDES:
        value = float(action[f"gripper_{side}"][0])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"gripper_{side} target must be in [0, 1], got {value}")


class TianjiController:
    """Route the joint mux to User/Custom input and back, via controller services.

    Service calls run on the command backend's worker, so no second ROS context
    is needed. Stop requests idle input only after activate() has started, so
    connecting and closing without enabling never changes controller state.
    """

    def __init__(self, config: TianjiConfig) -> None:
        self._config = config
        self._enabled = False
        self.backend: Ros2Backend | None = None

    def _call(self, backend: Ros2Backend, name: str, value: int | None) -> None:
        from std_srvs.srv import Trigger
        from marvin_msgs.srv import Int

        if value is None:
            service_type, request = Trigger, Trigger.Request()
        else:
            service_type, request = Int, Int.Request(data=int(value))
        backend.call_service(
            service_type,
            f"{self._config.service_namespace}/{name}",
            request,
            timeout_s=self._config.service_timeout_s,
        )

    def activate(self) -> None:
        """Ready, mode, optional velocity, then input=3; not a transaction."""
        assert self.backend is not None
        config = self._config
        # Allow stop to request idle even if a later call fails.
        self._enabled = True
        self._call(self.backend, "set_ready", None)
        self._call(self.backend, "set_mode", config.mode)
        if config.velocity_ratio is not None:
            self._call(self.backend, "set_vel_ratio", config.velocity_ratio)
        self._call(self.backend, "set_input", INPUT_USER)

    def on_stop(self, backend: Ros2Backend) -> None:
        if self._enabled:
            self._call(backend, "set_input", INPUT_IDLE)


def build(config: TianjiConfig, spec: PolicySpec) -> Embodiment:
    """Compose the robot, adapter and controller without any ROS I/O."""
    from std_msgs.msg import Float32, Float32MultiArray
    from marvin_msgs.msg import JointcmdArm, Jointfeedback
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import PoseStamped, WrenchStamped

    image_size = int(spec.metadata["image_size"])
    qos = config.qos
    ros = dict(domain_id=config.domain_id, io_timeout_s=config.io_timeout_s)
    observations: dict[str | tuple[str, ...], RosObservation] = {
        ("joint_position_left", "joint_position_right"): RosObservation(
            config.joint_feedback_topic,
            Jointfeedback,
            decode_arm_positions,
            qos["joint_feedback"],
        ),
    }
    for side, topic in config.gripper_feedback_topics.items():
        observations[f"gripper_feedback_{side}"] = RosObservation(
            topic, Float32MultiArray, decode_gripper_feedback, qos["gripper_feedback"]
        )
    for side, topic in config.eef_topics.items():
        observations[f"eef_{side}"] = RosObservation(
            topic, PoseStamped, decode_pose, qos["eef"]
        )
    for side, topic in config.wrench_topics.items():
        observations[f"wrench_{side}"] = RosObservation(
            topic, WrenchStamped, decode_wrench, qos["wrench"]
        )
    actions: dict[str | tuple[str, ...], RosCommand] = {}
    for side in SIDES:
        actions[f"joint_position_{side}"] = RosCommand(
            config.joint_command_topics[side],
            JointcmdArm,
            partial(encode_arm, message_type=JointcmdArm),
            qos["joint_command"],
            stamp=True,
        )
        actions[f"gripper_{side}"] = RosCommand(
            config.gripper_command_topics[side],
            Float32,
            partial(encode_gripper, message_type=Float32),
            qos["gripper_command"],
        )
    controller = TianjiController(config)
    controller.backend = Ros2Backend(
        observations=observations,
        actions=actions,
        on_stop=controller.on_stop,
        node_name=config.node_name,
        # Stop makes one set_input(0) call bounded by service_timeout_s.
        stop_timeout_s=config.service_timeout_s + SERVICE_POLL_MARGIN_S,
        **ros,
    )
    cameras = image_backends(
        config.camera_topics,
        message_type=Image,
        decode=lambda message: resize_with_pad(decode_nv12_image(message), image_size),
        node_name=config.node_name,
        qos=qos["camera"],
        **ros,
    )
    schema = observation_schema(config, image_size)
    robot = CompositeRobot(
        observation_schema=schema,
        action_schema=ACTION_SCHEMA,
        backends=[*cameras, controller.backend],
        action_guard=check_gripper_range,
    )
    return Embodiment(robot, make_adapter(config, schema), controller.activate)
