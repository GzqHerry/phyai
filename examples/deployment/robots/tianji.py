"""Tianji (Marvin Pro) robot: ROS bindings, model adapter and controller enable.

Data flow: ROS message -> decoder -> cached Sample -> Observation -> adapter ->
model state/images; model action rows -> adapter -> complete Action -> one
publication per arm and gripper. RobotDeployment owns timing and interpolation.

Cameras can come from an H.264 preview mosaic or individual NV12 topics.
The configured tile size can reproduce recording geometry before letterboxing
images to the model input size.

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
from phyai_robot.images import resize, resize_with_pad
from phyai_robot.remote import PolicySpec, Embodiment
from phyai_robot.backends.ros2 import (
    RosCommand,
    Ros2Backend,
    RosObservation,
    image_backends,
)
from phyai_robot.backends.ros2.video import H264Decoder, MosaicDecoder
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
CAMERA_SOURCES = ("mosaic", "raw")
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
    # Gripper feedback [lower, upper] in motor radians; maps model state to [0, 1].
    gripper_limits_rad: list[float] = field(default_factory=lambda: [0.0, 1.6])
    # "mosaic": decode the H.264 preview mosaic; "raw": per-camera NV12 images.
    camera_source: str = "mosaic"
    # Each camera image is resized to this [height, width] before letterboxing,
    # matching training geometry when specified; null keeps its aspect ratio.
    camera_size: list[int] | None = None
    # Alias for camera_size; accepted for compatibility with converted model configs.
    camera_resize_hw: list[int] | None = None
    # H.264 mosaic of the undistorted cameras and each field's [row, column].
    mosaic_topic: str = "/quad_tile/compressed_undistorted"
    mosaic_grid: list[int] = field(default_factory=lambda: [2, 2])
    mosaic_tiles: dict[str, list[int]] = field(
        default_factory=lambda: {
            "head_camera": [0, 0],
            "left_wrist_camera": [1, 0],
            "right_wrist_camera": [1, 1],
        }
    )
    # Converted mosaic frames per second; every access unit is still decoded.
    mosaic_max_rate_hz: float = 15.0
    # Observation field -> undistorted NV12 topic, used by camera_source: raw.
    camera_topics: dict[str, str] = field(
        default_factory=lambda: {
            "head_camera": "/camera/left_eye/image_nv12_undistorted",
            "left_wrist_camera": "/camera/left_wrist/image_nv12_undistorted",
            "right_wrist_camera": "/camera/right_wrist/image_nv12_undistorted",
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
        if self.camera_source not in CAMERA_SOURCES:
            raise ValueError(f"robot.camera_source must be one of {CAMERA_SOURCES}")
        if self.camera_size is None and self.camera_resize_hw is not None:
            self.camera_size = self.camera_resize_hw
        if self.camera_size is not None and (
            len(self.camera_size) != 2 or min(self.camera_size) < 1
        ):
            raise ValueError("robot.camera_size must be [height, width] or null")
        if self.camera_resize_hw is not None and (
            len(self.camera_resize_hw) != 2 or min(self.camera_resize_hw) < 1
        ):
            raise ValueError("robot.camera_resize_hw must be [height, width] or null")
        if len(self.mosaic_grid) != 2 or min(self.mosaic_grid) < 1:
            raise ValueError("robot.mosaic_grid must be [rows, columns]")
        rows, columns = self.mosaic_grid
        tiles = [tuple(tile) for tile in self.mosaic_tiles.values()]
        if len(set(tiles)) != len(tiles) or any(
            len(tile) != 2 or not (0 <= tile[0] < rows and 0 <= tile[1] < columns)
            for tile in tiles
        ):
            raise ValueError("robot.mosaic_tiles must name distinct grid cells")
        if not self.mosaic_max_rate_hz > 0:
            raise ValueError("robot.mosaic_max_rate_hz must be positive")
        if not set(self.model_cameras.values()) <= set(camera_fields(self)):
            raise ValueError(
                f"robot.model_cameras must name {self.camera_source} camera fields"
            )
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


def camera_fields(config: TianjiConfig) -> list[str]:
    """Observation fields provided by the selected camera source."""
    if config.camera_source == "mosaic":
        return list(config.mosaic_tiles)
    return list(config.camera_topics)


def camera_transform(config: TianjiConfig, image_size: int):
    """Reshape one full camera image like the recorded training tiles."""
    size = config.camera_size

    def transform(image: np.ndarray) -> np.ndarray:
        if size is not None:
            image = resize(image, int(size[0]), int(size[1]))
        return resize_with_pad(image, image_size)

    return transform


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
        for key in camera_fields(config)
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
    from sensor_msgs.msg import Image, CompressedImage
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
    transform = camera_transform(config, image_size)
    if config.camera_source == "mosaic":
        # One worker decodes the whole stream; the tiles share a frame and time.
        mosaic = MosaicDecoder(
            tiles={key: tuple(tile) for key, tile in config.mosaic_tiles.items()},
            grid=tuple(config.mosaic_grid),
            transform=transform,
            decoder=H264Decoder(max_rate_hz=config.mosaic_max_rate_hz),
        )
        cameras = [
            Ros2Backend(
                observations={
                    mosaic.fields: RosObservation(
                        config.mosaic_topic, CompressedImage, mosaic, qos["camera"]
                    )
                },
                actions={},
                node_name=f"{config.node_name}_cameras",
                **ros,
            )
        ]
    else:
        cameras = image_backends(
            config.camera_topics,
            message_type=Image,
            decode=lambda message: transform(decode_nv12_image(message)),
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
