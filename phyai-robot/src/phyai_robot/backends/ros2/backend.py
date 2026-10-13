"""ROS2 topic observations and commands with application-defined codecs.

ROS2, message packages, and a Python-compatible rclpy installation are supplied
by the host. Topic commands describe submission, not motion completion. Devices
requiring a specialized action/service protocol can implement Backend directly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
from collections.abc import Mapping, Callable

from numpy.typing import NDArray

from ...types import Action, Sample
from ..._compat import BaseExceptionGroup
from .._threaded import _ThreadedBackend

if TYPE_CHECKING:
    # Type-only imports keep this module importable without a ROS installation.
    from rclpy.qos import QoSProfile
    from rclpy.node import Node
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.publisher import Publisher


@dataclass(frozen=True)
class RosObservation:
    """Decode one topic into one field or a tuple-keyed group of fields.

    String keys produce one array. Tuple keys produce a mapping with exactly
    those names; all fields from that message share one receive timestamp. QoS
    is explicit because sensor publishers may use best-effort delivery.
    """

    topic: str  # Full subscription topic, not a field name inferred by the backend.
    message_type: type[Any]  # Actual ROS message class supplied by the application.
    # A string key decodes to an array; a tuple key decodes to those named fields.
    decode: Callable[[Any], NDArray[Any] | Mapping[str, NDArray[Any]]]
    qos: int | QoSProfile = (
        10  # Explicit QoSProfile or depth; match the actual publisher.
    )
    recover_on_next_message: bool = False
    # If enabled, read() reports a bad sample until this topic decodes successfully.


@dataclass(frozen=True)
class RosCommand:
    """Encode an action field/group into one topic message per complete write.

    stamp=True assigns header.stamp on the worker just before publication using
    its node clock. Leave it False when the encoder owns timestamp semantics.
    """

    topic: str  # Actual controller command topic.
    message_type: type[Any]  # Message class accepted by that controller.
    encode: Callable[[NDArray[Any] | Mapping[str, NDArray[Any]]], Any]
    qos: int | QoSProfile = 10  # Explicit publisher QoS or depth.
    stamp: bool = False  # Set header.stamp from the node clock before publishing.


def _field_keys(bindings: Mapping[str | tuple[str, ...], Any]) -> list[str]:
    """Flatten message groups into Backend ownership without overlapping fields."""
    keys: list[str] = []
    for key in bindings:
        fields = (key,) if isinstance(key, str) else key
        if not isinstance(fields, tuple) or not fields:
            raise ValueError("ROS binding keys must be strings or nonempty tuples")
        if any(not isinstance(field, str) or not field for field in fields):
            raise ValueError("ROS fields must be nonempty strings")
        keys.extend(fields)
    if len(keys) != len(set(keys)):
        raise ValueError("ROS bindings must not overlap fields")
    return keys


class Ros2Backend(_ThreadedBackend):
    """Own ROS resources on one worker; expose cached reads and bounded writes.

    Construction only records bindings. connect() creates the context, node,
    subscriptions and publishers on the worker. The inherited request queue
    submits each whole action there, while read() returns copied cached samples.
    A tuple binding can combine joints and a gripper into one JointState without
    adding a robot-specific backend or changing the CompositeRobot schema.

    For command backends, on_stop is the application's explicit device behavior.
    The transport stops accepting writes separately; a no-op hook can therefore
    mean stop publication without commanding a hold, home or motor-disable.
    """

    def __init__(
        self,
        *,
        observations: Mapping[str | tuple[str, ...], RosObservation],
        actions: Mapping[str | tuple[str, ...], RosCommand],
        on_stop: Callable[[Node], None] | None = None,
        # Called with our node on its worker thread; mandatory for command backends.
        node_name: str = "phyai_robot",  # Node owned exclusively by this backend.
        io_timeout_s: float = 0.1,  # Bound for publishing/stop task completion.
        connect_timeout_s: float = 5.0,  # Context/node/executor initialization budget.
        domain_id: int | None = None,  # None uses the native ROS environment.
    ) -> None:
        if actions and on_stop is None:
            raise ValueError("ROS command backends require a device-specific on_stop")
        super().__init__(
            _field_keys(observations),
            _field_keys(actions),
            io_timeout_s,
            connect_timeout_s,
        )
        self._observations = dict(observations)
        self._actions = dict(actions)
        self._grouped_actions = any(isinstance(key, tuple) for key in actions)
        self._decode_errors: dict[str | tuple[str, ...], str] = {}
        self._domain_id = domain_id
        self._on_stop: Callable[[Node], None] | None = on_stop
        self._node_name: str = node_name
        self._context: Context | None = (
            None  # Private ROS context; never shut down another user's context.
        )
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._publishers: dict[
            str | tuple[str, ...], Publisher[Any]
        ] = {}  # Field -> publisher, accessed only on the worker.

    def _open(self) -> None:
        try:
            from rclpy.node import Node
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
        except ModuleNotFoundError as error:
            if error.name and (
                error.name == "rclpy" or error.name.startswith("rclpy.")
            ):
                raise ImportError(
                    "Ros2Backend needs a sourced ROS2 installation with rclpy "
                    "and message packages matching this Python version; "
                    "the ros2 extra does not install ROS middleware"
                ) from error
            raise
        context: Context = Context()
        self._context = context
        context.init(
            args=[],
            **({"domain_id": self._domain_id} if self._domain_id is not None else {}),
        )
        node: Node = Node(self._node_name, context=context, use_global_arguments=False)
        self._node = node
        executor: SingleThreadedExecutor = SingleThreadedExecutor(context=context)
        self._executor = executor
        executor.add_node(node)
        for key, spec in self._observations.items():

            def receive(
                message: Any,
                key: str | tuple[str, ...] = key,
                spec: RosObservation = spec,
            ) -> None:
                # Message classes vary by topic; the application owns each codec.
                received_at_ns: int = time.monotonic_ns()
                try:
                    decoded = spec.decode(message)
                    if isinstance(key, tuple):
                        if not isinstance(decoded, Mapping) or set(decoded) != set(key):
                            raise ValueError(
                                "ROS decoder fields do not match its binding"
                            )
                        values = decoded
                    else:
                        values = {key: decoded}
                    self._cache(values, received_at_ns)
                except Exception as error:
                    if not spec.recover_on_next_message:
                        raise
                    with self._lock:
                        self._decode_errors[key] = str(error)
                else:
                    with self._lock:
                        self._decode_errors.pop(key, None)

            node.create_subscription(spec.message_type, spec.topic, receive, spec.qos)
        for key, spec in self._actions.items():
            self._publishers[key] = node.create_publisher(
                spec.message_type, spec.topic, spec.qos
            )

    def _poll(self) -> None:
        # A short spin lets the same thread service command tasks promptly.
        # Unrecoverable callback exceptions become observable worker faults.
        assert self._executor is not None  # _open completed on this worker.
        self._executor.spin_once(timeout_sec=0.001)

    def read(self) -> Mapping[str, Sample]:
        """Report a recoverable decode error until that binding receives valid data.

        This does not reconnect or retry a command. Other topics cannot clear
        the error, and the inherited worker/connection failures still propagate.
        """
        samples = super().read()
        with self._lock:
            if self._decode_errors:
                raise RuntimeError(f"ROS sensor error: {self._decode_errors}")
        return samples

    def _write_device(self, action: Action, deadline: float) -> None:
        # Grouped bindings use declaration order, independent of field-set order.
        # Single-field callers retain their original publication order.
        keys = self._actions if self._grouped_actions else action
        messages = {
            key: self._actions[key].encode(
                {field: action[field] for field in key}
                if isinstance(key, tuple)
                else action[key]
            )
            for key in keys
        }
        # Encode every field before publishing any of them. Publishing two topics
        # still is not atomic and does not guarantee simultaneous physical motion.
        for key, message in messages.items():
            if not isinstance(message, self._actions[key].message_type):
                raise TypeError(f"{key}: encoder returned the wrong ROS message type")
            if self._actions[key].stamp and not hasattr(
                getattr(message, "header", None), "stamp"
            ):
                raise TypeError(f"{key}: stamped commands require header.stamp")
        for key, message in messages.items():
            if time.monotonic() >= deadline:
                raise TimeoutError("ROS command task expired before publishing")
            if self._actions[key].stamp:
                message.header.stamp = self._node.get_clock().now().to_msg()
            self._publishers[key].publish(message)

    def _stop_device(self) -> None:
        if self._node is not None and self._on_stop is not None:
            # The hook must be bounded; it must not spin this same executor
            # recursively or wait indefinitely for a callback on this worker.
            self._on_stop(self._node)

    def _close_device(self) -> None:
        # Independent cleanup keeps a failed executor shutdown from leaking the
        # owned node/context. Do not call the global rclpy.shutdown().
        errors: list[BaseException] = []
        for cleanup in (
            lambda: (
                self._executor.shutdown(timeout_sec=self._io_timeout_s)
                if self._executor
                else None
            ),
            lambda: self._node.destroy_node() if self._node else None,
            lambda: self._context.try_shutdown() if self._context else None,
        ):
            try:
                cleanup()
            except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("ROS resource cleanup failed", errors)
