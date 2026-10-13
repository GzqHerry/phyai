"""ROS2 topic observations, topic commands and service calls on one worker.

ROS2, message packages, and a Python-compatible rclpy installation are supplied
by the host. Topic commands describe submission, not motion completion.
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
    from rclpy.node import Node
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.publisher import Publisher

# rclpy's Client.wait_for_service checks availability every 0.25 s.
SERVICE_POLL_MARGIN_S = 0.5

# A QoS is a depth, a QoSProfile, or one of these names (YAML-friendly):
#   "sensor_data"  rclpy's best-effort sensor profile
#   "reliable"     reliable, volatile, keep-last 10
#   "best_effort"  best-effort, volatile, keep-last 10
QoS = Any


def resolve_qos(qos: QoS) -> Any:
    """Turn a configured QoS into an rclpy QoS argument; call after ROS import."""
    if isinstance(qos, bool):
        raise TypeError("QoS must be a depth, a profile name or a QoSProfile")
    if isinstance(qos, int):
        if qos < 1:
            raise ValueError("QoS depth must be positive")
        return qos
    if not isinstance(qos, str):
        return qos
    from rclpy.qos import (
        QoSProfile,
        HistoryPolicy,
        DurabilityPolicy,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )

    if qos == "sensor_data":
        return qos_profile_sensor_data
    reliability = {
        "reliable": ReliabilityPolicy.RELIABLE,
        "best_effort": ReliabilityPolicy.BEST_EFFORT,
    }.get(qos)
    if reliability is None:
        raise ValueError(f"unknown QoS profile name: {qos!r}")
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


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
    qos: QoS = 10  # Must be compatible with the actual publisher.
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
    qos: QoS = 10  # Must be compatible with the controller's subscription.
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

    Construction only records bindings. connect() creates a private context,
    node, subscriptions and publishers on the worker. read() returns cached
    samples; each write publishes one message per binding on the worker.
    A tuple binding can combine several fields into one message, such as two
    arms in one feedback message, so they share a receive timestamp.

    call_service() runs a service request on the same worker, so device setup
    and stop hooks need no second ROS context or executor thread. For command
    backends, on_stop is the application's explicit device behavior: it runs on
    the worker with this backend and may call call_service(). A no-op hook means
    "stop publishing" without commanding a hold, home or motor-disable.
    """

    def __init__(
        self,
        *,
        observations: Mapping[str | tuple[str, ...], RosObservation],
        actions: Mapping[str | tuple[str, ...], RosCommand],
        on_stop: Callable[[Ros2Backend], None] | None = None,
        # Mandatory for command backends; bounded by stop_timeout_s.
        node_name: str = "phyai_robot",  # Node owned exclusively by this backend.
        io_timeout_s: float = 0.1,  # Bound for publishing one command.
        stop_timeout_s: float | None = None,  # Bound for on_stop; None = io.
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
            stop_timeout_s,
        )
        self._observations = dict(observations)
        self._actions = dict(actions)
        self._grouped_actions = any(isinstance(key, tuple) for key in actions)
        self._decode_errors: dict[str | tuple[str, ...], str] = {}
        self._domain_id = domain_id
        self._on_stop = on_stop
        self._node_name = node_name
        # Private ROS context; never shut down another user's context.
        self._context: Context | None = None
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        # Accessed only on the worker.
        self._publishers: dict[str | tuple[str, ...], Publisher[Any]] = {}
        self._service_clients: dict[str, Any] = {}

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

            node.create_subscription(
                spec.message_type, spec.topic, receive, resolve_qos(spec.qos)
            )
        for key, spec in self._actions.items():
            self._publishers[key] = node.create_publisher(
                spec.message_type, spec.topic, resolve_qos(spec.qos)
            )

    def _poll(self) -> None:
        # A short spin lets the same thread service requests promptly.
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

    def call_service(
        self, service_type: type[Any], name: str, request: Any, *, timeout_s: float
    ) -> Any:
        """Call a service on the worker with one end-to-end deadline.

        timeout_s covers waiting for the server and for its response. A timeout
        cannot undo a request the server may already have applied. A response
        with success == False raises RuntimeError with the server's message.
        """
        # rclpy polls service availability in 0.25 s steps; allow for that.
        return self.call(
            lambda: self._call_service(service_type, name, request, timeout_s),
            timeout_s + SERVICE_POLL_MARGIN_S,
        )

    def _call_service(
        self, service_type: type[Any], name: str, request: Any, timeout_s: float
    ) -> Any:
        assert self._node is not None and self._executor is not None
        deadline = time.monotonic() + timeout_s
        client = self._service_clients.get(name)
        if client is None:
            client = self._node.create_client(service_type, name)
            self._service_clients[name] = client
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise TimeoutError(f"ROS service {name} is not available")
        future = client.call_async(request)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            # Not re-entrant: requests run between spins, never inside a callback.
            self._executor.spin_until_future_complete(future, timeout_sec=remaining)
        if not future.done():
            client.remove_pending_request(future)
            raise TimeoutError(f"ROS service {name} did not respond in time")
        response = future.result()
        if response is None:
            raise RuntimeError(f"ROS service {name} returned no response")
        if getattr(response, "success", True) is False:
            message = getattr(response, "message", "")
            raise RuntimeError(f"ROS service {name} rejected the request: {message}")
        return response

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
            # Runs on this worker; call_service() executes inline here.
            self._on_stop(self)

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


def image_backends(
    topics: Mapping[str, str],
    *,
    message_type: type[Any],
    decode: Callable[[Any], NDArray[Any]],
    node_name: str,
    qos: QoS = "sensor_data",
    **options: Any,
) -> list[Ros2Backend]:
    """Build one observation-only backend per camera field.

    Each backend decodes on its own worker, so expensive image conversion for
    one camera never delays another camera or a command backend. options are
    passed to every Ros2Backend (for example domain_id and io_timeout_s).
    """
    return [
        Ros2Backend(
            observations={
                field: RosObservation(
                    topic, message_type, decode, qos, recover_on_next_message=True
                )
            },
            actions={},
            node_name=f"{node_name}_{field}",
            **options,
        )
        for field, topic in topics.items()
    ]
