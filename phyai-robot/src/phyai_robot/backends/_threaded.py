"""Private worker plumbing shared by the two asynchronous transports.

Socket/node ownership stays on one thread. Only frozen sensor snapshots and
bounded requests cross the thread boundary. This is not a public backend
framework; its hooks only remove identical lifecycle code.
"""

from __future__ import annotations

import math
import time
import queue
import threading
from types import MappingProxyType
from typing import Any, Literal
from _thread import LockType
from dataclasses import dataclass
from collections.abc import Mapping, Callable, Iterable
from concurrent.futures import Future, wait

import numpy as np
from numpy.typing import NDArray

from ..types import Action, Sample
from .._compat import BaseExceptionGroup
from ..validation import freeze, freeze_values

_Kind = Literal["write", "stop", "call"]


@dataclass(frozen=True)
class _Request:
    kind: _Kind  # One bounded request at a time; never a stream of commands.
    value: Any  # Detached Action for writes, a callable for calls, None for stop.
    deadline: float  # Local monotonic deadline, checked again on the I/O thread.
    future: Future[Any]  # Submission result; never interpreted as physical completion.


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


class _ThreadedBackend:
    def __init__(
        self,
        observation_keys: Iterable[str],  # Fixed sensor names owned by this session.
        action_keys: Iterable[str],  # Fixed command names owned by this session.
        io_timeout_s: float,  # Maximum wait for a submitted command.
        connect_timeout_s: float,  # Maximum wait for worker startup and shutdown.
        stop_timeout_s: float | None = None,  # Maximum wait for stop; None = io.
    ) -> None:
        stop_timeout_s = io_timeout_s if stop_timeout_s is None else stop_timeout_s
        for name, value in (
            ("io_timeout_s", io_timeout_s),
            ("connect_timeout_s", connect_timeout_s),
            ("stop_timeout_s", stop_timeout_s),
        ):
            _positive(name, value)
        self._observation_keys: frozenset[str] = frozenset(observation_keys)
        self._action_keys: frozenset[str] = frozenset(action_keys)
        if any(
            not isinstance(key, str) or not key
            for key in self._observation_keys | self._action_keys
        ):
            raise ValueError("backend keys must be nonempty strings")
        self._io_timeout_s = io_timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._stop_timeout_s = stop_timeout_s
        # Protect cached samples and the first worker error.
        self._lock: LockType = threading.Lock()
        self._samples: dict[str, Sample] = {}
        self._error: BaseException | None = None
        self._requests: queue.Queue[_Request] = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._stop_requested = False
        self._device_stopped = False  # Updated only by the worker after a stop.

    @property
    def observation_keys(self) -> frozenset[str]:
        return self._observation_keys

    @property
    def action_keys(self) -> frozenset[str]:
        return self._action_keys

    def _raise_error(self) -> None:
        with self._lock:
            error: BaseException | None = self._error
        if error is not None:
            raise RuntimeError("transport worker failed") from error

    def _record_error(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
            elif self._error is not error:
                self._error = BaseExceptionGroup(
                    "transport failures", [self._error, error]
                )

    def _cache(self, values: Mapping[str, NDArray[Any]], received_at_ns: int) -> None:
        if not set(values) <= self.observation_keys:
            raise ValueError("decoder returned undeclared observation fields")
        if any(not isinstance(value, np.ndarray) for value in values.values()):
            raise TypeError("decoders must return NumPy arrays")
        # Freeze once here; read(), CompositeRobot and deployment then share
        # these read-only arrays instead of copying them again.
        samples = {key: Sample(freeze(v), received_at_ns) for key, v in values.items()}
        with self._lock:
            self._samples.update(samples)

    def _on_worker(self) -> bool:
        return self._thread is not None and threading.current_thread() is self._thread

    def connect(self) -> None:
        if self._closed or self._stop_requested:
            raise RuntimeError("transport is closed or stopped")
        if self._thread is not None:
            self._raise_error()
            return
        self._thread = threading.Thread(
            target=self._run, name=type(self).__name__, daemon=True
        )
        self._thread.start()
        if not self._ready.wait(self._connect_timeout_s):
            error: TimeoutError = TimeoutError("transport initialization timed out")
            self._record_error(error)
            self._halt.set()
            raise error
        self._raise_error()

    def read(self) -> Mapping[str, Sample]:
        if self._thread is None or self._closed:
            raise RuntimeError("transport is not connected")
        self._raise_error()
        with self._lock:
            return MappingProxyType(dict(self._samples))

    def _submit(self, kind: _Kind, value: Any, timeout_s: float) -> Any:
        if self._thread is None or not self._thread.is_alive() or self._closed:
            self._raise_error()
            raise RuntimeError("transport worker is not running")
        if kind != "stop":
            self._raise_error()
        future: Future[Any] = Future()
        deadline: float = time.monotonic() + timeout_s
        try:
            self._requests.put_nowait(_Request(kind, value, deadline, future))
        except queue.Full:
            raise RuntimeError("a transport request is already pending") from None
        # Wait separately from result(): the operation itself may raise
        # TimeoutError, which is the same class as a future wait timeout.
        if not wait([future], timeout=max(0.0, deadline - time.monotonic())).done:
            # A pending request is cancelled. An already-running native operation
            # cannot be undone, so fault the session and stop it on the worker.
            future.cancel()
            error = TimeoutError(f"transport {kind} timed out")
            self._record_error(error)
            self._halt.set()
            raise error
        return future.result()

    def write(self, action: Action) -> None:
        if self._stop_requested:
            raise RuntimeError("transport has been stopped")
        if set(action) != self.action_keys:
            raise ValueError("transport command fields do not match")
        self._submit("write", freeze_values(action), self._io_timeout_s)

    def call(self, function: Callable[[], Any], timeout_s: float) -> Any:
        """Run function on the worker that owns the transport, and return its result.

        Use this for bounded device requests that are not part of a command, such
        as enabling an input mode. function should enforce its own deadline;
        the caller waits io_timeout_s longer so it can receive that error.
        """
        _positive("timeout_s", timeout_s)
        if self._on_worker():
            return function()
        if self._stop_requested:
            raise RuntimeError("transport has been stopped")
        return self._submit("call", function, timeout_s + self._io_timeout_s)

    def stop(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        if (
            self._thread is not None
            and self._thread.is_alive()
            and not self._halt.is_set()
        ):
            self._submit("stop", None, self._stop_timeout_s)

    def close(self) -> None:
        if self._closed:
            return
        self._halt.set()
        if self._thread is not None:
            self._thread.join(self._connect_timeout_s + self._stop_timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("transport worker did not release its resources")
        self._closed = True
        self._raise_error()

    def _execute(self, request: _Request) -> Any:
        if time.monotonic() >= request.deadline:
            raise TimeoutError("transport request expired before execution")
        if request.kind == "stop":
            self._stop_device()
            self._device_stopped = True
            return None
        if request.kind == "call":
            return request.value()
        self._write_device(request.value, request.deadline)
        return None

    def _run(self) -> None:
        request: _Request | None
        try:
            self._open()
            self._ready.set()
            while not self._halt.is_set():
                try:
                    request = self._requests.get_nowait()
                except queue.Empty:
                    request = None
                if (
                    request is not None
                    and request.future.set_running_or_notify_cancel()
                ):
                    try:
                        request.future.set_result(self._execute(request))
                    except BaseException as error:
                        request.future.set_exception(error)
                        if request.kind != "call":
                            raise
                self._poll()
        except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
            self._record_error(error)
        finally:
            self._ready.set()
            # Even decoder failures stop the device. Cleanup still runs if stop
            # itself fails. Hooks tolerate resources that were only partly opened.
            for cleanup in (
                self._stop_device if not self._device_stopped else lambda: None,
                self._close_device,
            ):
                try:
                    cleanup()
                except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
                    self._record_error(error)
            while True:
                try:
                    pending: _Request = self._requests.get_nowait()
                except queue.Empty:
                    break
                if pending.future.set_running_or_notify_cancel():
                    pending.future.set_exception(
                        RuntimeError("transport worker stopped")
                    )

    def _open(self) -> None:
        raise NotImplementedError

    def _poll(self) -> None:
        raise NotImplementedError

    def _write_device(self, action: Action, deadline: float) -> None:
        raise NotImplementedError

    def _stop_device(self) -> None:
        raise NotImplementedError

    def _close_device(self) -> None:
        raise NotImplementedError
