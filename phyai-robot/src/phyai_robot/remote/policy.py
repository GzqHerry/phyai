"""Synchronous model RPC with a resident predictor and leased run ownership."""

from __future__ import annotations

import math
import time
import uuid
import logging
from typing import Any, Protocol
from threading import Lock, Event
from collections.abc import Mapping, Callable

from numpy.typing import NDArray

from ._wire import (
    MAX_BYTES,
    PolicySpec,
    pack,
    unpack,
    read_values,
    values_payload,
)

logger = logging.getLogger(__name__)

Arrays = Mapping[str, NDArray[Any]]


class RemotePolicyError(RuntimeError):
    """The server rejected a request or its reply did not match this run."""


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


class RemotePolicy:
    """A Policy whose request and result are named model arrays.

    Call info/begin_run before enabling hardware, then warmup exactly once and
    discard its result. RobotDeployment may call predict on its worker thread.
    Every RPC owns and closes its socket on the calling thread; a lock
    serializes calls. Requests are never retried: a failed begin or prediction
    invalidates this client. The server releases an abandoned run after its
    lease expires.
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        *,
        timeout_s: float = 2.0,
        warmup_timeout_s: float = 120.0,
    ) -> None:
        _positive("timeout_s", timeout_s)
        _positive("warmup_timeout_s", warmup_timeout_s)
        self.endpoint = endpoint
        self.timeout_s, self.warmup_timeout_s = timeout_s, warmup_timeout_s
        # close() never waits for a slow request. The closing event transfers
        # end-run responsibility to the thread that owns the lock.
        self._lock, self._closing = Lock(), Event()
        self._run_id: str | None = None
        self._request_id = 0
        self._spec: PolicySpec | None = None
        self._failed = False
        self._warmed = False

    @property
    def spec(self) -> PolicySpec | None:
        """The spec returned by begin_run, or None before a run starts."""
        return self._spec

    def _rpc(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        arrays: Any = (),
        *,
        timeout_s: float | None = None,
    ) -> tuple[dict[str, Any], list[NDArray[Any]]]:
        """Perform one exchange while the caller owns the client lock.

        One deadline covers connection readiness, send and response wait. There
        is no replay after a timeout: the server may already have advanced its
        RNG even when the reply never arrives.
        """
        import zmq

        self._request_id += 1
        request_id, run_id = self._request_id, self._run_id
        frames = pack(
            dict(
                payload or {}, operation=operation, request_id=request_id, run_id=run_id
            ),
            arrays,
        )
        deadline = time.monotonic() + (
            self.timeout_s if timeout_s is None else timeout_s
        )
        # No socket can outlive its caller or cross the warmup/worker seam.
        with zmq.Context() as context:
            with context.socket(zmq.REQ) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                # Do not queue a request while no server is connected.
                socket.setsockopt(zmq.IMMEDIATE, 1)
                socket.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES)
                socket.connect(self.endpoint)

                def wait(event: int) -> None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not socket.poll(
                        max(1, math.ceil(remaining * 1000)), event
                    ):
                        raise TimeoutError(
                            f"policy {operation} timed out; request was not retried"
                        )

                wait(zmq.POLLOUT)
                socket.send_multipart(frames, flags=zmq.DONTWAIT, copy=False)
                wait(zmq.POLLIN)
                response, values = unpack(socket.recv_multipart(flags=zmq.DONTWAIT))
        if response.get("request_id") != request_id:
            raise RemotePolicyError("policy response request ID mismatch")
        if operation != "begin" and response.get("run_id") != run_id:
            raise RemotePolicyError("policy response run ID mismatch")
        if not response.get("ok"):
            raise RemotePolicyError(response.get("error", "policy request failed"))
        return response, values

    def _ensure_open(self) -> None:
        if self._closing.is_set() or self._failed:
            raise RuntimeError("RemotePolicy is closed or failed; create a new client")

    def info(self) -> PolicySpec:
        """Read the model contract without reserving or resetting a run."""
        with self._lock:
            self._ensure_open()
            result, _ = self._rpc("info")
            return PolicySpec.from_dict(result["spec"])

    def begin_run(self) -> PolicySpec:
        """Reserve the server and reset its noise stream.

        A busy server is an error; this never takes ownership from another
        live client. Use a new RemotePolicy for every robot run.
        """
        with self._lock:
            self._ensure_open()
            if self._run_id is not None:
                raise RuntimeError("this client already owns a run")
            try:
                response, _ = self._rpc("begin")
                run_id = response.get("run_id")
                if not isinstance(run_id, str) or not run_id:
                    raise RemotePolicyError("missing server run ID")
                self._run_id = run_id
                self._spec = PolicySpec.from_dict(response["spec"])
                return self._spec
            except BaseException:
                self._failed = True
                raise

    def warmup(self, inputs: Arrays) -> dict[str, NDArray[Any]]:
        """Consume the run's first prediction; the caller must not execute it."""
        return self._predict(inputs, warmup=True)

    def predict(self, request: Arrays) -> dict[str, NDArray[Any]]:
        """Return the model outputs after this run completed its warmup."""
        return self._predict(request, warmup=False)

    def _predict(self, inputs: Arrays, *, warmup: bool) -> dict[str, NDArray[Any]]:
        with self._lock:
            self._ensure_open()
            if self._run_id is None or self._spec is None:
                raise RuntimeError("begin_run is required before prediction")
            if warmup == self._warmed:
                raise RuntimeError("warmup is required exactly once before predict")
            try:
                meta, arrays = values_payload(inputs, self._spec.inputs)
                response, values = self._rpc(
                    "warmup" if warmup else "predict",
                    meta,
                    arrays,
                    timeout_s=self.warmup_timeout_s if warmup else self.timeout_s,
                )
                outputs = read_values(response, values, self._spec.outputs)
                self._warmed = True
                return outputs
            except BaseException:
                self._failed = True
                raise
            finally:
                if self._closing.is_set():
                    self._shutdown()

    def _shutdown(self) -> None:
        """Release the run once; the caller holds the lock.

        Release is best effort: if the server is unreachable or already dropped
        the run, its lease frees the server, so close() only logs the failure.
        """
        if self._run_id is not None:
            try:
                self._rpc("end")
            except Exception as error:  # noqa: BLE001 - the lease covers a failed release
                logger.warning("Could not release run %s: %s", self._run_id, error)
            finally:
                self._run_id = None

    def close(self) -> None:
        """Call after robot stop/close; never destroys an in-flight request.

        If prediction is still running, its thread releases the run. A failed
        release leaves the run to the server's lease timeout.
        """
        self._closing.set()
        if self._lock.acquire(blocking=False):
            try:
                self._shutdown()
            finally:
                self._lock.release()


class ModelPolicy(Protocol):
    """Server-side predictor from named model inputs to named model outputs."""

    def predict(self, inputs: Arrays) -> Arrays: ...


class PolicyServer:
    """Serve an existing predictor to one robot run at a time.

    reset_run resets the predictor's noise stream, not its engine. The
    application owns the predictor; this server owns only its ZMQ resources.
    A run that sends no request for lease_timeout_s is released, so a crashed
    or disconnected client cannot keep the model busy. Keep lease_timeout_s
    larger than the client's longest gap between requests.
    Bind loopback unless a trusted remote robot explicitly needs network access.
    """

    def __init__(
        self,
        *,
        policy: ModelPolicy,
        spec: PolicySpec,
        reset_run: Callable[[], None],
        endpoint: str = "tcp://127.0.0.1:5555",
        lease_timeout_s: float = 60.0,
    ) -> None:
        spec.to_dict()
        _positive("lease_timeout_s", lease_timeout_s)
        self.policy, self.spec = policy, spec
        self.reset_run, self.endpoint = reset_run, endpoint
        self.lease_timeout_s = lease_timeout_s
        self._run_id: str | None = None
        self._last_request_id = -1
        self._warmed = False
        self._lease_deadline = 0.0

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def expire_lease(self, now: float | None = None) -> bool:
        """Release the active run if its lease has expired; return True if so."""
        now = time.monotonic() if now is None else now
        if self._run_id is None or now < self._lease_deadline:
            return False
        logger.warning(
            "Releasing run %s after %.1f s without requests",
            self._run_id,
            self.lease_timeout_s,
        )
        self._run_id = None
        return True

    def _dispatch(
        self, request: Mapping[str, Any], arrays: list[NDArray[Any]]
    ) -> tuple[dict[str, Any], list[NDArray[Any]], str | None]:
        """Advance idle -> begun -> warmed -> predictions -> idle, synchronously.

        info never changes ownership. Every operation after begin must carry
        that run's ID and a newer request ID; replay cannot consume more noise.
        Failures leave the run owned until end or lease expiry.
        """
        op, request_id = request["operation"], request["request_id"]
        if type(request_id) is not int or request_id < 1:
            raise ValueError("invalid request ID")
        if op == "info":
            return dict(spec=self.spec.to_dict()), [], request.get("run_id")
        self.expire_lease()
        if op == "begin":
            if self._run_id is not None:
                raise RuntimeError(
                    "server already has an active run; wait for it to end or for "
                    f"its {self.lease_timeout_s:g} s lease to expire"
                )
            self.reset_run()
            self._run_id, self._warmed = uuid.uuid4().hex, False
            self._last_request_id = request_id
            return (
                dict(spec=self.spec.to_dict(), lease_timeout_s=self.lease_timeout_s),
                [],
                self._run_id,
            )
        if self._run_id is None or request.get("run_id") != self._run_id:
            raise ValueError("stale, expired or unknown run ID")
        if request_id <= self._last_request_id:
            raise ValueError("stale request ID")
        self._last_request_id = request_id
        if op == "end":
            run_id, self._run_id = self._run_id, None
            return {}, [], run_id
        if op not in ("warmup", "predict"):
            raise ValueError("unknown policy operation")
        if (op == "warmup") == self._warmed:
            raise ValueError("warmup is required exactly once before predict")
        inputs = read_values(request, arrays, self.spec.inputs)
        payload, values = values_payload(self.policy.predict(inputs), self.spec.outputs)
        self._warmed = True
        return payload, values, self._run_id

    def serve(
        self, *, stop_event: Event | None = None, ready_event: Event | None = None
    ) -> None:
        """Own a REP socket on this calling thread until Ctrl-C or stop_event.

        ready_event signals a successful bind, not a model warmup. Every
        consumed request gets one reply, including validation/model errors, so
        the REP socket can accept another request without changing ownership.
        The lease is renewed after each reply, so long inference never counts
        against it.
        """
        import zmq

        stop = Event() if stop_event is None else stop_event
        with zmq.Context() as context:
            with context.socket(zmq.REP) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES)
                socket.bind(self.endpoint)
                if ready_event is not None:
                    ready_event.set()
                while not stop.is_set():
                    if not socket.poll(100, zmq.POLLIN):
                        self.expire_lease()
                        continue
                    request: dict[str, Any] = {}
                    run_id = None
                    try:
                        request, arrays = unpack(socket.recv_multipart())
                        meta, values, run_id = self._dispatch(request, arrays)
                        reply = pack(
                            dict(
                                meta,
                                ok=True,
                                request_id=request["request_id"],
                                run_id=run_id,
                            ),
                            values,
                        )
                    except Exception as error:
                        reply = pack(
                            dict(
                                ok=False,
                                error=f"{type(error).__name__}: {error}",
                                request_id=request.get("request_id"),
                                run_id=request.get("run_id"),
                            )
                        )
                    socket.send_multipart(reply, copy=False)
                    # Only successful requests from the owner renew its lease.
                    if run_id is not None and run_id == self._run_id:
                        self._lease_deadline = time.monotonic() + self.lease_timeout_s
