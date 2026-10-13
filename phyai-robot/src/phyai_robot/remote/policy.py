"""Synchronous policy RPC with a resident predictor and explicit run ownership."""

from __future__ import annotations

import math
import time
import uuid
from threading import Lock, Event
from collections.abc import Callable

from ._wire import (
    MAX_BYTES,
    PolicySpec,
    pack,
    unpack,
    read_chunk,
    chunk_payload,
    read_observation,
    observation_payload,
)


class RemotePolicyError(RuntimeError):
    """The server rejected a request or its reply did not match this run."""


class RemotePolicy:
    """An Observation -> ActionChunk policy; no model or robot SDK is imported.

    Call info/begin_run before opening hardware, then warmup exactly once and
    discard its result. RobotDeployment can call predict on its worker thread.
    Every RPC owns and closes its socket/context on the calling thread. Requests
    are not retried. A failed begin or prediction invalidates this client; close
    only attempts to release its run after hardware has stopped. info() is a
    read-only query and does not reserve a run.
    """

    def __init__(
        self, endpoint="tcp://127.0.0.1:5555", *, timeout_s=2.0, warmup_timeout_s=120.0
    ):
        for value in (timeout_s, warmup_timeout_s):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("RPC timeouts must be finite and positive")
        self.endpoint = endpoint
        self.timeout_s, self.warmup_timeout_s = timeout_s, warmup_timeout_s
        # Serialize info/begin/warmup/predict without making close wait on a
        # potentially slow GPU request. The closing event transfers end-run
        # responsibility to that request's owner when the lock is occupied.
        self._lock, self._closing = Lock(), Event()
        self._run_id = None
        self._request_id = 0
        self._spec = None
        self._failed = False
        self._warmed = False

    def _rpc(self, operation, payload=None, arrays=(), *, timeout_s=None):
        """Perform one exchange while the caller owns the client lock.

        One deadline covers connection readiness, send and response wait. There
        is no replay after a timeout: the server may already have advanced its
        RNG even when the reply never arrives. begin is the only response that
        creates a run ID instead of echoing the client's existing one.
        """
        import zmq

        timeout = self.timeout_s if timeout_s is None else timeout_s
        self._request_id += 1
        request_id, run_id = self._request_id, self._run_id
        frames = pack(
            dict(
                payload or {}, operation=operation, request_id=request_id, run_id=run_id
            ),
            arrays,
        )
        deadline = time.monotonic() + timeout
        # No socket can outlive its caller or cross the warmup/worker seam.
        with zmq.Context() as context:
            with context.socket(zmq.REQ) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.IMMEDIATE, 1)
                socket.setsockopt(zmq.MAXMSGSIZE, MAX_BYTES)
                socket.connect(self.endpoint)

                def wait(event):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not socket.poll(
                        max(1, math.ceil(remaining * 1000)), event
                    ):
                        raise TimeoutError(
                            f"policy {operation} timed out; request was not retried"
                        )

                wait(zmq.POLLOUT)
                socket.send_multipart(frames, flags=zmq.DONTWAIT)
                wait(zmq.POLLIN)
                response, values = unpack(socket.recv_multipart(flags=zmq.DONTWAIT))
        if response.get("request_id") != request_id:
            raise RemotePolicyError("policy response request ID mismatch")
        if operation != "begin" and response.get("run_id") != run_id:
            raise RemotePolicyError("policy response run ID mismatch")
        if not response.get("ok"):
            raise RemotePolicyError(response.get("error", "policy request failed"))
        return response, values

    def _ensure_open(self):
        if self._closing.is_set() or self._failed:
            raise RuntimeError("RemotePolicy is closed or failed; create a new client")

    def info(self) -> PolicySpec:
        """Read model/schema metadata without reserving or resetting a run."""
        with self._lock:
            self._ensure_open()
            result, _ = self._rpc("info")
            return PolicySpec.from_dict(result["spec"])

    def begin_run(self, *, embodiment, observation_schema, action_schema) -> PolicySpec:
        """Reserve this server, check the physical contract and reset its RNG.

        Call before robot.connect(). A busy server is an error; this method never
        takes ownership from another client. Use a new RemotePolicy per robot run.
        """
        with self._lock:
            self._ensure_open()
            if self._run_id is not None:
                raise RuntimeError("this client already owns a run")
            contract = PolicySpec(embodiment, observation_schema, action_schema)
            try:
                response, _ = self._rpc("begin", dict(spec=contract.to_dict()))
                self._run_id = response["run_id"]
                if not isinstance(self._run_id, str) or not self._run_id:
                    raise RemotePolicyError("missing server run ID")
                self._spec = PolicySpec.from_dict(response["spec"])
                self._spec.check(embodiment, observation_schema, action_schema)
                return self._spec
            except BaseException:
                self._failed = True
                raise

    def warmup(self, observation):
        """Consume the first prediction. The caller must discard this chunk."""
        return self._predict(observation, warmup=True)

    def predict(self, observation):
        """Return a physical ActionChunk after this run completed its warmup."""
        return self._predict(observation, warmup=False)

    def _predict(self, observation, *, warmup):
        """Share framing, lifecycle checks and failure handling for both calls."""
        with self._lock:
            self._ensure_open()
            if self._run_id is None or self._spec is None:
                raise RuntimeError("begin_run is required before prediction")
            if warmup == self._warmed:
                raise RuntimeError("warmup is required exactly once before predict")
            try:
                meta, arrays = observation_payload(observation)
                response, values = self._rpc(
                    "warmup" if warmup else "predict",
                    meta,
                    arrays,
                    timeout_s=self.warmup_timeout_s if warmup else self.timeout_s,
                )
                chunk = read_chunk(response, values, self._spec.action_schema)
                self._warmed = True
                return chunk
            except BaseException:
                self._failed = True
                raise
            finally:
                if self._closing.is_set():
                    self._end_run()

    def _end_run(self):
        """Release once under the client lock; forget ownership even if RPC fails."""
        if self._run_id is not None:
            try:
                self._rpc("end")
            finally:
                self._run_id = None

    def close(self):
        """Call after robot stop/close; never destroy an in-flight RPC socket.

        If prediction is still unwinding, its owning thread releases the run.
        A failed release requires a manual server restart, not motion takeover.
        """
        self._closing.set()
        if self._lock.acquire(blocking=False):
            try:
                self._end_run()
            finally:
                self._lock.release()


class PolicyServer:
    """Serve an existing policy/adapter pair without owning model construction.

    One run at a time. reset_run resets the policy's RNG, not its engine. The
    application owns policy.close; this server owns only its ZMQ resources.
    Bind loopback unless a trusted remote robot explicitly needs network access.
    """

    def __init__(
        self,
        *,
        policy,
        adapter,
        spec: PolicySpec,
        reset_run: Callable[[], None],
        endpoint="tcp://127.0.0.1:5555",
    ):
        spec.to_dict()
        self.policy, self.adapter, self.spec = policy, adapter, spec
        self.reset_run, self.endpoint = reset_run, endpoint
        self._run_id = None
        self._last_request_id = -1
        self._warmed = False

    def _dispatch(self, request, arrays):
        """Advance idle -> begun -> warmed -> predictions -> idle, synchronously.

        info never changes ownership. Every operation after begin must carry
        that run's ID and a newer request ID; replay cannot consume more noise.
        Failures leave the run owned until end or a manual server restart.
        """
        op, request_id = request["operation"], request["request_id"]
        if type(request_id) is not int or request_id < 1:
            raise ValueError("invalid request ID")
        if op == "info":
            return dict(spec=self.spec.to_dict()), [], request.get("run_id")
        if op == "begin":
            if self._run_id is not None:
                raise RuntimeError(
                    "server already has an active run; end it or restart the server"
                )
            contract = PolicySpec.from_dict(request["spec"])
            self.spec.check(
                contract.embodiment, contract.observation_schema, contract.action_schema
            )
            self.reset_run()
            self._run_id, self._warmed = uuid.uuid4().hex, False
            self._last_request_id = request_id
            return dict(spec=self.spec.to_dict()), [], self._run_id
        if self._run_id is None or request.get("run_id") != self._run_id:
            raise ValueError("stale or unknown run ID")
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
        observation = read_observation(request, arrays, self.spec.observation_schema)
        # Relative-action restoration must use the snapshot that produced this
        # request. Reading newer feedback here would change the model's result.
        result = self.policy.predict(self.adapter.to_request(observation))
        chunk = self.adapter.to_actions(result, observation)
        payload, values = chunk_payload(chunk, self.spec.action_schema)
        self._warmed = True
        return payload, values, self._run_id

    def serve(self, *, stop_event=None, ready_event=None):
        """Own a REP socket on this calling thread until Ctrl-C or stop_event.

        ready_event signals successful bind, not a model warmup. The supplied
        policy/adapter already exist and remain alive across client runs. Every
        consumed request gets one reply, including validation/model errors, so
        the REP socket can accept another request without changing ownership.
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
                        continue
                    request = {}
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
                    socket.send_multipart(reply)
