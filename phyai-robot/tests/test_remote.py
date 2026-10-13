"""Wire and lifecycle tests use synthetic data; no model or ROS installation."""

from __future__ import annotations

import json
import time
import socket
from threading import Event, Thread
from contextlib import contextmanager

import numpy as np
import pytest

pytest.importorskip("zmq")
from phyai_robot import (
    Sample,
    ActionChunk,
    FeatureSpec,
    Observation,
    ObservationAdapter,
)
from phyai_robot.remote import PolicySpec, PolicyServer, RemotePolicy, RemotePolicyError
from phyai_robot.remote._wire import (
    pack,
    unpack,
    read_chunk,
    chunk_payload,
    read_observation,
    observation_payload,
)

SCHEMA = {"position": FeatureSpec((2,), "float64", "rad", ("left", "right"))}
SPEC = PolicySpec("mock", SCHEMA, SCHEMA, {"horizon": 3})
OBS = Observation({"position": Sample(np.array([1.0, 2.0]), 12345)})


def endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


class Predictor:
    def __init__(self):
        self.resets = 0
        self.calls = 0
        self.seen = []
        self.gate = None

    def reset(self):
        self.resets += 1
        self.calls = 0

    def predict(self, observation):
        self.calls += 1
        self.seen.append(observation)
        if self.gate is not None:
            assert self.gate.wait(2)
        return ActionChunk(
            tuple(
                {"position": observation.samples["position"].value + self.calls + i}
                for i in range(3)
            ),
            0.04,
        )


@contextmanager
def running():
    predictor = Predictor()
    stop, ready = Event(), Event()
    url = endpoint()
    server = PolicyServer(
        policy=predictor,
        adapter=ObservationAdapter(),
        spec=SPEC,
        reset_run=predictor.reset,
        endpoint=url,
    )
    errors = []

    def serve():
        try:
            server.serve(stop_event=stop, ready_event=ready)
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=serve)
    thread.start()
    try:
        assert ready.wait(3)
        yield server, predictor, url
    finally:
        if predictor.gate is not None:
            predictor.gate.set()
        stop.set()
        thread.join(3)
        assert not thread.is_alive()
        assert not errors


def begin(client):
    return client.begin_run(
        embodiment="mock", observation_schema=SCHEMA, action_schema=SCHEMA
    )


@pytest.mark.parametrize(
    "value",
    [
        np.array(1.5),
        np.arange(12, dtype=">f8").reshape(3, 4)[:, ::2],
        np.zeros((0, 2), dtype=np.uint8),
        np.array([True, False]),
        np.array([-0.0, 1.0], dtype=np.float32),
    ],
)
def test_wire_preserves_bytes_dtype_shape(value):
    header, values = unpack(pack({"key": 4}, [value]))
    actual = values[0]
    assert actual.shape == value.shape
    assert actual.dtype.str == value.dtype.str
    assert actual.tobytes() == value.tobytes()
    assert actual.flags.writeable
    assert header["key"] == 4


@pytest.mark.parametrize("dtype", ["object", "complex128", "U2"])
def test_wire_rejects_non_numeric(dtype):
    with pytest.raises(ValueError):
        pack({}, [np.zeros(1, dtype=dtype)])


@pytest.mark.parametrize(
    "frames",
    [
        [b"[]"],
        [b'{"protocol":2,"arrays":[]}'],
        [b'{"protocol":true,"arrays":[]}'],
        [b'{"protocol":1,"arrays":[{"shape":[-1],"dtype":"f8"}]}', b""],
        [b'{"protocol":1,"arrays":[{"shape":[1],"dtype":"O"}]}', b"12345678"],
        [b'{"protocol":1,"arrays":[]}', b"extra"],
    ],
)
def test_wire_rejects_invalid_frames(frames):
    with pytest.raises(ValueError):
        unpack(frames)


def test_observation_preserves_robot_timestamp_without_server_clock():
    meta, arrays = observation_payload(OBS)
    obs = read_observation(*unpack(pack(meta, arrays)), SCHEMA)
    assert obs.samples["position"].received_at_ns == 12345
    np.testing.assert_array_equal(obs.samples["position"].value, [1, 2])
    meta["samples"][0]["received_at_ns"] = -1
    with pytest.raises(ValueError):
        read_observation(meta, arrays, SCHEMA)


def test_schema_checks_order_dtype_units_and_embodiment():
    SPEC.check("mock", SCHEMA, SCHEMA)
    for spec in [
        FeatureSpec((2,), "float32", "rad", ("left", "right")),
        FeatureSpec((2,), "float64", "m", ("left", "right")),
        FeatureSpec((2,), "float64", "rad", ("right", "left")),
    ]:
        with pytest.raises(ValueError):
            SPEC.check("mock", {"position": spec}, SCHEMA)
    with pytest.raises(ValueError):
        SPEC.check("other", SCHEMA, SCHEMA)


def test_chunk_roundtrip_and_validation():
    chunk = Predictor().predict(OBS)
    meta, values = chunk_payload(chunk, SCHEMA)
    actual = read_chunk(*unpack(pack(meta, values)), SCHEMA)
    assert actual.step_period_s == chunk.step_period_s
    for left, right in zip(actual.actions, chunk.actions):
        assert left["position"].tobytes() == right["position"].tobytes()
    for chunk in [
        ActionChunk((), 0.04),
        ActionChunk(({"position": np.array([np.nan, 1.0])},), 0.04),
        ActionChunk(({"position": np.ones(2)},), 0),
    ]:
        with pytest.raises(ValueError):
            chunk_payload(chunk, SCHEMA)


def test_run_reset_warmup_exactly_once_and_worker_socket_ownership():
    with running() as (_, predictor, url):
        for run in range(3):
            client = RemotePolicy(url)
            assert client.info().metadata["horizon"] == 3
            begin(client)
            with pytest.raises(RuntimeError):
                client.predict(OBS)
            warm = client.warmup(OBS)
            with pytest.raises(RuntimeError):
                client.warmup(OBS)
            result = []
            worker = Thread(target=lambda: result.append(client.predict(OBS)))
            worker.start()
            worker.join(3)
            assert not worker.is_alive()
            np.testing.assert_array_equal(warm.actions[0]["position"], [2, 3])
            np.testing.assert_array_equal(result[0].actions[0]["position"], [3, 4])
            assert predictor.calls == 2
            client.close()
            client.close()
        assert predictor.resets == 3


def test_incorrect_schema_does_not_claim_server_and_busy_does_not_reset_rng():
    with running() as (_, predictor, url):
        bad = RemotePolicy(url)
        with pytest.raises(RemotePolicyError, match="embodiment"):
            bad.begin_run(
                embodiment="wrong", observation_schema=SCHEMA, action_schema=SCHEMA
            )
        bad.close()
        owner = RemotePolicy(url)
        begin(owner)
        other = RemotePolicy(url)
        with pytest.raises(RemotePolicyError, match="active run"):
            begin(other)
        other.close()
        assert predictor.resets == 1
        owner.warmup(OBS)
        owner.close()


def test_request_and_run_ids_reject_replay():
    with running() as (server, _, url):
        client = RemotePolicy(url)
        begin(client)
        client.warmup(OBS)
        with pytest.raises(ValueError, match="stale request"):
            server._dispatch(
                dict(operation="predict", request_id=1, run_id=client._run_id), []
            )
        with pytest.raises(ValueError, match="run ID"):
            server._dispatch(dict(operation="end", request_id=99, run_id="old"), [])
        client.close()


def test_timeout_is_bounded_and_never_retried():
    client = RemotePolicy(endpoint(), timeout_s=0.04)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        begin(client)
    assert time.monotonic() - start < 1
    with pytest.raises(RuntimeError, match="failed"):
        begin(client)
    assert client._request_id == 1
    client.close()


def test_server_survives_malformed_message():
    import zmq

    with running() as (_, _, url), zmq.Context() as ctx:
        with ctx.socket(zmq.REQ) as sock:
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(url)
            sock.send_multipart([b"[]"])
            assert sock.poll(1000)
            assert not unpack(sock.recv_multipart())[0]["ok"]
        client = RemotePolicy(url)
        begin(client)
        client.warmup(OBS)
        client.close()


def test_close_while_worker_inflight_releases_only_after_prediction():
    with running() as (server, predictor, url):
        client = RemotePolicy(url)
        begin(client)
        client.warmup(OBS)
        predictor.gate = Event()
        result = []
        thread = Thread(target=lambda: result.append(client.predict(OBS)))
        thread.start()
        limit = time.monotonic() + 1
        while predictor.calls != 2 and time.monotonic() < limit:
            time.sleep(0.001)
        assert predictor.calls == 2
        client.close()
        assert server._run_id is not None
        predictor.gate.set()
        thread.join(3)
        assert len(result) == 1
        assert server._run_id is None


def test_adapter_receives_the_request_observation():
    class Adapter:
        def to_request(self, obs):
            self.obs = obs
            return obs

        def to_actions(self, result, obs):
            assert obs is self.obs
            return result

    with running() as (server, _, url):
        server.adapter = Adapter()
        client = RemotePolicy(url)
        begin(client)
        client.warmup(OBS)
        client.predict(OBS)
        client.close()


@pytest.mark.parametrize("field,value", [("request_id", 999), ("run_id", "stale")])
def test_client_rejects_mismatched_reply(field, value):
    import zmq

    urls = []
    ready, answered = Event(), Event()
    errors = []

    def fake_server():
        try:
            with zmq.Context() as context, context.socket(zmq.REP) as sock:
                sock.setsockopt(zmq.LINGER, 0)
                # Bind directly: a probed-then-released port can be taken first.
                port = sock.bind_to_random_port("tcp://127.0.0.1")
                urls.append(f"tcp://127.0.0.1:{port}")
                ready.set()
                assert sock.poll(5000)
                request, _ = unpack(sock.recv_multipart())
                response = dict(
                    ok=True,
                    spec=SPEC.to_dict(),
                    request_id=request["request_id"],
                    run_id=request["run_id"],
                )
                response[field] = value
                sock.send_multipart(pack(response))
                # LINGER=0 drops unsent replies on close; keep the socket open
                # until the client has its answer.
                assert answered.wait(5)
        except BaseException as error:
            errors.append(error)
            ready.set()

    thread = Thread(target=fake_server)
    thread.start()
    assert ready.wait(2) and urls, errors
    client = RemotePolicy(urls[0], timeout_s=5)
    try:
        with pytest.raises(RemotePolicyError, match="ID mismatch"):
            client.info()
    finally:
        answered.set()
        client.close()
        thread.join(6)
    assert not thread.is_alive() and not errors


def test_predict_failure_is_not_retried_and_run_must_be_released():
    with running() as (server, predictor, url):
        client = RemotePolicy(url)
        begin(client)
        client.warmup(OBS)
        calls = []

        def fail(request):
            calls.append(request)
            raise ValueError("injected model error")

        predictor.predict = fail
        with pytest.raises(RemotePolicyError, match="injected model error"):
            client.predict(OBS)
        with pytest.raises(RuntimeError, match="failed"):
            client.predict(OBS)
        assert len(calls) == 1 and server._run_id is not None
        client.close()
        assert server._run_id is None
