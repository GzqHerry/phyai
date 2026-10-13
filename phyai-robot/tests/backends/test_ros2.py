"""Test the ROS backend contract with in-memory topics, never real ROS contexts."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from phyai_robot.backends.ros2 import Ros2Backend, RosCommand, RosObservation
from phyai_robot.backends.ros2.codecs import decode_rgb_image, decode_joint_positions


def test_grouped_topics_preserve_fields_stamp_order_and_ownership(fake_ros):
    fields = ("joints", "gripper")

    def encode(values):
        msg = fake_ros.JointState()
        msg.position = np.concatenate([values[k] for k in fields]).tolist()
        return msg

    backend = Ros2Backend(
        observations={
            fields: RosObservation(
                "/feedback",
                fake_ros.JointState,
                lambda m: {
                    "joints": np.array(m.position[:2]),
                    "gripper": np.array(m.position[2:]),
                },
            )
        },
        actions={
            fields: RosCommand("/first", fake_ros.JointState, encode, stamp=True),
            ("other",): RosCommand(
                "/second",
                fake_ros.JointState,
                lambda v: fake_ros.JointState(),
                stamp=True,
            ),
        },
        on_stop=lambda node: None,
        domain_id=37,
    )
    try:
        backend.connect()
        msg = fake_ros.JointState()
        msg.position = [1.0, 2.0, 3.0]
        fake_ros.feed("/feedback", msg)
        observed = backend.read()
        assert set(observed) == set(fields)
        assert observed["joints"].received_at_ns == observed["gripper"].received_at_ns
        np.testing.assert_array_equal(observed["joints"].value, [1.0, 2.0])
        msg.position[0] = 99.0
        np.testing.assert_array_equal(backend.read()["joints"].value, [1.0, 2.0])
        backend.write(
            {
                "other": np.array([9.0]),
                "gripper": np.array([0.2]),
                "joints": np.array([0.3, 0.4]),
            }
        )
        assert [topic for topic, _ in fake_ros.publications] == ["/first", "/second"]
        assert fake_ros.publications[0][1].position == [0.3, 0.4, 0.2]
        assert [msg.header.stamp for _, msg in fake_ros.publications] == [1, 2]
        assert fake_ros.contexts[0].domain_id == 37
        backend.stop()
        with pytest.raises(RuntimeError, match="stopped"):
            backend.write(
                {"other": np.zeros(1), "gripper": np.zeros(1), "joints": np.zeros(2)}
            )
        assert len(fake_ros.publications) == 2
    finally:
        backend.close()
    assert all(thread != threading.get_ident() for _, _, thread in fake_ros.events)
    assert len({thread for _, _, thread in fake_ros.events}) == 1


def test_single_field_legacy_order_and_no_stamp(fake_ros):
    backend = Ros2Backend(
        observations={},
        actions={
            key: RosCommand(
                "/" + key, fake_ros.JointState, lambda value: fake_ros.JointState()
            )
            for key in ("a", "b")
        },
        on_stop=lambda node: None,
    )
    try:
        backend.connect()
        backend.write({"b": np.zeros(1), "a": np.ones(1)})
        assert [topic for topic, _ in fake_ros.publications] == ["/b", "/a"]
        assert all(message.header.stamp is None for _, message in fake_ros.publications)
        assert fake_ros.contexts[0].domain_id is None
    finally:
        backend.close()


@pytest.mark.parametrize(
    "bindings",
    [{(): None}, {("a", "a"): None}, {("a", "b"): None, "b": None}, {("a", ""): None}],
)
def test_reject_ambiguous_field_bindings(bindings):
    with pytest.raises(ValueError):
        Ros2Backend(observations=bindings, actions={})


def test_recoverable_decode_error_clears_only_after_valid_message(fake_ros):
    def decode(message):
        if message.bad:
            raise ValueError("bad frame")
        return np.array([message.value])

    backend = Ros2Backend(
        observations={
            "camera": RosObservation(
                "/camera", object, decode, recover_on_next_message=True
            )
        },
        actions={},
    )
    try:
        backend.connect()
        fake_ros.feed("/camera", SimpleNamespace(bad=True))
        with pytest.raises(RuntimeError, match="bad frame"):
            backend.read()
        fake_ros.feed("/camera", SimpleNamespace(bad=False, value=7))
        np.testing.assert_array_equal(backend.read()["camera"].value, [7])
    finally:
        backend.close()


def test_bad_group_cannot_cache_partial_observation(fake_ros):
    backend = Ros2Backend(
        observations={
            ("a", "b"): RosObservation(
                "/s", object, lambda msg: msg, recover_on_next_message=True
            )
        },
        actions={},
    )
    try:
        backend.connect()
        fake_ros.feed("/s", {"a": np.array([1])})
        with pytest.raises(RuntimeError, match="fields do not match"):
            backend.read()
        fake_ros.feed("/s", {"a": np.array([2]), "b": np.array([3])})
        assert set(backend.read()) == {"a", "b"}
    finally:
        backend.close()


def test_invalid_command_type_publishes_nothing(fake_ros):
    backend = Ros2Backend(
        observations={},
        actions={
            ("a",): RosCommand(
                "/a", fake_ros.JointState, lambda values: fake_ros.JointState()
            ),
            ("b",): RosCommand("/b", fake_ros.JointState, lambda values: object()),
        },
        on_stop=lambda node: None,
    )
    backend.connect()
    with pytest.raises(TypeError, match="message type"):
        backend.write({"a": np.zeros(1), "b": np.zeros(1)})
    with pytest.raises(RuntimeError, match="transport worker failed"):
        backend.close()
    assert not fake_ros.publications


@pytest.mark.parametrize(
    "encoding,data,expected",
    [
        ("rgb8", [1, 2, 3, 4, 5, 6, 0, 0], [1, 2, 3, 4, 5, 6]),
        ("bgr8", [1, 2, 3, 4, 5, 6, 0, 0], [3, 2, 1, 6, 5, 4]),
        ("rgba8", [1, 2, 3, 99, 4, 5, 6, 99], [1, 2, 3, 4, 5, 6]),
        ("bgra8", [1, 2, 3, 99, 4, 5, 6, 99], [3, 2, 1, 6, 5, 4]),
        ("mono8", [1, 2, 0, 0], [1, 1, 1, 2, 2, 2]),
    ],
)
def test_rgb_image_row_padding_and_encoding(encoding, data, expected):
    result = decode_rgb_image(
        SimpleNamespace(
            encoding=encoding, data=bytes(data), height=1, width=2, step=len(data)
        )
    )
    np.testing.assert_array_equal(
        result.reshape(-1), np.array(expected, dtype=np.uint8)
    )
    assert (
        result.shape == (1, 2, 3)
        and result.dtype == np.uint8
        and result.flags.c_contiguous
    )


def test_joint_positions_follow_requested_names():
    result = decode_joint_positions(
        SimpleNamespace(name=["b", "a", "extra"], position=[2, 1, 3]), ("a", "b")
    )
    np.testing.assert_array_equal(result, [1.0, 2.0])
    assert result.dtype == np.float64
    with pytest.raises(ValueError):
        decode_joint_positions(
            SimpleNamespace(name=["a", "a"], position=[1, 2]), ("a",)
        )


def test_multiple_bindings_can_share_one_topic(fake_ros):
    backend = Ros2Backend(
        observations={
            key: RosObservation(
                "/shared", object, lambda message, key=key: message[key]
            )
            for key in ("a", "b")
        },
        actions={},
    )
    try:
        backend.connect()
        fake_ros.feed("/shared", {"a": np.array([1.0]), "b": np.array([2.0])})
        result = backend.read()
        np.testing.assert_array_equal(result["a"].value, [1.0])
        np.testing.assert_array_equal(result["b"].value, [2.0])
    finally:
        backend.close()


def test_default_decode_error_remains_fatal(fake_ros):
    def fail(message):
        raise ValueError("bad decode")

    backend = Ros2Backend(
        observations={"q": RosObservation("/q", object, fail)}, actions={}
    )
    backend.connect()
    fake_ros.feed("/q", object())
    with pytest.raises(RuntimeError, match="transport worker failed"):
        backend.close()
    assert not fake_ros.publications
