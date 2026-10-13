"""Entry-point configuration parsing and server-side checkpoint metadata."""

import json

import pytest

from examples.deployment import agilex, tianji
from examples.deployment.pi05 import Pi05ProcessorConfig, _read_metadata
from examples.deployment.client import validate_client
from examples.deployment.serve_pi05 import DEFAULT_CONFIG, Pi05ServerConfig
from examples.deployment.configuration import parse_config


@pytest.mark.parametrize(
    "module, schema, action_hz, horizon",
    [
        (tianji, tianji.TianjiClientConfig, 25.0, 20),
        (agilex, agilex.AgilexClientConfig, 30.0, 25),
    ],
)
def test_bundled_client_configs_are_valid(module, schema, action_hz, horizon):
    config = parse_config(
        schema, default_config=module.DEFAULT_CONFIG, argv=[], validate=validate_client
    )
    assert config.schedule.action_hz == action_hz
    assert config.schedule.execution_horizon == horizon
    assert config.control.to_options().control_hz == 200.0


def test_overrides_reach_nested_robot_settings():
    config = parse_config(
        tianji.TianjiClientConfig,
        default_config=tianji.DEFAULT_CONFIG,
        argv=["robot.mode=1", "robot.wrench_topics.left=/w", "schedule.action_hz=50"],
        validate=validate_client,
    )
    assert config.robot.mode == 1
    assert config.robot.wrench_topics == {"left": "/w"}
    assert config.robot.domain_id == 24
    assert config.schedule.create().action_hz == 50


@pytest.mark.parametrize(
    "override", ["schedule.strategy=rtc", "robot.mode=2", "control.control_hz=0"]
)
def test_invalid_client_settings_exit_before_any_io(override):
    with pytest.raises(SystemExit):
        parse_config(
            tianji.TianjiClientConfig,
            default_config=tianji.DEFAULT_CONFIG,
            argv=[override],
            validate=validate_client,
        )


def test_server_config_requires_an_existing_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        parse_config(
            Pi05ServerConfig,
            default_config=DEFAULT_CONFIG,
            argv=["task=pick", f"policy.checkpoint={tmp_path / 'missing'}"],
            validate=Pi05ServerConfig.validate,
        )
    config = parse_config(
        Pi05ServerConfig,
        default_config=DEFAULT_CONFIG,
        argv=["task=pick", f"policy.checkpoint={tmp_path}"],
        validate=Pi05ServerConfig.validate,
    )
    assert config.processor.image_resize_backend == "pil"


def metadata(**overrides):
    values = dict(
        action_dim=2,
        action_horizon=50,
        camera_names=["a", "b"],
        delta_mask=[True, False],
        norm_eps=1e-6,
    )
    values.update(overrides)
    return values


def test_metadata_discovery_is_generic_and_strict(tmp_path):
    (tmp_path / "x_metadata.json").write_text(json.dumps(metadata()))
    assert _read_metadata(tmp_path, None)["camera_names"] == ["a", "b"]
    (tmp_path / "y_metadata.json").write_text(json.dumps(metadata()))
    with pytest.raises(ValueError, match="metadata_file"):
        _read_metadata(tmp_path, None)
    assert _read_metadata(tmp_path, "y_metadata.json")["action_dim"] == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"delta_mask": [True]},
        {"camera_names": ["a", "a"]},
        {"action_dim": 0},
        {"norm_eps": 0},
        {"delta_mask": [1, 0]},
    ],
)
def test_inconsistent_metadata_is_rejected(tmp_path, overrides):
    (tmp_path / "deployment_metadata.json").write_text(
        json.dumps(metadata(**overrides))
    )
    with pytest.raises(ValueError):
        _read_metadata(tmp_path, None)


def test_processor_config_validation():
    Pi05ProcessorConfig(camera_names=["a"], action_dim=3).validate()
    for bad in (
        Pi05ProcessorConfig(camera_names=[]),
        Pi05ProcessorConfig(camera_names=["a", "a"]),
        Pi05ProcessorConfig(action_dim=0),
        Pi05ProcessorConfig(image_resize_backend="cv2"),
    ):
        with pytest.raises(ValueError):
            bad.validate()
