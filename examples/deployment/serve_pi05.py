"""Resident Pi05 server: adapter preprocessing -> PhyAI -> physical ActionChunk.

This entry point never imports ROS or creates a robot. Start the corresponding
robot-only client separately. Each begin_run resets noise and requires exactly
one discarded warmup. Only one robot run may own the server at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import field, dataclass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from phyai_robot.remote import PolicySpec, PolicyServer

from .policies import Pi05Policy, Pi05PolicyConfig
from .configuration import parse_config, repository_path
from .adapters.tianji_pi05 import TianjiPi05AdapterConfig

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "pi05_agilex.yaml"


@dataclass
class Pi05ServerConfig:
    """Model-process settings, separate from control rate and execution horizon.

    checkpoint and task are required at startup. AgileX camera dimensions define
    its raw observation contract; Tianji obtains resized image_size from its
    processor. A running server keeps these settings fixed across client runs.
    """

    embodiment: str = "agilex"
    task: str = ""
    endpoint: str = "tcp://127.0.0.1:5555"
    policy: Pi05PolicyConfig = field(default_factory=Pi05PolicyConfig)
    adapter: TianjiPi05AdapterConfig = field(default_factory=TianjiPi05AdapterConfig)
    camera_height: int = 480
    camera_width: int = 640

    def validate(self):
        if self.embodiment not in ("agilex", "tianji"):
            raise ValueError("embodiment must be agilex or tianji")
        if not self.task.strip() or not self.endpoint.strip():
            raise ValueError("task and endpoint must not be empty")
        if min(self.camera_height, self.camera_width) < 1:
            raise ValueError("camera dimensions must be positive")
        self.policy.validate()
        if self.embodiment == "tianji":
            self.adapter.validate(execution_horizon=50)


def build_adapter(config):
    """Pair model preprocessing with a pure physical schema, without robot I/O.

    Return the adapter and handshake metadata. Import layouts rather than robot
    factories so a GPU host never needs ROS just to learn joint order or units.
    """
    checkpoint = repository_path(config.policy.checkpoint)
    if config.embodiment == "agilex":
        from .layouts.agilex import ACTION_SCHEMA, observation_schema
        from .adapters.agilex_pi05 import AgilexPi05Adapter

        adapter = AgilexPi05Adapter(checkpoint, task=config.task)
        schema = observation_schema(config.camera_height, config.camera_width)
        metadata = dict(horizon=adapter.horizon)
    else:
        from .layouts.tianji import ACTION_SCHEMA, observation_schema
        from .adapters.tianji_pi05 import ACTION_HORIZON, TianjiPi05Adapter

        adapter = TianjiPi05Adapter(
            checkpoint,
            task=config.task,
            metadata_file=config.adapter.metadata_file,
            gripper_limits_rad=tuple(config.adapter.gripper_limits_rad),
        )
        schema = observation_schema(adapter.image_size)
        metadata = dict(horizon=ACTION_HORIZON, image_size=adapter.image_size)
    metadata.update(task=config.task, seed=config.policy.seed)
    return adapter, PolicySpec(config.embodiment, schema, ACTION_SCHEMA, metadata)


def main(argv=None):
    """Load once, serve many sequential robot runs, close the engine on exit."""
    config = parse_config(
        Pi05ServerConfig,
        default_config=DEFAULT_CONFIG,
        argv=argv,
        validate=Pi05ServerConfig.validate,
    )
    adapter, spec = build_adapter(config)
    policy = Pi05Policy(
        repository_path(config.policy.checkpoint),
        kernel_policy=repository_path(config.policy.kernel_policy),
        seed=config.policy.seed,
        num_inference_steps=config.policy.num_inference_steps,
        use_cuda_graph=config.policy.use_cuda_graph,
        num_threads=config.policy.num_threads,
    )
    try:
        spec.metadata["num_inference_steps"] = policy.num_inference_steps
        # begin_run resets only the noise stream, not weights or CUDA graphs.
        # The application owns model cleanup; the generic server owns sockets.
        server = PolicyServer(
            policy=policy,
            adapter=adapter,
            spec=spec,
            reset_run=policy.reset_run,
            endpoint=config.endpoint,
        )
        print(
            f"Pi05 {config.embodiment} server: {config.endpoint}; task={config.task!r}; denoise_steps={policy.num_inference_steps}",
            flush=True,
        )
        server.serve()
    finally:
        policy.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
