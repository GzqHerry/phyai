"""Resident pi0.5 server: model arrays -> preprocessing -> PhyAI -> actions.

The server knows the checkpoint, never the robot: it imports no ROS, robot
schema or calibration. Robot clients map their sensors to the advertised
inputs and the returned action vectors to their commands. Each run resets the
noise stream and starts with one discarded warmup. One run owns the server at
a time; a run that stops sending requests is released after its lease.

    python -m examples.deployment.serve_pi05 \\
        --config examples/deployment/configs/pi05_tianji.yaml \\
        policy.checkpoint=/path/to/checkpoint
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import field, dataclass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from phyai_robot.remote import PolicyServer

from .pi05 import Pi05Policy, Pi05PolicyConfig, Pi05ProcessorConfig
from .configuration import parse_config, repository_path
from .processor_loader import build_processing

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "pi05.yaml"


@dataclass
class Pi05ServerConfig:
    """Model-process settings; robot timing and calibration live in the client."""

    task: str = ""
    endpoint: str = "tcp://127.0.0.1:5555"
    # Release a run after this long without requests from its client.
    lease_timeout_s: float = 60.0
    policy: Pi05PolicyConfig = field(default_factory=Pi05PolicyConfig)
    processor: Pi05ProcessorConfig = field(default_factory=Pi05ProcessorConfig)

    def validate(self) -> None:
        if not self.task.strip() or not self.endpoint.strip():
            raise ValueError("task and endpoint must not be empty")
        if self.lease_timeout_s <= 0:
            raise ValueError("lease_timeout_s must be positive")
        self.policy.validate()
        self.processor.validate()


def main(argv=None) -> None:
    """Load once, serve many sequential robot runs, close the engine on exit."""
    config = parse_config(
        Pi05ServerConfig,
        default_config=DEFAULT_CONFIG,
        argv=argv,
        validate=Pi05ServerConfig.validate,
    )
    checkpoint = repository_path(config.policy.checkpoint)
    processing = build_processing(checkpoint, task=config.task, config=config.processor)
    policy = Pi05Policy(config.policy, processing)
    try:
        spec = policy.spec(
            {**processing.metadata, "task": processing.task, "seed": config.policy.seed}
        )
        server = PolicyServer(
            policy=policy,
            spec=spec,
            reset_run=policy.reset_run,
            endpoint=config.endpoint,
            lease_timeout_s=config.lease_timeout_s,
        )
        print(
            f"pi0.5 server on {config.endpoint}: task={processing.task!r}, "
            f"cameras={processing.camera_names}, action_dim={processing.action_dim}, "
            f"horizon={policy.horizon}, denoise_steps={policy.num_inference_steps}",
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
