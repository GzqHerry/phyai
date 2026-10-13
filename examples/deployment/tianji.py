"""Tianji (Marvin Pro) robot client. See ROBOT_ENVIRONMENTS.md for setup.

Start serve_pi05 with configs/pi05_tianji.yaml first. This process needs ROS
Jazzy plus the controller's marvin_msgs overlay, but no model stack. It sends
robot commands after warmup and a fresh observation; Ctrl-C requests input=0.

    python -m examples.deployment.tianji --print-config
    python -m examples.deployment.tianji schedule.execution_horizon=20
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import field, dataclass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from .client import RemoteConfig, ControlConfig, ScheduleConfig, main
from .robots.tianji import TianjiConfig, build

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "tianji.yaml"


@dataclass
class TianjiClientConfig:
    max_steps: int | None = None  # Actual sends; null runs until Ctrl-C.
    robot: TianjiConfig = field(default_factory=TianjiConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)


if __name__ == "__main__":
    main(TianjiClientConfig, build, default_config=DEFAULT_CONFIG)
