"""AgileX dual-Piper robot client. See AGILEX.md for the full startup procedure.

Start serve_pi05 with configs/pi05_agilex.yaml first. This process needs ROS
Humble and its camera/Piper drivers, but no model stack. It sends robot
commands after warmup and a fresh observation; Ctrl-C stops publishing.

    python -m examples.deployment.agilex --print-config
    python -m examples.deployment.agilex schedule.execution_horizon=20
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import field, dataclass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from .client import RemoteConfig, ControlConfig, ScheduleConfig, main
from .robots.agilex import AgilexConfig, build

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "agilex.yaml"


@dataclass
class AgilexClientConfig:
    max_steps: int | None = None  # Actual sends; null runs until Ctrl-C.
    robot: AgilexConfig = field(default_factory=AgilexConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)


if __name__ == "__main__":
    main(AgilexClientConfig, build, default_config=DEFAULT_CONFIG)
