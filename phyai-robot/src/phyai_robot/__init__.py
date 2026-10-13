"""Optional robot support; importing this package loads no GPU or transport SDK."""

from .robot import Robot, CompositeRobot
from .types import (
    Action,
    Sample,
    ActionChunk,
    FeatureSpec,
    Observation,
    ActionSchema,
    ObservationSchema,
    ObservationNotReady,
)
from .policy import Policy, EnginePolicy, PolicyAdapter, ObservationAdapter
from .deployment import RobotDeployment, DeploymentOptions

__all__ = [
    "Action",
    "ActionChunk",
    "ActionSchema",
    "CompositeRobot",
    "DeploymentOptions",
    "EnginePolicy",
    "FeatureSpec",
    "Observation",
    "ObservationAdapter",
    "ObservationNotReady",
    "ObservationSchema",
    "Policy",
    "PolicyAdapter",
    "Robot",
    "RobotDeployment",
    "Sample",
]
