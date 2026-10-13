"""Optional robot support; importing this package loads no GPU or transport SDK."""

from .robot import Robot, CompositeRobot, take_snapshot, wait_for_observation
from .types import (
    Action,
    Sample,
    Schema,
    ActionChunk,
    FeatureSpec,
    Observation,
    ActionSchema,
    ObservationSchema,
    ObservationNotReady,
)
from .policy import Policy, EnginePolicy, PolicyAdapter
from .scheduling import Scheduler, StopAndRefill, InferenceContext
from .deployment import RobotDeployment, DeploymentOptions

__all__ = [
    "Action",
    "ActionChunk",
    "ActionSchema",
    "CompositeRobot",
    "DeploymentOptions",
    "EnginePolicy",
    "FeatureSpec",
    "InferenceContext",
    "Observation",
    "ObservationNotReady",
    "ObservationSchema",
    "Policy",
    "PolicyAdapter",
    "Robot",
    "RobotDeployment",
    "Sample",
    "Scheduler",
    "Schema",
    "StopAndRefill",
    "take_snapshot",
    "wait_for_observation",
]
