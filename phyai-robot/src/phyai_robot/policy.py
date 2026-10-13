"""Prediction and robot/checkpoint conversion, without importing a model stack."""

from __future__ import annotations

from typing import Any, Generic, TypeVar, Protocol

from .types import ActionChunk, Observation

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class Policy(Protocol[RequestT, ResultT]):
    """Bind model input/output types without prescribing their data layout.

    RequestT is the object produced by the paired adapter. ResultT is the
    model output consumed by that adapter; neither type must be a robot Action.
    """

    def predict(self, request: RequestT) -> ResultT:
        """Run one synchronous inference and return a model-native result."""
        ...


class PolicyAdapter(Protocol[RequestT, ResultT]):
    """Connect robot snapshots to one policy's concrete input/output types."""

    def to_request(self, observation: Observation) -> RequestT:
        """Select fields and apply checkpoint-specific preprocessing."""
        ...

    def to_actions(self, result: ResultT, observation: Observation) -> ActionChunk:
        """Produce physical-unit targets using the exact request snapshot.

        The paired observation permits conversion of relative predictions to
        absolute targets. Never reread hardware here or silently guess units.
        """
        ...


class EnginePolicy(Generic[RequestT, ResultT]):
    """Wrap Engine.step; the application supplies its request and result types.

    The engine itself stays dynamic so this standalone package does not need
    to import PhyAI. Its step(request) must implement RequestT -> ResultT.
    The application owns the engine and remains responsible for closing it.
    """

    def __init__(self, engine: Any) -> None:
        if not callable(getattr(engine, "step", None)):
            raise TypeError("engine must provide step(request)")
        self._engine: Any = (
            engine  # Shared or private Engine; never closed by this wrapper.
        )

    def predict(self, request: RequestT) -> ResultT:
        return self._engine.step(request)


class ObservationAdapter:
    """Pair an Observation -> ActionChunk policy with the existing control loop."""

    def to_request(self, observation: Observation) -> Observation:
        return observation

    def to_actions(self, result: ActionChunk, observation: Observation) -> ActionChunk:
        return result
