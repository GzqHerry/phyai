"""Prediction and robot/model conversion, without importing a model stack."""

from __future__ import annotations

from typing import Any, Generic, TypeVar, Protocol

from .types import ActionChunk, Observation
from .scheduling import InferenceContext

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

    def to_request(
        self, observation: Observation, context: InferenceContext
    ) -> RequestT:
        """Select fields and convert them to the model's input convention.

        context describes what the scheduler will still execute while this
        prediction runs. Stop-and-refill always passes an empty context.
        """
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
        self._engine: Any = engine  # Shared or private; never closed by this wrapper.

    def predict(self, request: RequestT) -> ResultT:
        return self._engine.step(request)
