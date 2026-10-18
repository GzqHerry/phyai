"""PhyAI Pi0.5 model server policy: preprocessing, engine, and postprocessing.

Design Rationale:
    This module encapsulates all checkpoint-specific mathematical and tensor operations
    on the model server side. It is 100% embodiment-agnostic (zero imports of robot
    drivers, ROS 2, or robot-specific hardware parameters).

Protocol Contract (VLA Vector Interface):
    - Request: Input dictionary mapping feature keys to NumPy arrays:
        - 'image.<camera_name>': HWC uint8 RGB image array per checkpoint camera
        - 'state': 1D float32 array containing the concatenated physical state vector
    - Response: Output dictionary containing:
        - 'actions': 2D array of shape (action_horizon, action_dim) containing absolute
          unnormalized actuator targets in physical engineering units.

Dual Export Format Support:
    1. LeRobot Format (AgileX / Enactive):
       Identified by the presence of `policy_preprocessor.json` and `policy_postprocessor.json`.
       The serialized pipelines handle quantile normalization, custom prompt encoding,
       and internal DeltaActionStep target restoration.
    2. Metadata Format (Tianji / RLinf / OpenPI JAX):
       Identified by `norm_stats.json` plus `deployment_metadata.json` (or `*_metadata.json`).
       Uses high-precision float64 quantile statistics (eps=1e-6) and applies delta_mask
       anchoring to reconstruct absolute joint positions from predicted deltas.
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from threading import Lock, Event
from dataclasses import field, replace, dataclass
from collections.abc import Mapping

import numpy as np
from omegaconf import MISSING
from phyai_robot import FeatureSpec
from numpy.typing import NDArray
from phyai_robot.remote import PolicySpec
from phyai_robot.adapters import STATE_KEY, ACTIONS_KEY, IMAGE_PREFIX

from .configuration import repository_path


@dataclass
class Pi05PolicyConfig:
    """Configuration dataclass for Pi0.5 model weights and the PhyAI inference engine.

    Attributes:
        checkpoint: Local path or repo path to the exported Pi0.5 checkpoint directory.
            Must contain config.json, safetensors model weights, norm_stats.json, and tokenizer files.
        kernel_policy: Path to kernel selection YAML policy enforcing exact GELU implementations.
        seed: Random seed for the CPU diffusion initial Gaussian noise generator.
        num_inference_steps: Number of Euler integration steps during flow-matching denoising.
            Set to null to use the checkpoint's native default (e.g. 5 steps for Tianji, 10 for AgileX).
        use_cuda_graph: Enable CUDA graph capture and replay for minimized execution jitter.
        num_threads: Number of CPU worker threads allocated to the PhyAI engine runtime.
    """

    checkpoint: str = MISSING
    kernel_policy: str = "examples/deployment/configs/pi05_bf16.yaml"
    seed: int = 42
    num_inference_steps: int | None = None
    use_cuda_graph: bool = True
    num_threads: int = 16

    def validate(self) -> None:
        """Validate configuration paths and parameters without allocating GPU memory."""
        if self.num_inference_steps is not None and self.num_inference_steps < 1:
            raise ValueError("policy.num_inference_steps must be positive or null")
        if self.num_threads < 1:
            raise ValueError("policy.num_threads must be positive")
        if not 0 <= self.seed < 2**64:
            raise ValueError("policy.seed must be in [0, 2**64)")
        if not self.checkpoint.strip() or not repository_path(self.checkpoint).is_dir():
            raise ValueError("policy.checkpoint must name an existing directory")
        if not repository_path(self.kernel_policy).is_file():
            raise ValueError("policy.kernel_policy must name an existing YAML file")


@dataclass
class Pi05ProcessorConfig:
    """Configuration options for model tokenization, normalization, and image geometry.

    Attributes:
        factory: Optional custom processor factory specification ('module:Class' or 'path.py:Class').
            Null selects the default Pi05Processing wrapper.
        options: Extra keyword arguments passed exclusively to an external custom factory.
        camera_names: Ordered list of camera view names expected by the model vision tower.
            Required for LeRobot exports; verified against deployment metadata for RLinf exports.
        action_dim: Unpadded physical action dimensionality (e.g. 14 for AgileX, 16 for Tianji).
            Required for LeRobot exports; verified against deployment metadata for RLinf exports.
        metadata_file: Relative filename of export metadata JSON inside checkpoint directory.
            If null, automatically discovers deployment_metadata.json or unique *_metadata.json.
        image_resize_backend: Resizing implementation ('pil' or 'torch').
            'pil' is recommended to preserve exact training-time PIL bilinear rounding.
    """

    factory: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    camera_names: list[str] | None = None
    action_dim: int | None = None
    metadata_file: str | None = None
    image_resize_backend: str = "pil"

    def validate(self) -> None:
        """Verify that processor configuration options are well-formed."""
        if self.factory is not None and (
            not self.factory.strip() or ":" not in self.factory
        ):
            raise ValueError(
                "processor.factory must be module:factory or file.py:factory"
            )
        if self.factory is None and self.options:
            raise ValueError("processor.options requires processor.factory")
        if {"checkpoint", "task", "config"}.intersection(self.options):
            raise ValueError(
                "processor.options cannot override checkpoint, task or config"
            )
        names = self.camera_names
        if names is not None and (
            not names or len(set(names)) != len(names) or not all(names)
        ):
            raise ValueError("processor.camera_names must be unique nonempty names")
        if self.action_dim is not None and self.action_dim < 1:
            raise ValueError("processor.action_dim must be positive or null")
        if self.image_resize_backend not in ("pil", "torch"):
            raise ValueError("processor.image_resize_backend must be pil or torch")


def _read_metadata(checkpoint: Path, metadata_file: str | None) -> dict[str, Any]:
    """Discover and parse checkpoint export metadata JSON for metadata-style checkpoints."""
    if metadata_file is not None:
        path = checkpoint / Path(metadata_file).expanduser()
    elif (checkpoint / "deployment_metadata.json").is_file():
        path = checkpoint / "deployment_metadata.json"
    else:
        candidates = sorted(checkpoint.glob("*_metadata.json"))
        if len(candidates) != 1:
            raise ValueError("set processor.metadata_file: expected one metadata JSON")
        path = candidates[0]

    metadata = json.loads(path.read_text())
    action_dim = metadata.get("action_dim")
    names = metadata.get("camera_names")
    mask = metadata.get("delta_mask")
    eps = metadata.get("norm_eps")

    if type(action_dim) is not int or action_dim < 1:
        raise ValueError("checkpoint action_dim must be a positive integer")
    if (
        not isinstance(names, list)
        or not names
        or len(set(names)) != len(names)
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError("checkpoint camera_names must be unique nonempty strings")
    if (
        not isinstance(mask, list)
        or len(mask) != action_dim
        or any(type(v) is not bool for v in mask)
    ):
        raise ValueError("checkpoint delta_mask must have one boolean per action")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not np.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("checkpoint norm_eps must be finite and positive")
    return metadata


class Pi05Processing:
    """Encapsulates image preprocessing, state/action normalization, and tokenization."""

    def __init__(
        self, checkpoint: Path, *, task: str, config: Pi05ProcessorConfig
    ) -> None:
        """Initialize preprocessor and postprocessor pipelines for a Pi0.5 checkpoint.

        Inspects the checkpoint directory to determine whether it follows LeRobot format
        or OpenPI metadata format, configuring the appropriate processor steps.
        """
        import torch
        from phyai_utils_tools.models.pi05 import PI05Processor
        from phyai_utils_tools.processing.steps import NormalizerStep, UnnormalizerStep

        if not task.strip():
            raise ValueError("task must be non-empty")
        self.task = task.strip()
        self._torch = torch
        self.metadata: dict[str, Any] = {}

        if (checkpoint / "policy_preprocessor.json").is_file():
            # Format A: LeRobot pipeline export (e.g. AgileX Enactive).
            if (
                config.metadata_file is not None
                or (checkpoint / "deployment_metadata.json").is_file()
            ):
                self.metadata = _read_metadata(checkpoint, config.metadata_file)
            names = (
                config.camera_names
                if config.camera_names is not None
                else self.metadata.get("camera_names")
            )
            width = (
                config.action_dim
                if config.action_dim is not None
                else self.metadata.get("action_dim")
            )
            if names is None or width is None:
                raise ValueError(
                    "lerobot-style exports need processor.camera_names and "
                    "processor.action_dim or deployment metadata declaring them"
                )
            for name, value in (("camera_names", names), ("action_dim", width)):
                if name in self.metadata and value != self.metadata[name]:
                    raise ValueError(f"processor.{name} differs from the checkpoint")

            self.camera_names = list(names)
            self.action_dim = width
            self._delta_mask = (
                None  # DeltaActionStep is serialized inside policy_postprocessor.json.
            )
            self._processor = PI05Processor.from_pretrained(
                checkpoint,
                tokenizer_name=str(checkpoint),
                num_images=len(self.camera_names),
                image_resize_backend=config.image_resize_backend,
                normalize_pixels=True,
                action_dim=self.action_dim,
                params_dtype=torch.float32,
            )
            self.action_dtype = self.metadata.get("action_dtype", "float32")
            if self.action_dtype not in ("float32", "float64"):
                raise ValueError("checkpoint action_dtype must be float32 or float64")
        else:
            # Format B: Metadata export (e.g. Tianji RLinf / OpenPI JAX).
            metadata = _read_metadata(checkpoint, config.metadata_file)
            self.metadata = metadata
            self.camera_names = list(metadata["camera_names"])
            self.action_dim = metadata["action_dim"]

            for name, value in (
                ("camera_names", config.camera_names),
                ("action_dim", config.action_dim),
            ):
                if value is not None and value != metadata[name]:
                    raise ValueError(f"processor.{name} differs from the checkpoint")

            self._delta_mask = torch.tensor(metadata["delta_mask"], dtype=torch.bool)
            stats = json.loads((checkpoint / "norm_stats.json").read_text())[
                "norm_stats"
            ]
            self._processor = PI05Processor(
                tokenizer_name=str(checkpoint),
                num_images=len(self.camera_names),
                dataset_stats={
                    "observation.state": stats["state"],
                    "action": stats["actions"],
                },
                action_dim=self.action_dim,
                normalize_pixels=True,
                image_resize_backend=config.image_resize_backend,
                params_dtype=torch.float32,
            )
            # Enforce high-precision float64 quantile normalization and exact epsilon parity.
            for pipeline in (
                self._processor.preprocessor,
                self._processor.postprocessor,
            ):
                pipeline.steps = [
                    replace(step, eps=metadata["norm_eps"], dtype=torch.float64)
                    if isinstance(step, (NormalizerStep, UnnormalizerStep))
                    else step
                    for step in pipeline.steps
                ]
            self.action_dtype = "float64"

        trained_tasks = self.metadata.get("trained_tasks")
        if trained_tasks is not None and (
            not isinstance(trained_tasks, list) or self.task not in trained_tasks
        ):
            raise ValueError("task is not one of the checkpoint's trained instructions")

        self.image_size = int(self._processor.image_size)

    def input_schema(self) -> dict[str, FeatureSpec]:
        """Generate the FeatureSpec mapping describing expected model inputs."""
        image = FeatureSpec((self.image_size, self.image_size, 3), "uint8")
        return {
            **{IMAGE_PREFIX + name: image for name in self.camera_names},
            STATE_KEY: FeatureSpec((self.action_dim,), "float32"),
        }

    def preprocess(self, inputs: Mapping[str, NDArray[Any]]) -> Any:
        """Run raw camera arrays, language instruction, and state through PI05Processor."""
        torch = self._torch
        # Copy NumPy arrays to avoid PyTorch mutating read-only wire buffers.
        images = [
            torch.from_numpy(np.array(inputs[IMAGE_PREFIX + name]))
            .permute(2, 0, 1)
            .unsqueeze(0)
            for name in self.camera_names
        ]
        state = torch.from_numpy(np.array(inputs[STATE_KEY])).unsqueeze(0)
        return self._processor.preprocess(
            {"images": images, "task": self.task, "state": state}
        )

    def postprocess(self, normalized: Any, state: NDArray[np.float32]) -> NDArray[Any]:
        """Unnormalize action predictions and anchor joint deltas to the request state.

        Args:
            normalized: Raw model output tensor from engine.step().
            state: The exact 1D float32 state vector received in the inference request.

        Returns:
            NumPy array of shape (action_horizon, action_dim) containing absolute targets.
        """
        torch = self._torch
        if self._delta_mask is None:
            # LeRobot path: postprocessor contains DeltaActionStep.
            actions = self._processor.postprocess(
                {"action": normalized, "state": np.array(state)}
            )
        else:
            # Metadata path: slice padding, unnormalize, then anchor delta masked dimensions.
            actions = self._processor.postprocess(normalized[..., : self.action_dim])
            anchor = torch.as_tensor(np.array(state), dtype=actions.dtype)
            actions = actions + torch.where(self._delta_mask, anchor, 0)[..., None, :]

        values = actions.detach().cpu().numpy().astype(self.action_dtype)[0]
        if not np.isfinite(values).all():
            raise ValueError("model produced non-finite actions")
        return values


class Pi05Policy:
    """Manages the resident PhyAI Pi0.5 inference engine and reproducible noise stream."""

    def __init__(self, config: Pi05PolicyConfig, processing: Pi05Processing) -> None:
        """Load Pi0.5 weights and initialize the PhyAI execution engine."""
        import torch
        from phyai.utils import load_config
        from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
        from phyai.kernel.config import KernelConfig
        from phyai.models.pi05.main_pi05 import PI05Args
        from phyai.models.pi05.scheduler_pi05 import PI05Request
        from phyai.models.pi05.configuration_pi05 import PI05Config
        from phyai import Engine, EngineArgs

        checkpoint = repository_path(config.checkpoint)
        model_config = load_config(checkpoint, PI05Config)
        if config.num_inference_steps is not None:
            # Override flow-matching Euler steps before engine initialization.
            model_config = replace(
                model_config, num_inference_steps=config.num_inference_steps
            )

        self.processing = processing
        self.num_inference_steps = int(model_config.num_inference_steps)
        self.horizon = int(model_config.chunk_size)
        expected = processing.metadata.get("action_horizon", self.horizon)
        if expected != self.horizon:
            raise ValueError("checkpoint metadata horizon differs from chunk_size")

        self._torch = torch
        self._request_type = PI05Request
        self._max_action_dim = int(model_config.max_action_dim)
        self._seed = config.seed
        self._generator = torch.Generator().manual_seed(config.seed)

        # Initialize the high-performance PhyAI CUDA inference engine.
        self._engine = Engine(
            EngineArgs(
                plugin="pi05",
                plugin_args=PI05Args(
                    checkpoint_dir=checkpoint, config=model_config, max_batch_size=1
                ),
                config=EngineConfig(
                    device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                    runtime=RuntimeConfig(
                        use_cuda_graph=config.use_cuda_graph,
                        num_threads=config.num_threads,
                    ),
                    kernel=KernelConfig(
                        config_path=str(repository_path(config.kernel_policy))
                    ),
                ),
            )
        )
        self._lock = Lock()
        self._close_requested = Event()
        self._closed = False

    def spec(self, metadata: Mapping[str, Any]) -> PolicySpec:
        """Assemble the complete PolicySpec describing model inputs, outputs, and metadata."""
        processing = self.processing
        return PolicySpec(
            inputs=processing.input_schema(),
            outputs={
                ACTIONS_KEY: FeatureSpec(
                    (self.horizon, processing.action_dim), processing.action_dtype
                )
            },
            metadata=dict(
                metadata,
                horizon=self.horizon,
                image_size=processing.image_size,
                camera_names=processing.camera_names,
                num_inference_steps=self.num_inference_steps,
            ),
        )

    def predict(self, inputs: Mapping[str, NDArray[Any]]) -> dict[str, NDArray[Any]]:
        """Synchronously execute preprocessing, Euler denoising, and postprocessing.

        Must be called while holding self._lock. Generates deterministic Gaussian noise
        for the full chunk horizon using the CPU generator stream.
        """
        torch = self._torch
        with self._lock:
            if self._close_requested.is_set():
                raise RuntimeError("Pi05Policy is closed")
            try:
                processed = self.processing.preprocess(inputs)
                # Sample initial Gaussian noise (shape: (1, horizon, max_action_dim)).
                noise = torch.randn(
                    1,
                    self.horizon,
                    self._max_action_dim,
                    generator=self._generator,
                    dtype=torch.float32,
                )
                with torch.inference_mode():
                    normalized = (
                        self._engine.step(
                            self._request_type(
                                pixel_values=processed.pixel_values,
                                input_ids=processed.input_ids,
                                lang_lens=processed.lang_lens,
                                noise=noise,
                            )
                        )
                        .detach()
                        .float()
                        .cpu()
                    )
                actions = self.processing.postprocess(normalized, inputs[STATE_KEY])
                return {ACTIONS_KEY: actions}
            finally:
                if self._close_requested.is_set():
                    self._close_engine()

    def reset_run(self) -> None:
        """Reset the CPU initial noise generator seed at the beginning of a client run."""
        with self._lock:
            if self._close_requested.is_set():
                raise RuntimeError("Pi05Policy is closed")
            self._generator.manual_seed(self._seed)

    def _close_engine(self) -> None:
        """Release underlying PhyAI engine resources once."""
        if not self._closed:
            self._engine.close()
            self._closed = True

    def close(self) -> None:
        """Request engine shutdown, safely releasing GPU resources."""
        self._close_requested.set()
        if self._lock.acquire(blocking=False):
            try:
                self._close_engine()
            finally:
                self._lock.release()
