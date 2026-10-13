"""PhyAI pi0.5 model server policy: preprocessing, engine and postprocessing.

Everything here is defined by the checkpoint, never by a robot. Requests are
named model arrays: one RGB uint8 image per checkpoint camera ("image.<name>")
and the raw state vector ("state"), in the dataset's order and units. Replies
contain absolute "actions" with shape (horizon, action_dim) in the same
convention. Robot field names, units and calibration live in the robot client.

Two export formats are supported:
- lerobot style: policy_preprocessor.json / policy_postprocessor.json. Their
  pipelines own normalization and, when exported, delta-to-absolute decoding.
- metadata style: norm_stats.json plus deployment_metadata.json (or exactly
  one *_metadata.json) declaring cameras, action_dim, delta_mask and norm_eps.
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from threading import Lock, Event
from dataclasses import replace, dataclass
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
    """Local checkpoint and PhyAI engine settings."""

    # Required local export directory, resolved relative to the checkout root.
    checkpoint: str = MISSING
    # Operator-level kernel selection is kept separate from model weights.
    kernel_policy: str = "examples/deployment/configs/pi05_bf16.yaml"
    # Seed one CPU generator per run; do not reseed before every prediction.
    seed: int = 42
    # Flow-matching Euler updates per prediction, not the action horizon.
    # None keeps the checkpoint's value; changing it needs a new engine.
    num_inference_steps: int | None = None
    use_cuda_graph: bool = True
    # PhyAI runtime CPU threads.
    num_threads: int = 16

    def validate(self) -> None:
        """Check local paths and engine arguments without allocating GPU resources."""
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
    """Model-contract details that an export may not record itself."""

    # Camera names in model order. Required for lerobot-style exports; must
    # match a metadata-style export when given.
    camera_names: list[str] | None = None
    # Real action/state dimension. Required for lerobot-style exports.
    action_dim: int | None = None
    # Metadata JSON relative to the checkpoint; null discovers it.
    metadata_file: str | None = None
    # PIL matches clients that pre-resize with phyai_robot.images.
    image_resize_backend: str = "pil"

    def validate(self) -> None:
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
    """Load and check a metadata-style export without assuming an embodiment."""
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
    """Checkpoint-defined conversion between model arrays and engine tensors."""

    def __init__(
        self, checkpoint: Path, *, task: str, config: Pi05ProcessorConfig
    ) -> None:
        import torch
        from phyai_utils_tools.models.pi05 import PI05Processor
        from phyai_utils_tools.processing.steps import NormalizerStep, UnnormalizerStep

        if not task.strip():
            raise ValueError("task must be non-empty")
        self.task = task.strip()
        self._torch = torch
        self.metadata: dict[str, Any] = {}
        if (checkpoint / "policy_preprocessor.json").is_file():
            if config.camera_names is None or config.action_dim is None:
                raise ValueError(
                    "lerobot-style exports need processor.camera_names and "
                    "processor.action_dim"
                )
            self.camera_names = list(config.camera_names)
            self.action_dim = config.action_dim
            self._delta_mask = None  # The exported postprocessor decodes deltas.
            self._processor = PI05Processor.from_pretrained(
                checkpoint,
                tokenizer_name=str(checkpoint),
                num_images=len(self.camera_names),
                image_resize_backend=config.image_resize_backend,
                normalize_pixels=True,
                action_dim=self.action_dim,
                params_dtype=torch.float32,
            )
            self.action_dtype = "float32"
        else:
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
            # Exported statistics are named "state" and "actions"; the shared
            # processor expects canonical feature names.
            stats = json.loads((checkpoint / "norm_stats.json").read_text())
            stats = stats["norm_stats"]
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
            # Keep the export's float64 statistics and epsilon: float32
            # normalization can change tokenized state and predictions.
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
        self.image_size = int(self._processor.image_size)

    def input_schema(self) -> dict[str, FeatureSpec]:
        image = FeatureSpec((self.image_size, self.image_size, 3), "uint8")
        return {
            **{IMAGE_PREFIX + name: image for name in self.camera_names},
            STATE_KEY: FeatureSpec((self.action_dim,), "float32"),
        }

    def preprocess(self, inputs: Mapping[str, NDArray[Any]]) -> Any:
        torch = self._torch
        # Copy read-only wire buffers before handing them to torch.
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
        """Unnormalize and anchor relative targets to the request's raw state."""
        torch = self._torch
        if self._delta_mask is None:
            actions = self._processor.postprocess(
                {"action": normalized, "state": np.array(state)}
            )
        else:
            actions = self._processor.postprocess(normalized[..., : self.action_dim])
            # One anchor for the whole horizon: deltas are relative to the
            # observed pose, not increments to be summed over time.
            anchor = torch.as_tensor(np.array(state), dtype=actions.dtype)
            actions = actions + torch.where(self._delta_mask, anchor, 0)[..., None, :]
        values = actions.detach().cpu().numpy().astype(self.action_dtype)[0]
        if not np.isfinite(values).all():
            raise ValueError("model produced non-finite actions")
        return values


class Pi05Policy:
    """Own one BF16 PhyAI engine, its processing and a seeded noise stream.

    PolicyServer calls predict synchronously. A lock serializes the engine, its
    graph buffers and the generator. close may run while predict is still
    active; engine destruction is then deferred to the predicting thread.
    """

    def __init__(self, config: Pi05PolicyConfig, processing: Pi05Processing) -> None:
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
            # The plugin builds time embeddings and captures N Euler steps at
            # construction, so the override must happen before Engine(...).
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
        # CPU noise is independent of the engine's CUDA RNG state.
        self._seed = config.seed
        self._generator = torch.Generator().manual_seed(config.seed)
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
        torch = self._torch
        with self._lock:
            if self._close_requested.is_set():
                raise RuntimeError("Pi05Policy is closed")
            try:
                processed = self.processing.preprocess(inputs)
                # Fresh noise for the whole horizon, not just an executed prefix.
                noise = torch.randn(
                    1,
                    self.horizon,
                    self._max_action_dim,
                    generator=self._generator,
                    dtype=torch.float32,
                )
                with torch.inference_mode():
                    # Detach from reusable CUDA-graph output buffers.
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
                # A shutdown that timed out waiting for inference leaves engine
                # release to the thread that still holds the lock.
                if self._close_requested.is_set():
                    self._close_engine()

    def reset_run(self) -> None:
        """Restart the noise stream once per client run; warmup takes the first draw."""
        with self._lock:
            if self._close_requested.is_set():
                raise RuntimeError("Pi05Policy is closed")
            self._generator.manual_seed(self._seed)

    def _close_engine(self) -> None:
        if not self._closed:
            self._engine.close()
            self._closed = True

    def close(self) -> None:
        """Do not destroy CUDA resources underneath an active prediction."""
        self._close_requested.set()
        if self._lock.acquire(blocking=False):
            try:
                self._close_engine()
            finally:
                self._lock.release()
