# Robot / inference environment isolation

Run commands from the deployment checkout root, each process in its own shell.
The robot host needs the matching deployment sources and its native ROS
environment; it does not need model weights.

There are two processes. The **model server** owns the checkpoint, task, seed,
pi0.5 preprocessing, inference and action decoding. It knows nothing about any
robot and stays loaded between client runs. The **robot client** owns native
ROS dependencies, sensor reads, the mapping between robot fields and model
vectors, stop-and-refill scheduling, interpolation and actuator writes.

```
Native robot Python                             PhyAI Python 3.12
Backends -> Observation -> VectorAdapter
            images + state  ---- RemotePolicy ---->  preprocess -> engine -> postprocess
            Action chunk  <---- actions (H, D) ----  absolute targets in dataset units
RobotDeployment (StopAndRefill, 200 Hz interpolation) -> Backends
```

Control ticks never cross the RPC. The server decodes relative actions with the
raw state it received in the same request, so it never needs newer feedback.

## What belongs where

| Component | Source | Process |
|---|---|---|
| `Observation`, `ActionChunk`, `VectorAdapter`, `StopAndRefill` | `phyai-robot` | Robot client |
| `RemotePolicy`, `run_remote` | `phyai_robot.remote` | Robot client |
| `PolicyServer`, `PolicySpec` | `phyai_robot.remote` | Model server |
| ROS nodes, threads, caches, service calls | `phyai_robot.backends.ros2` | Robot client |
| Topics, QoS, field layout, calibration, controller enable | `robots/<robot>.py` | Robot client |
| Shared client config and entry point | `client.py` | Robot client |
| Checkpoint processing and engine | `pi05.py`, `serve_pi05.py` | Model server |

Each robot module defines its configuration, schemas, codecs, a `VectorAdapter`
and a `build(config, spec)` function that returns an `Embodiment`. The model
server contains no robot code: adding a robot means adding a robot module and a
config, not changing the server.

| Robot | Model inputs | Action row |
|---|---|---|
| AgileX | `image.front`, `image.left`, `image.right`, 14-D state | left 6 joints (rad), left gripper (m), right 6 joints, right gripper |
| Tianji | `image.head_left`, `image.left_wrist`, `image.right_wrist`, 16-D state | left 7 joints (rad), left gripper, right 7 joints, right gripper (normalized) |

Tianji gripper state is the feedback position mapped from
`robot.gripper_limits_rad` to [0, 1] on the robot; gripper commands are clamped
to [0, 1]. Cameras are decoded and letterboxed to the model's image size on the
robot with the same PIL arithmetic as the server.

### Tianji cameras

`robot.camera_source` selects how Tianji images reach the client:

| Source | Default topic | Processing |
|---|---|---|
| `mosaic` | `/quad_tile/compressed_undistorted` | One H.264 decoder; selected tiles share a frame and receive timestamp |
| `raw` | `/camera/<name>/image_nv12_undistorted` | One NV12 decoder worker per camera |

Topics, mosaic grid and tile coordinates are configurable. Compressed streams
reduce DDS bandwidth compared with full-resolution raw images. Measure delivery
latency and frame loss on the deployment host before choosing freshness limits.

Every H.264 access unit is decoded in order. `robot.mosaic_max_rate_hz` limits
RGB conversion without skipping decoder reference updates. A missing reference
picture can damage subsequent frames without an FFmpeg error. The decoder
checks `frame_num` continuity and withholds output until the next IDR after a
detected gap. Cached samples keep aging during recovery. Freshness is checked
before the next inference; it does not interrupt an already executing chunk.
Recovery time depends on the source encoder's keyframe interval.

`robot.camera_size` optionally resizes each complete tile to `[height, width]`
before letterboxing. Set it to reproduce the checkpoint's recorded image
geometry. For example, a recording stretched to 960x540 needs `[540, 960]`:
its content becomes 224x126 with 49 black rows above and below in a 224x224 model
input. The default `null` preserves the decoded tile's aspect ratio. Deployment
does not apply training-time random crops, rotations or color jitter.

## Checkpoint processing and external processors

Use `examples.deployment.serve_pi05` for both built-in and external processing.
Prefer serialized pipelines for transforms that `phyai-utils-tools` supports:
`PI05Processor.from_pretrained()` loads `policy_preprocessor.json` and
`policy_postprocessor.json` directly from the checkpoint. Keep `processor.factory`
null for these exports. External factories cover processing that cannot yet be
represented by registered steps.
`processor.factory: null` selects `Pi05Processing`, the deployment wrapper around
`phyai_utils_tools.models.pi05.PI05Processor`. The wrapper owns the input schema,
checkpoint statistics and decoding of action targets. The engine consumes its
processed tensors and does not infer training conventions from weights.

For an external processor, set the factory in the server YAML:

```yaml
policy:
  checkpoint: /path/to/export
processor:
  factory: /path/to/processors/dataset_processing.py:DatasetProcessing
  options: {}
```

An installed Python module also works: `my_processors.dataset:DatasetProcessing`.
A processor stored alongside the checkpoint can use
`${policy.checkpoint}/dataset_processing.py:DatasetProcessing`. Relative file
paths resolve from the checkout root, just like `policy.checkpoint`; the selected
YAML's location and the shell's working directory do not change that meaning.
`--print-config` prints the selection without importing external code or loading
the model. Normal startup imports only the factory explicitly named by the YAML.

The factory receives `checkpoint`, keyword arguments `task` and `config`, and
any `processor.options` as extra keyword arguments. It must return a
`Pi05Processing` instance, usually a subclass. This example adds a constant input
coordinate before the original normalization:

```python
import numpy as np
from examples.deployment.pi05 import Pi05Processing

class DatasetProcessing(Pi05Processing):
    def preprocess(self, inputs):
        values = dict(inputs)
        values["state"] = np.array(inputs["state"], copy=True)
        for index, value in self.metadata.get("state_constants", {}).items():
            if self.metadata["delta_mask"][int(index)]:
                raise ValueError("Do not replace a joint delta anchor")
            values["state"][int(index)] = value
        return super().preprocess(values)
```

A subclass can also override `postprocess(normalized, raw_state)`. It must return
finite absolute targets with shape `(action_horizon, action_dim)` in the dataset's
units. The same raw request state anchors every action in a chunk. The factory
also supplies the schema and metadata through the inherited interface; do not
pass the lower-level `PI05Processor` directly to `Pi05Policy`.

External source files need not be installed or copied into the checkout or
virtual environment. They run inside the server process, so dependencies must
be available to that interpreter. Use relative imports such as `from .helpers
import transform` for sibling files. The loader gives each source path a private
package namespace and leaves `sys.path` unchanged. Restart the server after
editing a loaded processor. Selecting another Python environment is a separate
process boundary, not a factory setting.

### Where training conventions are stored

The built-in wrapper supports two export layouts:

| Export layout | Configuration | Processing |
|---|---|---|
| Metadata export | `deployment_metadata.json` and `norm_stats.json` | Quantile state/action normalization; per-dimension delta mask |
| LeRobot pipeline export | `policy_preprocessor.json`, `policy_postprocessor.json` and their statistics sidecars | The serialized steps select normalization and action decoding |

For metadata exports, `processor.metadata_file` selects an alternative metadata
JSON. Otherwise the wrapper prefers `deployment_metadata.json`, then requires
exactly one `*_metadata.json`. The metadata declares `camera_names`, `action_dim`,
`delta_mask` and `norm_eps`; optional `action_horizon` must match the model config.
`norm_stats.json` stores `norm_stats.state` and `norm_stats.actions`. This path
uses q01/q99 normalization with float64 statistics and the declared epsilon.
It does not select mean/std normalization from the presence of those numbers.

After inverse normalization, a `true` delta-mask element adds the corresponding
raw state value; a `false` element is already an absolute target. For a model
trained entirely on absolute targets, set every element to `false`. For joint
deltas with absolute grippers, mark only the joint dimensions `true`. This mask
assumes deltas relative to the observation at the start of the chunk. It cannot
decode incremental deltas or end-effector rotations by itself.

Pipeline exports load the two serialized pipelines. Their normalizer and
unnormalizer steps record the normalization mode and epsilon, with statistics
in the referenced `.safetensors` files. Non-default statistic precision is saved
as `dtype`, for example `"float64"`; configs that omit it retain float32 behavior.
A `state_constants_processor` step can replace selected raw state coordinates
before normalization. Its `values` map contains coordinate indices and constants,
so this training convention stays in the model's pipeline rather than an external
server wrapper. These optional extensions require a PhyAI version that supports
them; the JSON layout alone does not guarantee every upstream LeRobot step is
available.

A delta model must include
the matching absolute-action decoding step in its postprocessor; the wrapper
does not add another delta transform on that path. The wrapper reads camera names and action dimension from deployment metadata
when present; otherwise set `processor.camera_names` and `processor.action_dim`.
Metadata may also declare `action_dtype` (`float32` by default, or `float64`) for
the returned NumPy array and wire schema. It does not override the pipeline's
normalization or apply a second delta transform. Only steps supported by
`phyai-utils-tools` can be loaded. A different training transform needs its
matching exported step or an external processing subclass.

For padded model actions and statistics covering only real action dimensions,
put `slice_action_step` before `unnormalizer_processor` in the serialized
postprocessor. Follow it with `delta_action_processor` when the training labels
are relative; omit the delta step for absolute labels. The same raw state anchors
every target in the chunk. Keep state normalization out of the postprocessor.

A complete serving bundle therefore contains converted model weights,
`config.json`, local tokenizer files, matching normalization statistics and
processing metadata or pipelines. A raw training checkpoint with weights alone
cannot establish action units, dimension order, camera geometry or normalization.
The server YAML still selects the task and runtime settings; the robot client
must map its sensors into the advertised schema using the training conventions.

## Supported robot profiles

| Profile | Native interpreter | ROS / overlays | Important pins |
|---|---|---|---|
| `agilex-humble-py310-x86_64` | `/usr/bin/python3`, 3.10 | Humble, `sensor_msgs` | NumPy 1.26.4, Pillow 12.3.0 |
| `tianji-jazzy-py312-aarch64` | `/usr/bin/python3`, 3.12 | Jazzy, `marvin_msgs`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `std_srvs` | NumPy 2.5.3, Pillow 12.3.0, PyAV 18.1.0, OpenCV 5.0.0.93 |

Each directory under `environments/` holds pinned inputs and a hash-locked
dependency list; both use PyZMQ 27.2.0 and OmegaConf 2.3.1. Pillow matches the
model environment so robot-side letterboxing equals the server's. PyAV decodes
the Tianji H.264 mosaic. NV12 decoding (`camera_source: raw`) requires OpenCV
and fails instead of falling back to other colors.

## Configuration ownership

| Setting | Model server | Robot client |
|---|---|---|
| Checkpoint, task, seed, inference steps, kernel policy | `policy.*`, `task` | Not accepted |
| Camera names / action dimension of lerobot-style exports | `processor.*` | Not accepted |
| Bind / connect address | `endpoint` | `remote.endpoint` |
| Release an abandoned run | `lease_timeout_s` (60 s) | Not configured |
| RPC waits | Not configured | `remote.timeout_s`, `remote.warmup_timeout_s` |
| Target consumption rate | Not configured | `schedule.action_hz` |
| Targets executed per prediction | Returns the full horizon | `schedule.execution_horizon` |
| Command rate (interpolation density) | Not configured | `control.control_hz` |
| Topics, QoS, domain, controller mode, calibration | Not accepted | `robot.*` |
| Stop after a number of sends | Not configured | `max_steps`; null runs until stopped |

Defaults: AgileX 30 Hz / horizon 25, Tianji 25 Hz / horizon 20, both at 200 Hz
control. `execution_horizon / action_hz` is how long each chunk executes (0.8 s
for Tianji). `policy.num_inference_steps` is the number of Euler steps inside a
prediction, not a rate. After each chunk the robot pauses while the client takes
a fresh observation and the server predicts.

`robot.domain_id` selects the ROS domain explicitly (Tianji's Apex stack uses
24); null uses `ROS_DOMAIN_ID` from the shell. Discovery range is not changed by
the client; export it in the shell when needed.

Print the merged configuration of any entry point without opening ROS, RPC or
the model:

```bash
.venv-robot/bin/python -m examples.deployment.tianji --print-config
.venv-robot/bin/python -m examples.deployment.agilex --print-config
.venv/bin/python -m examples.deployment.serve_pi05 \
  --config examples/deployment/configs/pi05_tianji.yaml --print-config
```

## Build and distribute the same code

From a reviewed checkout, with `uv` available:

```bash
.venv/bin/python examples/deployment/environments/build_bundle.py
```

The output under `.cache/robot-isolation/dist/<source-hash>/` contains a pure
Python `phyai_robot-...-py3-none-any.whl`, `deployment-sources.tar.gz` with the
matching examples and configs, and `build-manifest.json` with the base commit,
source hashes and artifact SHA-256 values. Transfer the same wheel and examples
to both endpoints and check the hashes. Do not copy a virtual environment between
architectures.

## Install the robot environment independently

The root workspace requires Python 3.12, so install the robot client with the
native interpreter instead of `uv sync`:

```bash
WHEEL=/absolute/path/to/phyai_robot-VERSION-py3-none-any.whl
bash examples/deployment/environments/install_robot.sh \
  tianji-jazzy-py312-aarch64 "$WHEEL"   # or agilex-humble-py310-x86_64
```

The script creates `.venv-robot` with `--system-site-packages`, installs the
profile lock, then the wheel with `--no-deps`. It never installs Torch, builds
PhyAI or touches system Python. Source ROS and message overlays in the client
shell only. On the model side, `uv sync --extra deployment` (or installing the
same wheel with `--no-deps`) provides `phyai-robot[zmq,images]`.

To regenerate a lock deliberately:

```bash
uv pip compile --python-version 3.12 --python-platform aarch64-unknown-linux-gnu \
  --generate-hashes --no-header \
  examples/deployment/environments/tianji-jazzy-py312-aarch64/requirements.in \
  -o examples/deployment/environments/tianji-jazzy-py312-aarch64/requirements.lock
```

## Start Tianji: two terminals

Model terminal, PhyAI environment without ROS:

```bash
source .venv/bin/activate   # also puts ninja on PATH for FlashInfer's JIT
export PHYAI_FLASHINFER_WORKSPACE_BYTES=268435456
export TRITON_PTXAS_BLACKWELL_PATH="$(command -v ptxas)"
python -m examples.deployment.serve_pi05 \
  --config examples/deployment/configs/pi05_tianji.yaml \
  "policy.checkpoint=$CKPT" 'task=Plug in the Ethernet cable'
```

On a GPU whose target is unsupported by Triton's bundled assembler, put a
compatible CUDA toolkit on PATH and set `TRITON_PTXAS_BLACKWELL_PATH` to its
`ptxas`. Check that the command above resolves to an executable. This setting
belongs to the model process. If FlashInfer cannot find `ninja`, activate the
model environment before launching.

Robot terminal. **This calls controller services and sends commands.**

```bash
source /opt/ros/jazzy/setup.bash
source /opt/kernelmind/apex/install/setup.bash   # marvin_msgs
.venv-robot/bin/python -m examples.deployment.tianji
```

Order: read model spec -> build robot -> reserve run -> connect -> fresh
observation -> discarded warmup -> fresh observation -> `set_ready` ->
`set_mode` -> optional `set_vel_ratio` -> `set_input(3)` -> control. Stopping
requests `set_input(0)` within `robot.service_timeout_s`, then releases the run.
The client never homes the arms or restores the previous mode.

## Start AgileX: two terminals

Drivers must already be running; see `AGILEX.md`. Model terminal:

```bash
source .venv/bin/activate
python -m examples.deployment.serve_pi05 \
  --config examples/deployment/configs/pi05_agilex.yaml \
  "policy.checkpoint=$CKPT" policy.num_inference_steps=10
```

Robot terminal. **This sends commands.**

```bash
source /opt/ros/humble/setup.bash
.venv-robot/bin/python -m examples.deployment.agilex
```

## Run and protocol lifecycle

Protocol version 2 carries one JSON frame plus raw C-order array frames; no
pickle, ROS message or Torch tensor crosses the wire, and messages are bounded
at 128 MiB. `info` returns the model spec; `begin` reserves the server and resets
its noise stream; exactly one `warmup` follows, whose result the client checks
and discards; `predict` returns one action matrix; `end` releases the run. Run
and increasing request IDs reject stale or replayed replies. Both ends validate
every array against the spec.

| Operation | State requirement | Effect |
|---|---|---|
| `info` | Any | Read the spec without reserving a run |
| `begin` | No active run | Reset the noise stream, assign a run ID |
| `warmup` | Active run, not warmed | One prediction for the client to discard |
| `predict` | Active, warmed run | One prediction |
| `end` | Matching run | Release ownership; the model stays loaded |

A failed prediction invalidates the client; requests are never retried and
motion never resumes automatically. A run without requests for
`lease_timeout_s` is released, so a crashed client does not require a server
restart. **The protocol has no authentication or encryption.** Bind loopback
or use a trusted network or SSH tunnel between hosts.

## Offline checks

```bash
uv pip install --python .venv-robot/bin/python pytest
.venv-robot/bin/python -m pytest -q -c phyai-robot/pyproject.toml \
  --confcutdir=phyai-robot phyai-robot/tests
.venv/bin/python -m pytest -q phyai-robot/tests examples/deployment/tests
```

These tests use in-memory ROS and mock backends. They never initialize a real
ROS context, read live sensors or call services. Actual robot acceptance is a
separate, operator-controlled step.
