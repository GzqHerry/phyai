# Robot / inference environment isolation

Run commands from the deployment checkout root. Start each process in its own
Bash shell, and set `CKPT` to the converted PhyAI checkpoint directory on the
model host. The robot host needs the matching deployment source/configs and
native environment; it does not need model weights.

Run two processes. The **robot client** owns native ROS/SDK dependencies, sensor
reads, the existing stop-and-refill queue, interpolation and actuator writes. The
**model server** owns the checkpoint, task, seed, Pi05 preprocessing, inference
and action restoration. The model stays loaded when a client exits.

```
Native robot Python                         PhyAI Python 3.12
Backend -> Observation -> RemotePolicy ----> adapter.to_request(observation)
                                             Pi05Policy.predict(request)
                                             adapter.to_actions(result, observation)
Backend <- RobotDeployment <- ActionChunk <---- one complete chunk per RPC
```

The second adapter call uses the **same observation** as the first, not newer
robot feedback. Control ticks do not cross RPC. Tianji's camera resize remains
in its robot backend; AgileX sends the original RGB pixels. There is no RTC,
prefetch, automatic inference retry, or motion recovery.

## What belongs where

| Component | Package / source | Process |
|---|---|---|
| `Observation`, `ActionChunk`, `PolicyAdapter` | `phyai-robot` | Either |
| `RemotePolicy`, `ObservationAdapter` | `phyai-robot` | Robot client |
| `PolicyServer` | `phyai-robot` | Model server |
| ROS nodes, threads, caches and publication | `phyai-robot.backends.ros2` | Robot client |
| Topic/QoS bindings, joint/gripper layout and controller modes | `robots/` | Robot client |
| Joint layouts and pure data helpers | `layouts/` | Either; no ROS imports |
| Concrete Pi05 adapters | `adapters/` | Model server |
| `Pi05Policy`, PhyAI engine | Model integration / PhyAI | Model server |

Installing `phyai-robot` on the model server does not install ROS or make the
engine depend on robot drivers. Its base dependencies are NumPy and a Python
3.10 exception-group backport. PyZMQ is the optional `zmq` extra. The existing
`ZmqBackend` is a robot I/O backend, not this inference protocol.

Both examples compose a camera `Ros2Backend` and a feedback/command
`Ros2Backend` into `CompositeRobot`. Camera decoding runs on a separate worker
from command submission. AgileX uses tuple field bindings because each
JointState message contains six arm joints plus a gripper. Both fields share
one receive timestamp and produce one outgoing message per arm. ROS resources
are owned by the framework; Tianji's mode/input service helper stays in the
robot integration layer.

## Supported robot profiles

| Profile | Native interpreter | ROS / overlays | Important pins |
|---|---|---|---|
| `agilex-humble-py310-x86_64` | `/usr/bin/python3`, 3.10 | Humble, `sensor_msgs` | NumPy 1.26.4 |
| `tianji-jazzy-py312-aarch64` | `/usr/bin/python3`, 3.12 | Jazzy, `marvin_msgs`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `std_srvs` | NumPy 2.5.3, Pillow 12.3.0, OpenCV 5.0.0.93 |

Each directory under `environments/` includes pinned inputs and a hash-locked
transitive dependency list. Both use PyZMQ 27.2.0 and OmegaConf 2.3.1. The Tianji
Pillow version matches the model environment used for the old camera geometry;
pixel equivalence must be rechecked before changing it. OpenCV is pinned too:
its NV12 decoder must not silently switch to the numerically different NumPy
fallback because the robot environment lacks a compatible OpenCV build.

These profiles cover Python/architecture combinations, not separate framework
branches. Python 3.8, other ROS releases, and arbitrary vendor SDK environments
are **not** certified by these profiles. Add a tested profile when needed.

## Configuration ownership and timing

| Setting | Model server | Robot client |
|---|---|---|
| Checkpoint, task, seed, inference steps, kernel policy | `policy.*`, `task` | Not accepted |
| Server bind / client connect address | `endpoint` | `remote.endpoint` |
| RPC waits | Not a control frequency | `remote.timeout_s=2`, `remote.warmup_timeout_s=120` |
| Model target spacing | Returned in ActionChunk | `deployment.action_hz` overrides it |
| Command submission frequency | Not configured here | `deployment.control_hz=200` |
| Number of source targets executed | Returns the full predicted chunk | `deployment.execution_horizon` |
| Robot topics, domain, controller modes | No robot connection | `robot.*` |
| Stop after a number of sends | Server remains resident | `max_steps`; null means until stopped |

The default robot configs are AgileX 30 Hz / horizon 25 and Tianji 25 Hz /
horizon 20. `policy.num_inference_steps=10` sets Euler updates inside a model
prediction; it does not mean 10 Hz or 10 executed actions. The client preserves
the existing stop-and-refill loop: execute a chunk prefix, stop publishing while
waiting for a fresh observation and the next prediction, then execute the next
prefix. Inference frequency therefore depends on model latency and the execution
horizon. There is no prediction prefetch during an executing chunk.

Camera and feedback publication rates are supplied by the external drivers.
Backends cache the latest received sample; `wait_for_observation` waits for all
required fields and enforces receive-age limits. It does not synchronize camera
capture times or require a different sample on every read. After warmup it checks
freshness again before the control loop starts.

AgileX raw camera dimensions must agree between `robot.camera_height/width` and
the model server's `camera_height/width`. Tianji reads processor image_size from
server metadata and uses that size in its native camera backend. The schema
handshake rejects mismatches before robot.connect().

## Build and distribute the same code

From a reviewed checkout, with `uv` available:

```bash
.venv/bin/python examples/deployment/environments/build_bundle.py
```

The output under `.cache/robot-isolation/dist/<source-hash>/` contains:

- a pure Python `phyai_robot-...-py3-none-any.whl`;
- `deployment-sources.tar.gz` with the corresponding examples and configs;
- `build-manifest.json` with the base commit, every source hash, and artifact
  SHA-256 hashes. Its local `0.0.0+source.<hash>` version is a review build, not
  a published release.

Transfer this **same wheel and matching examples** to both endpoints. Check the
hashes against the manifest. Examples are not contained in the wheel; extract
the source archive into the deployment checkout on a robot without the full
PhyAI workspace. Keep the `examples/deployment` directory layout. No engine
source or native extension is required there. Do not copy a virtual environment
between architectures.

## Install the robot environment independently

Do not use root-workspace `uv sync --package phyai-robot` for Python 3.10. The
root workspace still requires Python 3.12. Use the native interpreter and an
independent environment instead:

```bash
# From the root containing examples/deployment; select the appropriate profile.
WHEEL=/absolute/path/to/phyai_robot-VERSION-py3-none-any.whl
bash examples/deployment/environments/install_robot.sh \
  agilex-humble-py310-x86_64 "$WHEEL"
# On Tianji use tianji-jazzy-py312-aarch64 instead.
```

The script creates `.venv-robot` using `venv --without-pip
--system-site-packages`, installs the profile lock with `uv pip --python`, then
installs the wheel with `--no-deps`. This also works when the distribution does
not include `ensurepip`. It never installs into system Python, installs Torch,
builds PhyAI, or runs root `uv sync`. `--system-site-packages` makes host-provided
packages visible; it does not make the two environments identical. Installed
wheel identity is recorded in `.venv-robot/phyai-robot-wheel.sha256`.

Source ROS and custom overlays **in the client shell only**. Continue to use
`.venv-robot/bin/python` explicitly. On the model side install the same wheel in
the existing, prepared PhyAI environment without resolving heavy dependencies:

```bash
uv pip install --python .venv/bin/python --no-deps "pyzmq==27.2.0"
uv pip install --python .venv/bin/python --no-deps \
  --reinstall-package phyai-robot "$WHEEL"
```

For development only, independent editable installation is also possible:
`uv pip install --python .venv-robot/bin/python --no-deps -e ./phyai-robot`.
Acceptance testing uses the wheel, not just editable imports.

To regenerate the dependency locks deliberately (not during normal install):

```bash
uv pip compile --python-version 3.10 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --no-header \
  examples/deployment/environments/agilex-humble-py310-x86_64/requirements.in \
  -o examples/deployment/environments/agilex-humble-py310-x86_64/requirements.lock
uv pip compile --python-version 3.12 --python-platform aarch64-unknown-linux-gnu \
  --generate-hashes --no-header \
  examples/deployment/environments/tianji-jazzy-py312-aarch64/requirements.in \
  -o examples/deployment/environments/tianji-jazzy-py312-aarch64/requirements.lock
```

## Import-only checks (no ROS initialization or motion)

AgileX shell:

```bash
source /opt/ros/humble/setup.bash
.venv-robot/bin/python -c 'import rclpy; from sensor_msgs.msg import Image, JointState'
.venv-robot/bin/python -m examples.deployment.agilex --print-config
```

Tianji shell (set the actual local message workspace, without starting drivers):

```bash
source /opt/ros/jazzy/setup.bash
source "$TIANJI_MESSAGE_OVERLAY/install/setup.bash"
.venv-robot/bin/python -c 'import rclpy; from marvin_msgs.msg import Jointfeedback, JointcmdArm; from marvin_msgs.srv import Int'
.venv-robot/bin/python -m examples.deployment.tianji --print-config
```

Neither snippet calls `rclpy.init` or creates a node. The model configuration can
be printed in a shell without ROS:

```bash
.venv/bin/python -m examples.deployment.serve_pi05 --print-config
```

## Start AgileX: two terminals

Drivers must already be running. See the opening docstring in `agilex.py` for
the detailed camera/CAN/Piper procedure. **The robot client sends real commands.**

Model terminal, clean shell without ROS:

```bash
source .venv/bin/activate
export PHYAI_FLASHINFER_WORKSPACE_BYTES=268435456
python -m examples.deployment.serve_pi05 \
  --config examples/deployment/configs/pi05_agilex.yaml \
  "policy.checkpoint=$CKPT" \
  'task=Place the red cube on the green cube.' policy.num_inference_steps=10
```

Robot terminal:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
.venv-robot/bin/python -m examples.deployment.agilex \
  deployment.execution_horizon=20
```

The original defaults remain 30 Hz source, 200 Hz control, horizon 25. The
explicit example above uses horizon 20. Model steps default to the checkpoint
value when not overridden. No `max_steps` means run until Ctrl-C.

## Start Tianji: two terminals

Model terminal, prepared PhyAI shell without ROS:

```bash
source .venv/bin/activate
export PHYAI_FLASHINFER_WORKSPACE_BYTES=268435456
python -m examples.deployment.serve_pi05 \
  --config examples/deployment/configs/pi05_tianji.yaml \
  "policy.checkpoint=$CKPT" 'task=Plug in the Ethernet cable' \
  policy.num_inference_steps=10
```

If the installed Triton bundles a `ptxas` that cannot compile the device's target,
use a compatible **already-installed** CUDA compiler, for example
`export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas`.
This is a model-environment setting, not a robot dependency change.

Robot terminal (**does invoke controller services and send commands**):

```bash
source /opt/ros/jazzy/setup.bash
source "$TIANJI_MESSAGE_OVERLAY/install/setup.bash"
.venv-robot/bin/python -m examples.deployment.tianji \
  deployment.execution_horizon=20
```

Defaults remain 25 Hz source, 200 Hz control, horizon 20. Mode/input order stays:
connect -> fresh observation -> one discarded warmup -> fresh observation ->
ready -> mode -> optional velocity ratio -> input=3 -> deployment. Stop requests
input=0 through the existing control helper before the run is released.

## Run and protocol lifecycle

The schema handshake runs before opening robot hardware. Protocol version 1
checks embodiment, field set, shape, dtype, units, names/order and frame. Model
metadata includes horizon, task, seed, inference steps, and Tianji image size.
Run and monotonically increasing request IDs reject mismatched/stale replies.
NumPy bytes, shape and dtype travel in multipart frames; JSON carries metadata.
No pickle, ROS message or Torch tensor is transmitted. Receive timestamps stay
in the robot's clock domain; the server does not compare clocks.

One server owns one active run. `begin` resets the original CPU generator;
it does not run inference. The client then sends exactly one `warmup` with its
first observation, discards the returned chunk, and sends normal predictions in
order. No additional model call is inserted by the server. The warmup result is not placed in the queue.
A robot exit stops/closes robot resources first, then sends `end`. The engine is
not unloaded. Each request socket is created, used and closed by its caller;
warmup on the main thread and prediction on the worker do not share a socket.

Timeouts are bounded (`remote.timeout_s`, `remote.warmup_timeout_s`). A failed
prediction invalidates the client; it is not retried and motion does not resume
automatically. After an unclean disconnect leaves the server busy, restart it
manually. There are no leases or automatic takeovers.

The default endpoint binds loopback. **This protocol has no authentication or
encryption.** Do not expose it on a public network; use a trusted network or SSH
tunnel if the robot and GPU are separate hosts. Protocol compatibility is not
permission to mix arbitrary framework releases: the initial supported setup
uses one source-identified wheel on both sides plus separately pinned host
profiles. Dependency or framework upgrades require repeating the offline tests.

The protocol operations are intentionally small:

| Operation | State requirement | Effect |
|---|---|---|
| `info` | Any server state | Read schema/model metadata without reserving a run |
| `begin` | No active run | Validate contract, reset RNG, assign a run ID |
| `warmup` | Active run, not warmed | Run once and return the chunk for the client to discard |
| `predict` | Active, warmed run | One adapter/policy/adapter call, one complete chunk |
| `end` | Matching active run | Release ownership without unloading the model |

The JSON header includes protocol/run/request identifiers and array descriptors.
Remaining frames are raw C-order array bytes, with dtype byte order and shape
preserved. Each ActionChunk field uses a leading horizon dimension. Only numeric
and boolean arrays are supported; total message size is bounded at 128 MiB.
A model/validation error returns an error reply while retaining run ownership.
Normal cleanup can still send `end`; an abandoned run requires manual restart.

## Review and offline checks

Read `tianji.py` or `agilex.py` for startup/cleanup order, `robots/` for backend
composition, `serve_pi05.py` for model ownership, then `phyai_robot.remote` for
RPC and `phyai_robot.backends.ros2` for topic ownership. Concrete model adapters
remain in `adapters/`, and physical layouts remain in `layouts/`.

Print each entry's configuration without opening ROS or loading model weights:

```bash
.venv-robot/bin/python -m examples.deployment.agilex --print-config
.venv-robot/bin/python -m examples.deployment.tianji --print-config
.venv/bin/python -m examples.deployment.serve_pi05 \
  --config examples/deployment/configs/pi05_tianji.yaml --print-config
```

Run framework tests in the source checkout using the native environment (pytest
is a development dependency and is not part of the robot runtime profile):

```bash
uv pip install --python .venv-robot/bin/python pytest
.venv-robot/bin/python -m pytest -q -c phyai-robot/pyproject.toml \
  --confcutdir=phyai-robot phyai-robot/tests
```

This configuration tests the source package and avoids the root CUDA test
bootstrap. To test the installed wheel instead, use a separate pytest config
without `pythonpath=src`; `python -c 'import phyai_robot; print(phyai_robot.__file__)'`
with that environment's interpreter verifies the import location.

The 2026-10-12 acceptance run passed 137 framework tests on native Python 3.10
x86_64, model Python 3.12 x86_64, and native Python 3.12 aarch64. Saved-data AgileX
backend replay matched 130 arrays and 3,010 mock publications across horizons
25/20; Tianji mock replay matched 18 arrays and 80 publications. Earlier saved-data
model/RPC audits checked pixels, state/tokens, noise, model outputs, restored
actions and interpolation with exact byte comparisons. These were offline tests;
site-specific fixtures/checkpoints and logs stay outside the public repository.

## Validation boundary

Numerical equality and control order are testable with saved data, a real model
and mock backends. RPC adds wall-clock latency, so unchanged numbers do not prove
identical timing or physical behavior. Offline tests must not initialize a real
ROS context, start drivers, read live sensors or call services. Actual robot
acceptance is a separate operator-controlled step.
