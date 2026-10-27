# ARX X5 2025 / AC One

This client uses the existing ARX `remote_slave` RobotStatus interface. It does
not start drivers, configure CAN, enable motors or move the arms to a ready
pose. Prepare the robot through its existing control application before running
the client. Only one application may publish policy targets at a time.

X5 2023 gripper scaling and `joint_control_v1` are not implemented here. They
need their own explicit codec and configuration.

## Data contract

| Field | Native source | Unit / order |
| --- | --- | --- |
| Left feedback | `/arm_slave_l_status`, `arx5_arm_msg/RobotStatus` | `joint_pos[0:6]`, then raw gripper `joint_pos[6]` |
| Right feedback | `/arm_slave_r_status`, same type | Same fixed order |
| Left target | `/arm_master_l_status`, same type | Six joint radians and raw gripper motor radians |
| Right target | `/arm_master_r_status`, same type | Same fixed order |
| Cameras | `/camera/camera_{h,l,r}/color/image_rect_raw/compressed` | JPEG/PNG RGB; head, left wrist, right wrist |

State and action vectors contain 14 values: left joints 1–6, left gripper,
right joints 1–6, right gripper. All coordinates are radians. Unlike the Piper
example, the gripper is not a distance in meters and is not normalized to
`[0, 1]`. The adapter does not clamp, rescale or add the observation to returned
actions. Joint-delta decoding belongs to the checkpoint's model processor.

Each camera has a separate decode worker and is letterboxed with PIL to the
model's image size. Set `robot.camera_transport=raw` and override camera topics
to use RGB/BGR `sensor_msgs/Image` instead. Compressed depth and H.264 are not
accepted by the JPEG/PNG decoder. Map model camera names explicitly when an
export uses names other than `base`, `left_wrist`, `right_wrist`.

## Independent robot environment

The `arx-jazzy-py312-x86_64` profile installs a Python 3.12 client environment.
ROS Jazzy and the compiled `arx5_arm_msg` package come from the host. NumPy 1.26.4
avoids replacing the NumPy ABI used by distribution ROS/OpenCV packages.
Pillow matches the model environment's 12.3.0 pin.

Build the wheel as described in [ROBOT_ENVIRONMENTS.md](ROBOT_ENVIRONMENTS.md),
then install it without synchronizing the model workspace:

```bash
bash examples/deployment/environments/install_robot.sh \
  arx-jazzy-py312-x86_64 /absolute/path/to/phyai_robot-VERSION-py3-none-any.whl
source /opt/ros/jazzy/setup.bash
source /path/to/arx-overlay/install/setup.bash
export PYTHONNOUSERSITE=1
.venv-robot/bin/python -m examples.deployment.arx --print-config
```

Printing configuration imports no ROS messages, connects no devices, and does
not query a model server. Plainly importing message classes also creates no
nodes; launching the client without `--print-config` does connect and execute.

## Checkpoint and model server

Use a converted PhyAI pi0.5 export with weights, tokenizer, `config.json`, and
native `policy_preprocessor.json` / `policy_postprocessor.json` pipelines and
statistics. An Enactive training checkpoint directory is not a converted export.
For ARX chunk-anchor checkpoints, add the original observation state to the
twelve arm joints only; both grippers remain absolute. Confirm this from the
source checkpoint metadata rather than assuming it for every ARX model.

Compare preprocessing and decoded actions on saved observations before physical
execution. [ENACTIVE_ALIGNMENT.md](../pi05/ENACTIVE_ALIGNMENT.md) describes the
existing Enactive export contract; its earlier numerical results do not validate
a new checkpoint automatically.

```bash
.venv/bin/python -m examples.deployment.serve_pi05 \
  --config examples/deployment/configs/pi05_arx.yaml \
  policy.checkpoint=/absolute/path/to/converted-arx-checkpoint \
  'task=Pour the water.'
```

The task belongs to the model server. Changing a robot-client option does not
change the instruction. The model stays loaded between client runs.

## Controller interfaces

`robot.controller_interface=external` is the default. The client sends stamped
RobotStatus targets to already-prepared remote-slave controllers. Stop ends
publication; it does not disable motors or change their retained target.

`robot.controller_interface=workbench` is for resident controllers exposing the
BodyPipeline workbench services. It performs these steps after model warmup:

1. Call both `health_check` services.
2. Call both `select_inference` services to close the policy gates.
3. Call both `remember_fold` services to open policy input using the existing
   folded references. Missing references fail activation; no home pose is guessed.

On stop, the adapter attempts `select_inference` on both sides, closing the gates
while preserving inference mode and the last motor target. It reports service
failures, including a partial handoff, and does not fall back to `enable`,
`reset` or capture mode. Cleanup before activation makes no controller calls.
Controller namespaces and service timeouts are configurable. These services
belong to the resident workbench controller, not every stock ARX driver.

The following command **starts physical execution**. Finish collection and stop
other policy publishers before using it; normal collection and this client must
not run concurrently:

```bash
.venv-robot/bin/python -m examples.deployment.arx \
  robot.controller_interface=workbench \
  remote.endpoint=tcp://127.0.0.1:5555
```

The default scheduler plays 25 targets at 30 Hz, with 200 Hz interpolated sends.
It sends no new targets while fetching the next prediction. Rate, execution
horizon, RPC, sensor-age and scheduling budgets use the shared deployment
configuration; the ARX adapter introduces no additional step/age thresholds.

## Offline timing

The offline benchmark loads a saved NPZ containing RGB uint8 HWC arrays named
`base`, `left_wrist`, `right_wrist`, plus a finite raw float state vector named
`state`. It creates no ROS nodes and sends no robot commands. It does use the
GPU and CPU model resources, so run it after collection if shared resources
could affect recording.

```bash
.venv/bin/python -m examples.deployment.benchmark_pi05 \
  policy.checkpoint=/absolute/path/to/converted-arx-checkpoint \
  observation=/absolute/path/to/observation.npz \
  output=.cache/arx/latency.json \
  'task=Pour the water.' warmup=3 iterations=20
```

The JSON separates warmup from steady-state median/p95 prediction latency.
Timing includes preprocessing, model execution, device-to-host output and
postprocessing, but excludes ROS and RPC. It is not a measurement of isolated
GPU kernels or physical control rate. Match checkpoint, input, task, precision,
denoising steps and CUDA Graph settings when comparing engines.

## Software checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES="" \
  .venv-robot/bin/python -m pytest -q -c phyai-robot/pyproject.toml \
  --confcutdir=examples/deployment/tests examples/deployment/tests/test_arx.py \
  examples/deployment/tests/test_benchmark.py
```

Tests cover raw-unit mapping, malformed feedback, image decoding, lazy startup,
partial controller activation and stop failures. They use fake ROS modules and
do not establish physical task success or controller-stop guarantees.
