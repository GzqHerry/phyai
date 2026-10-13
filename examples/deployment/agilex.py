r"""AgileX robot client (LIVE). The model runs in a separate resident process.

Run the following in Bash on the robot host. Driver scripts are site-local and
are not shipped by PhyAI. Commands here describe manual robot operation; offline
validation must not execute the driver, ROS inspection, or client commands.
No RTC, memory, overlapping prediction, homing, or motor-enable calls are added.

1. Install the robot client once
-------------------------------
Keep this deployment source at the same build as the phyai-robot wheel. From the
checkout root, use the native Python 3.10 profile (see ROBOT_ENVIRONMENTS.md)::

    cd "$HOME/phyai"
    WHEEL=/path/to/reviewed/phyai_robot-VERSION-py3-none-any.whl
    bash examples/deployment/environments/install_robot.sh \
        agilex-humble-py310-x86_64 "$WHEEL"

The existing .venv remains the Python 3.12 model environment. Never add Humble's
Python 3.10 packages to it. The robot client uses .venv-robot/bin/python directly.

2. Start the existing cameras and Piper drivers
----------------------------------------------
Check the physical emergency stop and workspace, and stop other command-producing
clients, including Enactive. The site's Piper launch defaults to auto_enable=true
and can enable motors. PhyAI does not enable, home, or clip trajectories.

Check CAN first. Expected interfaces are can_left and can_right, UP at 1 Mbit/s::

    ip -brief link show type can

Only if they are not configured, review the USB/CAN mapping before running the
site's existing setup; never reconfigure a live bus::

    cd "$HOME/piper_ros"
    bash can_config.sh

Check ``tmux ls`` and reuse healthy drivers instead of starting duplicates. The
following uses the existing camera/Piper wrappers and logs under .cache::

    mkdir -p "$HOME/phyai/.cache/deployment/agilex-live"
    tmux new-session -d -s phyai-agilex-live -n camera '
        set -o pipefail
        export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
        bash "$HOME/aloha_capture_app/scripts/start_camera.sh" 2>&1 |
            tee "$HOME/phyai/.cache/deployment/agilex-live/camera.log"
    '
    tmux set-window-option -t phyai-agilex-live:camera remain-on-exit on
    tmux new-window -d -t phyai-agilex-live -n piper '
        set -o pipefail
        export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
        bash "$HOME/aloha_capture_app/scripts/start_piper.sh" 2>&1 |
            tee "$HOME/phyai/.cache/deployment/agilex-live/piper.log"
    '
    tmux set-window-option -t phyai-agilex-live:piper remain-on-exit on

The camera wrapper sources ROS/camera overlays and starts the three Orbbec
cameras. The Piper wrapper sources ROS/driver overlays and runs
``ros2 launch piper start_two_piper.launch.py``. In another Bash shell, inspect::

    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
    ros2 topic list --no-daemon
    for topic in /camera_f/color/image_raw /camera_l/color/image_raw \
        /camera_r/color/image_raw /puppet/joint_left /puppet/joint_right \
        /joint_left_states /joint_right_states; do
        ros2 topic info "$topic" --verbose --no-daemon
    done

Before starting the client, command topics should have no competing publishers
and should have Piper subscribers. Check driver logs if observations time out.

3. Model terminal: Python 3.12, no ROS
-------------------------------------
Use a fresh shell without ROS overlays. The server loads the checkpoint once and
stays alive between client runs. To use 10 inference steps::

    cd "$HOME/phyai"
    source .venv/bin/activate
    export PHYAI_FLASHINFER_WORKSPACE_BYTES=268435456
    CKPT="$HOME/models/enactive-pi05-cube-pnp-rgb-step18155"
    python -m examples.deployment.serve_pi05 \
        --config examples/deployment/configs/pi05_agilex.yaml \
        "policy.checkpoint=$CKPT" \
        'task=Place the red cube on the green cube.' \
        policy.num_inference_steps=10

Omit the step override to use the checkpoint setting. Checkpoint, task, seed,
processor configuration, and inference steps all belong to this model process.

4. Robot terminal: native Python 3.10 + ROS Humble
-------------------------------------------------
Use another fresh shell. Printing configuration does not connect ROS or RPC::

    cd "$HOME/phyai"
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
    .venv-robot/bin/python -m examples.deployment.agilex --print-config

The following DOES SEND ROBOT COMMANDS, with execution horizon 20::

    .venv-robot/bin/python -m examples.deployment.agilex \
        deployment.execution_horizon=20

Default source/control frequencies remain 30/200 Hz and default horizon is 25.
No max_steps means continuous execution. Ctrl-C stops publishing and closes ROS,
then releases the model run. It does not disable motors or home the robot. A new
client resets the seed and discards one warmup before executing fresh predictions.
Stop the server with Ctrl-C separately. If a crashed client leaves the server
busy, restart the server manually; there is no automatic takeover or retry.

Data path
---------
Read RGB front/left/right cameras and JointState feedback. State order is left
six joint radians + gripper meters, then right. Publish joint0..joint6 to
/joint_left_states and /joint_right_states, one message per arm per control tick.
The server alone performs Pi05 preprocessing, inference and delta restoration.
The robot's unchanged RobotDeployment owns stop-and-refill and interpolation;
200 Hz commands never cross the inference RPC. No command is published while
waiting for the next model chunk. See AGILEX.md and ROBOT_ENVIRONMENTS.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from contextlib import ExitStack
from dataclasses import field, dataclass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from phyai_robot import RobotDeployment, ObservationAdapter

from .remote import RemotePolicyConfig
from .options import DeploymentConfig
from .configuration import parse_config
from .robots.agilex import (
    ACTION_SCHEMA,
    AgilexRobotConfig,
    make_agilex_robot,
    wait_for_observation,
)

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "agilex.yaml"


@dataclass
class AgilexDeploymentConfig:
    """Robot connection, RPC endpoint and control scheduling; no model settings."""

    max_steps: int | None = None
    robot: AgilexRobotConfig = field(default_factory=AgilexRobotConfig)
    remote: RemotePolicyConfig = field(default_factory=RemotePolicyConfig)
    deployment: DeploymentConfig = field(
        default_factory=lambda: DeploymentConfig(
            action_hz=30, execution_horizon=25, max_queued_actions=25
        )
    )

    def validate(self):
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or null")
        self.robot.validate()
        self.remote.validate()
        self.deployment.to_options()


def main(argv=None):
    """Compose the agilex client; configuration printing exits before any I/O."""
    config = parse_config(
        AgilexDeploymentConfig,
        default_config=DEFAULT_CONFIG,
        argv=argv,
        validate=AgilexDeploymentConfig.validate,
    )
    options = config.deployment.to_options(interpolate_keys=ACTION_SCHEMA)
    # ExitStack unwinds in reverse registration order: stop/close the robot
    # before releasing the model run. The server and its engine stay resident.
    with ExitStack() as resources:
        policy = config.remote.create()
        resources.callback(policy.close)
        robot = make_agilex_robot(config.robot)
        resources.callback(robot.close)
        spec = policy.begin_run(
            embodiment="agilex",
            observation_schema=robot.observation_schema,
            action_schema=robot.action_schema,
        )
        if options.execution_horizon > spec.metadata["horizon"]:
            raise ValueError("execution_horizon exceeds checkpoint chunk_size")
        robot.connect()
        observation = wait_for_observation(
            robot,
            timeout_s=options.observation_timeout_s,
            max_age_s=options.max_sample_age_s,
        )
        # Warmup consumes the same first noise draw as the original deployment.
        # Discard its chunk, then check sensor freshness again before running.
        started = time.monotonic()
        warmup = policy.warmup(observation)
        print(
            f"Warmup: {time.monotonic() - started:.3f}s; denoise_steps={spec.metadata['num_inference_steps']}; discarded {len(warmup.actions)} actions",
            flush=True,
        )
        wait_for_observation(
            robot,
            timeout_s=options.observation_timeout_s,
            max_age_s=options.max_sample_age_s,
        )
        print(
            f"AgileX: source={options.action_hz:g}Hz, control={options.control_hz:g}Hz, horizon={options.execution_horizon}; Ctrl-C stops publishing",
            flush=True,
        )
        # RemotePolicy already returns physical ActionChunks. The lightweight
        # adapter passes them through; real Pi05 transforms stay on the server.
        RobotDeployment(
            robot=robot, policy=policy, adapter=ObservationAdapter(), options=options
        ).run(max_steps=config.max_steps)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Deployment interrupted; robot cleanup requested.", file=sys.stderr)
        raise SystemExit(130) from None
