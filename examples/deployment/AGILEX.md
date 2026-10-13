# AgileX Pi05 deployment

The opening docstring in `agilex.py` contains the full camera/CAN/Piper startup
procedure and both launch commands. `ROBOT_ENVIRONMENTS.md` covers installation,
wheel identity, protocol lifecycle and Tianji startup.

The client now runs in native Python 3.10 / ROS Humble. The old system-Python
worker/socketpair bridge has been removed. Inference runs in a resident Python
3.12 server; stopping the client does not unload the model. Do not run the old
single-process command with model arguments on `agilex.py`: checkpoint, task,
seed and inference-step options now belong to `serve_pi05.py`.

## Unchanged numerical and control conventions

- Cameras: `/camera_f/color/image_raw`, `/camera_l/color/image_raw`,
  `/camera_r/color/image_raw`; original RGB uint8 pixels cross RPC unchanged.
- Feedback: `/puppet/joint_left` and `/puppet/joint_right`; reorder
  `sensor_msgs/msg/JointState` by `joint0` through `joint6`.
- State/actions: left six joint radians + gripper meters, then right.
- The existing Enactive processor performs resize/padding, normalization,
  tokenization and joint-only chunk anchoring. Grippers remain absolute.
- The server returns 50 targets; the client keeps 25 by default, at 30 Hz source
  and 200 Hz control. Horizon 20 is a client override. The shared
  `RobotDeployment` still implements stop-and-refill and interpolation.
- One write publishes both arms to `/joint_left_states` and
  `/joint_right_states`. There is no motor enable, homing, clipping or new
  controller-ownership mechanism. Stop means stop publication, not motor disable.

## Ownership

- `agilex.py`, `configs/agilex.yaml`: robot-client composition and timing.
- `serve_pi05.py`, `configs/pi05_agilex.yaml`: model-server composition.
- `adapters/agilex_pi05.py`: existing Pi05 mathematical transformations.
- `layouts/agilex.py`: physical schemas and pure helpers, with no driver imports.
- `robots/agilex.py`: compose two `Ros2Backend` instances into `CompositeRobot`.
  One subscribes to three cameras; the other receives both arms and publishes
  left then right commands. The framework owns nodes, caches, I/O threads,
  publication and cleanup.
- `robots/agilex_codecs.py`: bind six arm joints and one gripper to each
  JointState message. Generic image and named-joint decoders live in
  `phyai_robot.backends.ros2.codecs`.
- `phyai_robot.remote`: reusable inference transport, independent of embodiment.

For saved-data inference without any ROS, the standalone
`examples/pi05/run_enactive.py` is unchanged. Keep checkpoint, task, image/state
bytes, noise and inference steps identical when comparing implementations.
