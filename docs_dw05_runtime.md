# DW05 Runtime, Evaluation, and Demo Entrypoints

This repository keeps the DW05 runtime code in Dexbotic-native form and only
vendors the static assets that are directly needed by the RobotWin online demo.

- `dexbotic/`: the migrated DW05 model, dataset, exp, trainer, and shared deployment policy.
- `playground/online_demos/assets/`: URDF assets plus a `mesh/` subdirectory with the meshes referenced by those URDFs.

Dexbotic-native entrypoints:

- Real robot HTTP service: `hardware/dw05/real_robot_server.py`
- RoboTwin policy adapter: `playground/benchmarks/robotwin2/dw05_policy/`
- RoboTwin single-task launcher: `playground/benchmarks/robotwin2/eval_dw05_single.py`
- RoboTwin multi-task launcher: `playground/benchmarks/robotwin2/eval_dw05_manager.py`
- RoboTwin helper utilities: `script/dw/robotwin_eval_utils.py`
- WorldArena rollout: `playground/benchmarks/worldarena/eval_dw05.py`
- RobotWin online CLI/web demo: `playground/online_demos/robotwin_online_demo.py`

The full legacy external runtime tree is not vendored.  If an older script is needed
verbatim, use `dexbotic-dw0` as the reference and migrate only its direct
runtime dependencies.
