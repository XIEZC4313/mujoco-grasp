# Robothon Submission - MuJoCo 5-DOF Robotic Arm Grasping

## Project Title
AI-Driven 5-DOF Robotic Arm: Vision-Guided Pick-and-Place in MuJoCo

## One-Line Summary
A 5-DOF robotic arm with color vision that autonomously detects, grasps, and relocates colored blocks — built entirely by giving a repo to an AI agent.

## What It Does
This project demonstrates a complete robotic manipulation pipeline in MuJoCo physics simulation:

1. **Vision**: The robot uses MuJoCo's renderer + OpenCV HSV color detection to identify red, green, and blue blocks on a table
2. **Planning**: An IK solver (damped least squares / Jacobian iterative method) computes joint angles to reach target positions
3. **Execution**: A grasp controller sequences the full pick-and-place cycle — approach, grasp, lift, transport, place, retreat
4. **Feedback**: Ground-truth positions are compared with vision estimates for validation

## How AI Was Used
The entire project was built through human-AI collaboration:
- **Robot model (MJCF)**: AI designed the 5-DOF arm kinematic chain, gripper geometry, and scene layout
- **IK solver**: AI implemented Jacobian-based inverse kinematics with joint limit clamping
- **Grasp controller**: AI designed the pick-and-place state machine with incremental joint adjustments
- **Color detector**: AI integrated MuJoCo rendering with OpenCV for real-time object detection
- **Debugging**: AI diagnosed and fixed gripper geometry issues (finger contact mechanics, control signal mapping)

No manual coding was required from the human operator — just describing requirements and reviewing results.

## Technical Highlights

| Component | Detail |
|-----------|--------|
| Physics Engine | MuJoCo 3.9.0 |
| Robot | 5-DOF arm + 2-finger parallel gripper |
| IK Method | Damped least squares with joint limit clamping |
| Vision | MuJoCo renderer + OpenCV HSV color segmentation |
| Objects | 3cm cubes (red, green, blue) on a table |
| Language | Python (NumPy, OpenCV, MuJoCo) |

## Robot Configuration

| Joint | Range | Description |
|-------|-------|-------------|
| joint1 | +-180 deg | Base rotation |
| joint2 | +-103 deg | Shoulder |
| joint3 | +-172 deg | Elbow |
| joint4 | +-115 deg | Wrist pitch |
| joint5 | +-180 deg | Wrist roll |

## Grasp Sequence
1. Open gripper
2. Move above target (IK solve)
3. Lower to object surface (IK solve)
4. Close gripper (motor force, gear ratio 50)
5. Lift object (incremental joint adjustment)
6. Rotate base toward drop-off
7. Lower and release
8. Retreat to safe position

## Files
- `model/robot.xml` — 5-DOF arm + gripper MJCF model
- `model/scene.xml` — Scene with table, blocks, and target
- `src/ik_solver.py` — Inverse kinematics solver
- `src/grasp_controller.py` — Pick-and-place controller
- `src/color_detector.py` — HSV color detection
- `src/main.py` — Main simulation entry point
- `src/test_viewer.py` — Viewer test script
