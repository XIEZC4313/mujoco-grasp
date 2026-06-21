# MuJoCo 5-DOF Robotic Arm Grasping Simulation

> AI-Driven Vision-Guided Pick-and-Place in MuJoCo

## Overview

A complete robotic manipulation system built in MuJoCo physics simulation. A 5-DOF robotic arm with a 2-finger parallel gripper autonomously detects colored blocks using computer vision, plans grasps via inverse kinematics, and executes pick-and-place operations.

**Built entirely through AI agent collaboration — no manual coding required.**

## Demo

Run the simulation:
```bash
cd src
python main.py
```

The robot will:
1. Scan the table and detect red, green, blue blocks
2. Identify the green block's position
3. Pick it up and place it at the target location
4. Print final positions of all blocks

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Color       │───>│ IK Solver    │───>│ Grasp Controller│
│ Detector    │    │ (Jacobian)   │    │ (Pick & Place)  │
└─────────────┘    └──────────────┘    └─────────────────┘
       │                   │                     │
       v                   v                     v
  ┌──────────────────────────────────────────────────┐
  │              MuJoCo Physics Engine                │
  │  ┌─────────┐  ┌──────────┐  ┌─────────────────┐ │
  │  │ 5-DOF   │  │ Gripper  │  │ Scene (Table +  │ │
  │  │ Arm     │  │ (2-finger)│ │  3 Blocks)      │ │
  │  └─────────┘  └──────────┘  └─────────────────┘ │
  └──────────────────────────────────────────────────┘
```

## Components

### Robot Model (`model/robot.xml`)
- 5-DOF articulated arm: base rotation, shoulder, elbow, wrist pitch, wrist roll
- 2-finger parallel gripper with Y-axis sliding fingers
- Motor actuators with gear ratios (10-50)

### Scene (`model/scene.xml`)
- Table at (0.35, 0, 0.15)
- 3cm colored cubes: red, green, blue
- Target placement zone

### IK Solver (`src/ik_solver.py`)
- Damped least squares (Levenberg-Marquardt)
- Joint limit clamping at each iteration
- Position-only and full pose solving modes

### Grasp Controller (`src/grasp_controller.py`)
- State machine: approach → grasp → lift → transport → place → retreat
- Incremental joint adjustments for smooth motion
- Gripper state management (open/close with motor force)

### Color Detector (`src/color_detector.py`)
- MuJoCo renderer for scene capture
- OpenCV HSV color segmentation
- Pixel-to-world coordinate mapping
- Debug image output with bounding boxes

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Block size | 3cm (0.03m) |
| Block mass | 10g |
| Gripper force | 50N (gear ratio 50) |
| Friction | 3.0 (block surface) |
| IK tolerance | 5mm |
| IK max iterations | 500 |

## Dependencies

```
mujoco>=3.9.0
numpy>=1.24.0
opencv-python>=4.8.0
```

## Project Structure

```
mujoco-grasp/
├── model/
│   ├── robot.xml          # 5-DOF arm + gripper MJCF model
│   └── scene.xml          # Scene with table and objects
├── src/
│   ├── main.py            # Main simulation entry point
│   ├── ik_solver.py       # Inverse kinematics solver
│   ├── grasp_controller.py # Pick-and-place controller
│   ├── color_detector.py  # HSV color detection
│   └── test_viewer.py     # Viewer test
├── submission/            # Hackathon submission materials
├── debug_detection.png    # Detection debug output
└── README.md
```

## License

MIT
