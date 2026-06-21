# MuJoCo 5-DOF Robotic Arm Grasping Simulation

Multi-DOF robotic arm object grasping project built with MuJoCo physics simulator.

## Features

- **5-DOF Robotic Arm**: Base rotation, shoulder, elbow, wrist pitch, wrist roll
- **2-Finger Parallel Gripper**: Y-axis sliding fingers for side grasping
- **Inverse Kinematics**: Jacobian iterative method with joint limit clamping
- **Grasp Demo**: Automated grasp and lift sequence
- **Multiple Objects**: Cube, cylinder, sphere on a table

## Quick Start

```bash
cd F:\claude\mujoco-grasp\src
C:\Users\27206\AppData\Local\Programs\Python\Python312\python.exe main.py
```

## Project Structure

```
mujoco-grasp/
├── model/
│   ├── robot.xml          # 5-DOF arm + gripper MJCF model
│   └── scene.xml          # Scene with table and objects
├── src/
│   ├── ik_solver.py       # Inverse kinematics solver
│   ├── grasp_controller.py # Grasp controller
│   └── main.py            # Main simulation entry point
└── README.md
```

## Robot Configuration

| Joint | Type | Range (rad) | Description |
|-------|------|-------------|-------------|
| joint1 | Hinge | ±π | Base rotation |
| joint2 | Hinge | ±1.8 | Shoulder |
| joint3 | Hinge | ±3.0 | Elbow |
| joint4 | Hinge | ±2.0 | Wrist pitch |
| joint5 | Hinge | ±π | Wrist roll |

## Grasp Sequence

1. Open gripper
2. Move above object (IK)
3. Lower to object (IK)
4. Close gripper (motor force)
5. Lift object (incremental joint adjustment)
6. Hold position
7. Release gripper

## Technical Details

- **Physics Engine**: MuJoCo 3.9.0
- **IK Method**: Damped least squares with joint limit clamping
- **Gripper Control**: Motor actuators with gear ratio 50
- **Grasp Strategy**: Side grasp with incremental joint adjustment
