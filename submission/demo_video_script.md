# Demo Video Script

## Duration: 60-90 seconds

---

### Scene 1: Opening (5s)
**Screen**: Title card with project name
**Text overlay**: "AI-Driven 5-DOF Robotic Arm — MuJoCo Pick-and-Place"

---

### Scene 2: Environment Overview (10s)
**Screen**: MuJoCo viewer showing the full scene
**Narration**:
"A 5-DOF robotic arm with a parallel gripper sits at the origin. Three colored blocks — red, green, and blue — rest on a table. The robot will detect the green block using vision, grasp it, and move it to a target location."

**Action**: Camera slowly orbits the scene to show all components

---

### Scene 3: Vision Detection (15s)
**Screen**: Split view — MuJoCo scene + debug detection image
**Narration**:
"The robot scans the scene using MuJoCo's renderer. OpenCV HSV color segmentation identifies each block. Red, green, and blue contours are detected and their world positions estimated."

**Action**: Show debug_detection.png with bounding boxes around each colored block

---

### Scene 4: Pick Phase (20s)
**Screen**: MuJoCo viewer, close-up on the arm
**Narration**:
"The IK solver computes joint angles to reach above the green block. The arm descends, the gripper closes with 50N force, and the block is secured."

**Action**:
1. Arm moves above green block (2s)
2. Arm lowers to block (2s)
3. Gripper closes on block (3s)
4. Arm lifts block (3s)
**Text overlay**: "IK Solve → Approach → Grasp → Lift"

---

### Scene 5: Transport and Place (15s)
**Screen**: MuJoCo viewer, wider shot
**Narration**:
"The base rotates toward the target position. The arm lowers, releases the block, and retreats."

**Action**:
1. Base rotates to target (3s)
2. Arm lowers (2s)
3. Gripper opens, block placed (3s)
4. Arm retreats (2s)
**Text overlay**: "Transport → Place → Release"

---

### Scene 6: Results (10s)
**Screen**: Final positions printed in terminal + MuJoCo scene
**Narration**:
"The green block has been successfully moved from its original position to the target location. All three blocks are visible in their final positions."

**Text overlay**:
```
green_block: [0.400, -0.120, 0.190]  ← target
Pick and place complete!
```

---

### Scene 7: Closing (5s)
**Screen**: Project title + key features
**Text overlay**:
- "5-DOF Arm + Parallel Gripper"
- "Vision-Guided (MuJoCo + OpenCV)"
- "Jacobian IK Solver"
- "Built with AI Agent Collaboration"

---

## Recording Tips
- Use OBS or Windows Game Bar to record the MuJoCo viewer window
- Run `main.py` and let it execute the full sequence
- Capture the terminal output for the "Results" scene
- Record at 30fps, 1080p for best quality
- The debug_detection.png is saved automatically to the project root
