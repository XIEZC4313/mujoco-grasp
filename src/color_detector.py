"""Color-based object detection using MuJoCo camera + OpenCV HSV."""

import numpy as np
import cv2
import mujoco


class ColorDetector:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Camera setup
        self.camera_name = "detect_cam"
        self._setup_camera()

        # MuJoCo rendering
        self.renderer = mujoco.Renderer(model, height=480, width=640)

        # HSV ranges for colors (tuned for MuJoCo rendering)
        self.color_ranges = {
            "red": [(0, 150, 150), (10, 255, 255)],
            "red2": [(170, 150, 150), (180, 255, 255)],
            "green": [(35, 150, 150), (85, 255, 255)],
            "blue": [(100, 150, 150), (130, 255, 255)],
        }

    def _setup_camera(self):
        """Add a top-down camera to the scene for detection."""
        # We'll use mjv_camera to look from above
        pass

    def render_scene(self):
        """Render the current scene and return RGB image."""
        self.renderer.update_scene(self.data)
        pixels = self.renderer.render()
        return pixels

    def detect_color(self, color_name="green"):
        """Detect objects of specified color in the scene.

        Args:
            color_name: Color to detect ("red", "green", "blue")

        Returns:
            center: (x, y) pixel coordinates of detected object center, or None
            bbox: Bounding box (x, y, w, h), or None
            mask: Binary mask of detected color
        """
        # Render scene
        img = self.render_scene()

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Get color range
        if color_name not in self.color_ranges:
            return None, None, None

        lower, upper = self.color_ranges[color_name]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # For red, combine two ranges
        if color_name == "red":
            lower2, upper2 = self.color_ranges["red2"]
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask, mask2)

        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, mask

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 50:  # Too small
            return None, None, mask

        # Get bounding box and center
        x, y, w, h = cv2.boundingRect(largest)
        center_x = x + w // 2
        center_y = y + h // 2

        return (center_x, center_y), (x, y, w, h), mask

    def pixel_to_world(self, pixel_pos, height=0.19):
        """Convert pixel coordinates to world coordinates.

        Uses simple projection based on camera parameters.

        Args:
            pixel_pos: (x, y) pixel coordinates
            height: World z-coordinate of the object (table height)

        Returns:
            world_pos: [x, y, z] world coordinates
        """
        # Camera is looking from above
        # Image center corresponds to world origin area
        # We need to map pixel coordinates to world coordinates

        img_h, img_w = 480, 640

        # Camera position (top-down view)
        cam_x, cam_y, cam_z = 0.35, 0.0, 0.6
        cam_fov = 60  # degrees

        # Convert pixel to normalized coordinates
        nx = (pixel_pos[0] - img_w / 2) / img_w
        ny = (pixel_pos[1] - img_h / 2) / img_h

        # Calculate world coordinates
        fov_rad = np.radians(cam_fov)
        scale = 2 * cam_z * np.tan(fov_rad / 2)

        world_x = cam_x + nx * scale
        world_y = cam_y + ny * scale
        world_z = height

        return np.array([world_x, world_y, world_z])

    def find_green_block(self):
        """Find the green block and return its world position.

        Returns:
            world_pos: [x, y, z] position of green block, or None
        """
        center, bbox, mask = self.detect_color("green")

        if center is None:
            return None

        world_pos = self.pixel_to_world(center)
        return world_pos

    def save_debug_image(self, filepath="debug_detection.png"):
        """Save debug image showing detection results."""
        img = self.render_scene()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for color in ["red", "green", "blue"]:
            center, bbox, mask = self.detect_color(color)
            if center and bbox:
                x, y, w, h = bbox
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_bgr, color, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(filepath, img_bgr)
        print(f"Debug image saved to {filepath}")
