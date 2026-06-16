import cv2
import numpy as np
from typing import List, Dict, Optional

class PoseService:
    def __init__(self):
        self.mp_pose = None
        self.pose = None

    def _lazy_init(self):
        if self.pose is not None:
            return
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        # Initialize Pose detector with model_complexity=2 for maximum accuracy on the server side
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def extract_landmarks(self, image_path: str) -> Optional[List[Dict[str, float]]]:
        """
        Extracts 33 pose landmarks from a given image.
        Returns a list of dicts containing x, y, z, and visibility coordinates,
        or None if no human pose is detected or if the image fails to load.
        """
        self._lazy_init()
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert BGR (OpenCV default) to RGB (MediaPipe expected format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image with the MediaPipe model
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
            
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(lm.visibility)
            })
            
        return landmarks

# Instantiate the global pose service
pose_service = PoseService()
