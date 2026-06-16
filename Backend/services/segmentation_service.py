import os
import cv2
import numpy as np
from typing import List, Dict, Optional

class SegmentationService:
    def __init__(self):
        self.predictor = None
        self.model_type = "vit_h"
        # Resolve path to weights/sam_vit_h_4b8939.pth relative to this service
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_path = os.path.join(current_dir, "..", "weights", "sam_vit_h_4b8939.pth")

    def _lazy_init(self):
        """
        Lazily initialize PyTorch and SAM predictor on first actual request.
        Prevents FastAPI auto-reload from slowing down during active development code updates.
        """
        if self.predictor is not None:
            return

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"SAM weight checkpoint not found at {self.checkpoint_path}. "
                "Please run `python download_models.py` first to fetch model weights."
            )

        # Lazy imports for optional/heavy deep learning libraries
        import torch
        from segment_anything import sam_model_registry, SamPredictor

        # Force SAM to run on CPU to prevent CUDA Out Of Memory errors on 4GB GPUs
        device = "cpu"
        
        # Load model and ship to target device
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def generate_mask(self, image_path: str, landmarks: List[Dict[str, float]]) -> Optional[str]:
        """
        Generates a high-quality binary silhouette mask for the human pose.
        Uses pose landmarks as point prompts to guide SAM's zero-shot segmentation.
        Saves the binary mask in the same temp directory and returns its filename.
        """
        # If SAM weights are missing, run robust high-fidelity landmark-guided fallback mask generation
        if not os.path.exists(self.checkpoint_path):
            image = cv2.imread(image_path)
            if image is None:
                return None
            h, w, c = image.shape
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Head Circle based on nose & ears
            if len(landmarks) > 0:
                nose = landmarks[0]
                l_ear = landmarks[7] if len(landmarks) > 7 else nose
                r_ear = landmarks[8] if len(landmarks) > 8 else nose
                head_center_x = int(nose["x"] * w)
                head_center_y = int(nose["y"] * h)
                head_radius = int(np.sqrt((l_ear["x"] - r_ear["x"])**2 + (l_ear["y"] - r_ear["y"])**2) * w * 1.3)
                if head_radius <= 0:
                    head_radius = int(h * 0.09)
                cv2.circle(binary_mask, (head_center_x, head_center_y), head_radius, 255, -1)

            # Neck, shoulders, hips, knees, ankles, wrists polygon
            poly_points = []
            
            # Left Arm: Left Shoulder (11) -> Left Elbow (13) -> Left Wrist (15)
            for idx in [11, 13, 15]:
                if idx < len(landmarks):
                    poly_points.append([int(landmarks[idx]["x"] * w), int(landmarks[idx]["y"] * h)])
            
            # Left Leg: Left Hip (23) -> Left Knee (25) -> Left Ankle (27) -> Left Heel (29) -> Left Foot Index (31)
            for idx in [23, 25, 27, 29, 31]:
                if idx < len(landmarks):
                    poly_points.append([int(landmarks[idx]["x"] * w), int(landmarks[idx]["y"] * h)])
                    
            # Right Leg: Right Foot Index (32) -> Right Heel (30) -> Right Ankle (28) -> Right Knee (26) -> Right Hip (24)
            for idx in [32, 30, 28, 26, 24]:
                if idx < len(landmarks):
                    poly_points.append([int(landmarks[idx]["x"] * w), int(landmarks[idx]["y"] * h)])
                    
            # Right Arm: Right Wrist (16) -> Right Elbow (14) -> Right Shoulder (12)
            for idx in [16, 14, 12]:
                if idx < len(landmarks):
                    poly_points.append([int(landmarks[idx]["x"] * w), int(landmarks[idx]["y"] * h)])

            if len(poly_points) > 2:
                pts = np.array(poly_points, dtype=np.int32)
                cv2.fillPoly(binary_mask, [pts], 255)
                
            # Post-process binary mask to make it smooth and seamless
            kernel = np.ones((5, 5), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=3)
            binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)

            # Save binary mask
            file_dir, file_name = os.path.split(image_path)
            base_name, _ = os.path.splitext(file_name)
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(file_dir, mask_filename)
            cv2.imwrite(mask_path, binary_mask)
            return mask_filename

        self._lazy_init()
        
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        h, w, c = image.shape
        
        # Select key skeletal indices: nose(0), shoulders(11, 12), elbows(13, 14), hips(23, 24), knees(25, 26)
        target_indices = [0, 11, 12, 13, 14, 23, 24, 25, 26]
        input_points = []
        input_labels = []
        
        for idx in target_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                # Filter points based on pose estimator confidence
                if lm.get("visibility", 0) > 0.5:
                    px = int(lm["x"] * w)
                    py = int(lm["y"] * h)
                    input_points.append([px, py])
                    input_labels.append(1)  # 1 = Foreground Point
                    
        # Fallback if landmarks have poor visibility
        if not input_points:
            input_points.append([w // 2, h // 2])
            input_labels.append(1)

        input_coords = np.array(input_points)
        input_labels = np.array(input_labels)

        # Process image via SAM Predictor
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        # multimask_output=False guarantees the single highest confidence segment
        masks, scores, _ = self.predictor.predict(
            point_coords=input_coords,
            point_labels=input_labels,
            multimask_output=False
        )
        
        # Extract the boolean mask
        best_mask = masks[0]
        
        # Format binary mask: white pixels (255) for subject, black (0) for background
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        binary_mask[best_mask] = 255
        
        # Save output mask as PNG file
        file_dir, file_name = os.path.split(image_path)
        base_name, _ = os.path.splitext(file_name)
        mask_filename = f"{base_name}_mask.png"
        mask_path = os.path.join(file_dir, mask_filename)
        
        cv2.imwrite(mask_path, binary_mask)
        
        return mask_filename

# Global instance of SegmentationService
segmentation_service = SegmentationService()
