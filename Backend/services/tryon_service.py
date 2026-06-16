import os
import cv2
import numpy as np
from typing import List, Dict, Optional

class TryOnService:
    def __init__(self):
        self.weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "hr_viton_weights.pth")
        self.model = None

    def _lazy_init(self):
        """
        Lazily initialize PyTorch and load HR-VITON model weights.
        Prevents FastAPI startup delays and import errors during active development.
        """
        if self.model is not None:
            return
            
        # In full execution, load weights on GPU:
        # import torch
        # from hr_viton import WarpingModel
        # self.model = WarpingModel()
        # self.model.to("cuda")
        # self.model.load_state_dict(torch.load(self.weights_path))
        pass

    def run_tryon(self, person_img_path: str, mask_path: str, clothing_img_path: str, landmarks: List[Dict[str, float]]) -> Optional[str]:
        """
        Warp target clothing onto the person using landmarks & SAM binary mask.
        Uses advanced vector-projected Homography (Perspective Transform) to warp the garment
        anatomically, locking the roll and yaw to the shoulders while projecting the hemline
        naturally along the downward torso vector. This eliminates hip-squeeze distortions.
        Saves output clothed image in same temp directory and returns its filename.
        """
        self._lazy_init()

        # Load input assets (load clothing with UNCHANGED to keep alpha transparency!)
        person_img = cv2.imread(person_img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        clothing_img = cv2.imread(clothing_img_path, cv2.IMREAD_UNCHANGED)

        if person_img is None or mask is None or clothing_img is None:
            return None

        h, w, c = person_img.shape
        cloth_h, cloth_w, cloth_c = clothing_img.shape

        # If clothing has no alpha channel, add a solid alpha channel
        if cloth_c == 3:
            clothing_img = cv2.cvtColor(clothing_img, cv2.COLOR_BGR2BGRA)
            cloth_c = 4

        # Extract landmarks: key indices: shoulders(11, 12), hips(23, 24)
        if len(landmarks) <= 24:
            return None

        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        r_hip = landmarks[24]

        # Convert landmarks from relative [0,1] to absolute pixel coordinates
        ls_x, ls_y = l_shoulder["x"] * w, l_shoulder["y"] * h
        rs_x, rs_y = r_shoulder["x"] * w, r_shoulder["y"] * h
        lh_x, lh_y = l_hip["x"] * w, l_hip["y"] * h
        rh_x, rh_y = r_hip["x"] * w, r_hip["y"] * h

        # Calculate midpoints
        mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
        mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2

        # 1. Torso Downward Direction Vector (Shoulders -> Hips)
        torso_vx = mid_hip_x - mid_shoulder_x
        torso_vy = mid_hip_y - mid_shoulder_y
        torso_height = np.sqrt(torso_vx**2 + torso_vy**2)
        if torso_height == 0:
            torso_height = 1.0
        down_x = torso_vx / torso_height
        down_y = torso_vy / torso_height

        # 2. Shoulder Outwards Direction Vector (Right Shoulder -> Left Shoulder)
        sh_vx = ls_x - rs_x
        sh_vy = ls_y - rs_y
        shoulder_width = np.sqrt(sh_vx**2 + sh_vy**2)
        if shoulder_width == 0:
            shoulder_width = 1.0
        right_x = sh_vx / shoulder_width
        right_y = sh_vy / shoulder_width

        # Dynamic garment padding ratios based on anatomical posture
        # Garment width should pad slightly wider than skeletal joints.
        shoulder_pad_ratio = 0.12  # 12% shoulder width extension on each side for natural fit
        collar_pad_ratio = 0.08    # 8% torso height extension upwards
        
        # Adjust hemline length dynamically using the cloth aspect ratio
        cloth_aspect_ratio = cloth_h / cloth_w
        if cloth_aspect_ratio > 1.25:
            # Long garment / Dress: hemline extends below the hips significantly
            hem_pad_ratio = 0.55
        else:
            # Normal Shirt / Tee: hemline extends slightly below the hips
            hem_pad_ratio = 0.15

        # 3. Calculate anchor coordinates using robust vector projections
        # A. Collar center: projected upwards from shoulder center
        collar_x = mid_shoulder_x - collar_pad_ratio * torso_height * down_x
        collar_y = mid_shoulder_y - collar_pad_ratio * torso_height * down_y

        # B. Top-Left anchor: Right Shoulder (extended outwards from collar)
        rs_pad_x = collar_x - (0.5 + shoulder_pad_ratio) * shoulder_width * right_x
        rs_pad_y = collar_y - (0.5 + shoulder_pad_ratio) * shoulder_width * right_y

        # C. Top-Right anchor: Left Shoulder (extended outwards from collar)
        ls_pad_x = collar_x + (0.5 + shoulder_pad_ratio) * shoulder_width * right_x
        ls_pad_y = collar_y + (0.5 + shoulder_pad_ratio) * shoulder_width * right_y

        # D. Project bottom anchors straight down along the torso vector (preserves garment drape)
        garment_height = torso_height * (1.0 + collar_pad_ratio + hem_pad_ratio)
        
        # Bottom-Left anchor: Right Hip / Hemline
        rh_pad_x = rs_pad_x + garment_height * down_x
        rh_pad_y = rs_pad_y + garment_height * down_y
        
        # Bottom-Right anchor: Left Hip / Hemline
        lh_pad_x = ls_pad_x + garment_height * down_x
        lh_pad_y = ls_pad_y + garment_height * down_y

        # Define source and destination quad anchors
        src_pts = np.float32([
            [0, 0],
            [cloth_w - 1, 0],
            [0, cloth_h - 1],
            [cloth_w - 1, cloth_h - 1]
        ])
        
        dst_pts = np.float32([
            [rs_pad_x, rs_pad_y],
            [ls_pad_x, ls_pad_y],
            [rh_pad_x, rh_pad_y],
            [lh_pad_x, lh_pad_y]
        ])

        # Execute dynamic posture-aware Homography warping
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_cloth = cv2.warpPerspective(
            clothing_img, 
            M, 
            (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0, 0, 0, 0)
        )

        warped_bgr = warped_cloth[:, :, 0:3]
        warped_alpha = warped_cloth[:, :, 3]

        # 3. Use the warped clothing's own high-fidelity alpha channel directly
        # This completely prevents sleeves, hemlines, and collars from being cut off by the body mask
        final_cloth_mask = warped_alpha
        
        # Dilation and Gaussian blur for smooth, feather-blended edges
        kernel = np.ones((3, 3), np.uint8)
        final_cloth_mask = cv2.dilate(final_cloth_mask, kernel, iterations=1)
        final_cloth_mask = cv2.GaussianBlur(final_cloth_mask, (3, 3), 0)
        
        # 4. Premium Mathematical Alpha Blending: blended = person * (1 - alpha) + clothing * alpha
        mask_normalized = final_cloth_mask.astype(float) / 255.0
        mask_normalized = np.expand_dims(mask_normalized, axis=2)  # Shape (H, W, 1) for broadcasting
        
        clothed_output = (person_img.astype(float) * (1.0 - mask_normalized) + warped_bgr.astype(float) * mask_normalized).astype(np.uint8)

        # Save result file in the temp directory
        file_dir, file_name = os.path.split(person_img_path)
        base_name, _ = os.path.splitext(file_name)
        result_filename = f"{base_name}_result.png"
        result_path = os.path.join(file_dir, result_filename)

        cv2.imwrite(result_path, clothed_output)

        return result_filename

# Global instance of TryOnService
tryon_service = TryOnService()
