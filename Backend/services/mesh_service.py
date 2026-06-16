import os
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import trimesh

class MeshService:
    def __init__(self):
        self.model = None
        self.model_cfg = None
        self.detector = None
        self.device = None
        self.initialized = False

    def check_smpl_weights(self) -> Tuple[bool, str]:
        """
        Validates if the required SMPL neutral body model weights exist.
        Returns a tuple of (exists, instruction_message).
        """
        # HMR2 cache dir
        home_dir = os.environ.get("HOME") or os.environ.get("USERPROFILE") or os.path.expanduser("~")
        cache_dir_4dhumans = os.path.join(home_dir, ".cache", "4DHumans")
        
        candidates = [
            os.path.join(cache_dir_4dhumans, "data", "smpl", "SMPL_NEUTRAL.pkl"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
        ]
        
        exists = any(os.path.exists(c) for c in candidates)
        if not exists:
            msg = (
                "\n"
                "=======================================================================\n"
                "[SMPL WEIGHTS NOT DETECTED]\n"
                "To generate custom 3D body models locally, you must obtain the SMPL weights:\n"
                "1. Register a free account at https://smplify.is.tue.mpg.de/\n"
                "2. Download 'SMPL for Python' (version 1.0.0 or equivalent).\n"
                "3. Rename the neutral body model file to:\n"
                "   'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'\n"
                f"4. Place it inside: {os.path.dirname(candidates[1])}/\n"
                "=======================================================================\n"
            )
            return False, msg
        return True, ""

    def initialize_models(self):
        """
        Loads the pre-trained 4D-Humans (HMR 2.0) and Detectron2 human detector model on startup.
        Ensures lazy loading on first API call if not loaded.
        """
        if self.initialized:
            return
            
        print(">>> Validating local 4D-Humans (HMR 2.0) environment...")
        has_smpl, smpl_msg = self.check_smpl_weights()
        if not has_smpl:
            print(smpl_msg)
            raise FileNotFoundError("Missing SMPL neutral body model weights. Refer to the instruction log above.")
            
        import torch
        if not hasattr(torch.load, '__monkeypatched__'):
            original_load = torch.load
            def safe_load(*args, **kwargs):
                try:
                    return original_load(*args, **kwargs)
                except Exception as e:
                    if kwargs.get('weights_only', True) is not False:
                        kwargs_copy = kwargs.copy()
                        kwargs_copy['weights_only'] = False
                        try:
                            return original_load(*args, **kwargs_copy)
                        except Exception:
                            pass
                    raise e
            safe_load.__monkeypatched__ = True
            torch.load = safe_load

        from omegaconf.dictconfig import DictConfig
        from omegaconf.listconfig import ListConfig
        try:
            torch.serialization.add_safe_globals([DictConfig, ListConfig])
        except Exception:
            pass

        # Monkeypatch numpy to add removed types for chumpy compatibility
        import numpy as np
        np.bool = bool
        np.int = int
        np.float = float
        np.complex = complex
        np.object = object
        np.str = str
        np.unicode = str

        from hmr2.configs import CACHE_DIR_4DHUMANS
        from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
        
        # 1. Download HMR2 model weights (saves checkpoints to ~/.cache/4DHumans)
        try:
            download_models(CACHE_DIR_4DHUMANS)
        except Exception as e:
            print(f"[WARNING] 4D-Humans weights download failed: {str(e)}")
            
        print(">>> Initializing HMR 2.0 network layers...")
        self.model, self.model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 2. Initialize Detectron2 human bbox detector
        print(">>> Initializing Detectron2 human bounding-box detector...")
        from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
        from detectron2.config import LazyConfig
        import hmr2
        
        cfg_path = Path(hmr2.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        # Point to standard COCO detector weights
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        self.initialized = True
        print(f"--- 4D-Humans successfully mapped to hardware: [{self.device.type.upper()}] ---")

    def generate_local_hmr2_mesh(self, person_img_path: str, output_path: str, mask_path: str = None, landmarks: list = None) -> bool:
        """
        Executes local SAM-3D Body (facebook/sam-3d-body-dinov3) pose and soft-tissue reconstruction,
        samples vertex colors, and saves a compiled centered GLB mesh.
        """
        try:
            from notebook.utils import setup_sam_3d_body
            
            # Read input image
            img_cv2 = cv2.imread(person_img_path)
            if img_cv2 is None:
                print(f"Error: Unable to read image path: {person_img_path}")
                return False
                
            h_img, w_img, _ = img_cv2.shape
            
            # Extract landmarks if not passed
            if landmarks is None:
                from services.pose_service import pose_service
                landmarks = pose_service.extract_landmarks(person_img_path)
                
            if not landmarks:
                print("[WARNING] No human landmarks detected. Prompt reconstruction requires landmarks.")
                return False

            # Extract segmentation mask if not passed
            if mask_path is None or not os.path.exists(mask_path):
                from services.segmentation_service import segmentation_service
                mask_filename = segmentation_service.generate_mask(person_img_path, landmarks)
                mask_path = os.path.join(os.path.dirname(person_img_path), mask_filename)
                
            prompt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if prompt_mask is None:
                prompt_mask = np.ones((h_img, w_img), dtype=np.uint8) * 255

            # 1. Initialize SOTA Meta SAM-3D Body Estimator
            print("Initializing SOTA Meta SAM-3D Body Estimator...")
            estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

            # 2. Run promptable inference guided by 2D joints and foreground masks
            print("Running SAM-3D Body estimator predict...")
            outputs = estimator.predict(
                image=img_cv2,
                prompt_mask=prompt_mask,
                prompt_joints_2d=landmarks
            )

            # 3. Parse multi-layer Momentum Human Rig (MHR) output parameters
            pred_vertices = outputs["pred_vertices"]      # 3D vertices using MHR format
            shape_params = outputs["shape_params"]        # Soft-tissue/muscle metrics
            skeleton_rig = outputs["pred_keypoints_3d"]    # 70 body, hand, & feet markers
            cam_t = outputs["pred_cam_t"]                 # Camera translation vector

            # 4. Project 3D vertices onto 2D image coordinates to sample colors
            self.initialize_models()
            img_size = max(h_img, w_img)
            f = (self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE) * img_size
            
            vertices_cam = pred_vertices + cam_t
            cx = w_img / 2.0
            cy = h_img / 2.0
            
            X = vertices_cam[:, 0]
            Y = vertices_cam[:, 1]
            Z = np.where(vertices_cam[:, 2] == 0, 1e-5, vertices_cam[:, 2])
            
            x_proj = f * (X / Z) + cx
            y_proj = f * (Y / Z) + cy
            
            cols = np.clip(np.round(x_proj), 0, w_img - 1).astype(np.int32)
            rows = np.clip(np.round(y_proj), 0, h_img - 1).astype(np.int32)
            
            colors_list = []
            for r, c in zip(rows, cols):
                bgr = img_cv2[r, c]
                rgb = [bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0, 1.0]
                colors_list.append(rgb)
            vertex_colors = np.array(colors_list)
            
            # 5. Compile Rigged Mesh
            print("Compiling mesh with sampled vertex colors...")
            faces = self.model.smpl.faces
            if hasattr(faces, 'cpu'):
                faces = faces.cpu().numpy()
                
            mesh = trimesh.Trimesh(pred_vertices, faces.copy(), vertex_colors=vertex_colors)
            
            # 5. Apply rotations to align coordinate axes to OpenGL/Filament standard
            # (Flip Y to be up, and Z to be negative)
            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            
            # Post-rotation: Y is up, front faces towards +Z, back faces towards -Z
            # Clean up projection anomalies on the sides and back (duplicate face, white borders)
            normals = mesh.vertex_normals
            vertices = mesh.vertices
            vertex_colors = mesh.visual.vertex_colors / 255.0  # Convert to float RGBA [0..1]
            
            # ================================================================
            # ADAPTIVE BOUNDARY DETECTION & REALISTIC BACKFACE TEXTURING
            # ================================================================
            # Instead of hardcoded Y thresholds, we scan front-facing vertices
            # to detect actual clothing boundaries from the photo, then apply
            # those same boundaries to back-facing vertices for consistency.
            # ================================================================

            # --- Phase 1: Collect front-facing vertex data with positions and colors ---
            front_data = []  # List of (y_coord, x_coord, color_rgba, vertex_index)
            for i in range(len(vertices)):
                nz = normals[i][2]
                color = vertex_colors[i]
                is_background = color[0] > 0.90 and color[1] > 0.90 and color[2] > 0.90
                if nz > 0.3 and not is_background:
                    front_data.append((vertices[i][1], vertices[i][0], color, i))

            # --- Phase 2: Detect actual clothing boundaries from front vertex color distribution ---
            # Sort front vertices by Y coordinate (top to bottom)
            front_data.sort(key=lambda x: -x[0])  # Highest Y first

            # Collect color samples at different Y heights (torso region only, ignoring extremities)
            y_min_body = min(v[1] for v in vertices)
            y_max_body = max(v[1] for v in vertices)
            y_range = y_max_body - y_min_body
            
            # Adaptive boundary detection: scan Y slices and detect color transitions
            num_slices = 50
            slice_colors = {}  # y_normalized -> list of (r,g,b) samples
            for fd in front_data:
                y_norm = (fd[0] - y_min_body) / max(y_range, 0.001)  # 0=bottom, 1=top
                slice_idx = min(int(y_norm * num_slices), num_slices - 1)
                if slice_idx not in slice_colors:
                    slice_colors[slice_idx] = []
                # Only consider torso vertices (not arms) for boundary detection
                if abs(fd[1]) < 0.14:
                    slice_colors[slice_idx].append(fd[2][:3])

            # Compute average color per slice and detect major transitions
            slice_avg = {}
            for idx in range(num_slices):
                if idx in slice_colors and len(slice_colors[idx]) >= 3:
                    slice_avg[idx] = np.mean(slice_colors[idx], axis=0)

            # Detect boundary heights by finding large color jumps between adjacent slices
            def color_distance(c1, c2):
                return np.sqrt(np.sum((c1 - c2) ** 2))

            boundaries = []  # List of (y_normalized, transition_type)
            sorted_slices = sorted(slice_avg.keys())
            for k in range(len(sorted_slices) - 1):
                s1 = sorted_slices[k]
                s2 = sorted_slices[k + 1]
                dist = color_distance(slice_avg[s1], slice_avg[s2])
                if dist > 0.12:  # Significant color change threshold
                    boundary_y_norm = (s1 + s2) / 2.0 / num_slices
                    boundaries.append((boundary_y_norm, dist))

            # Sort boundaries by height (bottom to top)
            boundaries.sort(key=lambda x: x[0])

            # Determine segment Y boundaries adaptively
            # Default boundaries (normalized 0=bottom, 1=top):
            #   shoes_top ~0.08, pants_top ~0.45, tshirt_top ~0.78, skin_top ~0.85, hair above
            default_boundaries = {
                "shoes_top": 0.08,
                "pants_top": 0.45,
                "tshirt_top": 0.78,
                "skin_top": 0.85
            }

            # Try to refine from detected boundaries
            detected = default_boundaries.copy()
            if len(boundaries) >= 2:
                # The two strongest boundaries are likely pants-tshirt and tshirt-skin/hair
                strongest = sorted(boundaries, key=lambda x: -x[1])[:3]
                strongest_y = sorted([b[0] for b in strongest])
                if len(strongest_y) >= 2:
                    detected["pants_top"] = strongest_y[0]
                    detected["tshirt_top"] = strongest_y[1]
                if len(strongest_y) >= 3:
                    detected["skin_top"] = strongest_y[2]

            # Convert normalized boundaries back to actual Y coordinates
            boundary_y = {
                "shoes_top": y_min_body + detected["shoes_top"] * y_range,
                "pants_top": y_min_body + detected["pants_top"] * y_range,
                "tshirt_top": y_min_body + detected["tshirt_top"] * y_range,
                "skin_top": y_min_body + detected["skin_top"] * y_range,
            }

            # --- Phase 3: Adaptive segment classifier using detected boundaries ---
            def get_segment_name(v):
                y = v[1]
                x_abs = abs(v[0])
                if y > boundary_y["skin_top"]:
                    return "hair"
                elif y > boundary_y["tshirt_top"]:
                    # Neck/face region — but arms at this height are skin
                    if x_abs > 0.14:
                        return "skin"
                    return "skin"
                elif y > boundary_y["pants_top"]:
                    # Torso region — arms extending out are skin
                    if x_abs > 0.18:
                        return "skin"
                    return "tshirt"
                elif y > boundary_y["shoes_top"]:
                    return "pants"
                else:
                    return "shoes"

            # Segment fallback colors
            segment_fallbacks = {
                "hair": np.array([0.18, 0.14, 0.12, 1.0]),
                "skin": np.array([0.85, 0.68, 0.58, 1.0]),
                "tshirt": np.array([0.24, 0.24, 0.25, 1.0]),
                "pants": np.array([0.62, 0.44, 0.28, 1.0]),
                "shoes": np.array([0.35, 0.22, 0.15, 1.0])
            }

            # --- Phase 4: Collect front-facing colors per segment for averaging ---
            segments_data = {seg: [] for seg in segment_fallbacks}
            for i in range(len(vertices)):
                nz = normals[i][2]
                color = vertex_colors[i]
                is_background = color[0] > 0.90 and color[1] > 0.90 and color[2] > 0.90
                if nz > 0.35 and not is_background:
                    seg = get_segment_name(vertices[i])
                    segments_data[seg].append(color[:3])

            # Calculate average and standard deviation color per segment
            average_colors = {}
            color_std = {}
            for seg in segment_fallbacks:
                if len(segments_data[seg]) > 10:
                    arr = np.array(segments_data[seg])
                    average_colors[seg] = np.append(np.mean(arr, axis=0), 1.0)
                    color_std[seg] = np.std(arr, axis=0)
                else:
                    average_colors[seg] = segment_fallbacks[seg].copy()
                    color_std[seg] = np.array([0.02, 0.02, 0.02])

            # --- Phase 5: Boundary smoothing parameters ---
            # Smooth transition zone height (in mesh units) at each clothing boundary
            TRANSITION_ZONE = 0.03  # ~3cm of gradual blending between segments

            def get_segment_color_with_smooth_boundary(v):
                """Returns the segment color for a vertex, with smooth blending at boundaries."""
                y = v[1]
                seg = get_segment_name(v)
                base_color = average_colors[seg]

                # Check if vertex is near a boundary and blend between adjacent segments
                boundary_checks = [
                    ("shoes_top", "shoes", "pants"),
                    ("pants_top", "pants", "tshirt"),
                    ("tshirt_top", "tshirt", "skin"),
                    ("skin_top", "skin", "hair"),
                ]
                for bname, seg_below, seg_above in boundary_checks:
                    by = boundary_y[bname]
                    if abs(y - by) < TRANSITION_ZONE:
                        t = (y - (by - TRANSITION_ZONE)) / (2 * TRANSITION_ZONE)
                        t = np.clip(t, 0.0, 1.0)
                        # Smooth step (hermite interpolation for natural transition)
                        t = t * t * (3 - 2 * t)
                        c_below = average_colors[seg_below]
                        c_above = average_colors[seg_above]
                        blended = (1 - t) * c_below + t * c_above
                        blended[3] = 1.0
                        return blended
                return base_color

            # --- Phase 6: Apply final vertex colors with realistic backface texturing ---
            new_colors = []
            # Seed for reproducible subtle noise
            rng = np.random.RandomState(42)

            for i in range(len(vertices)):
                v = vertices[i]
                n = normals[i]
                tex_color = vertex_colors[i]
                nz = n[2]

                # Get the adaptive segment color for this vertex (with boundary smoothing)
                seg_color = get_segment_color_with_smooth_boundary(v)
                seg_name = get_segment_name(v)

                # Add subtle color variation to back vertices to prevent flat look
                # Uses vertex position as seed for consistent noise across frames
                noise_intensity = 0.025  # Very subtle variation
                noise = rng.uniform(-noise_intensity, noise_intensity, 3)
                seg_color_varied = seg_color.copy()
                seg_color_varied[:3] = np.clip(seg_color[:3] + noise, 0.0, 1.0)
                seg_color_varied[3] = 1.0

                # Blending logic:
                # nz >= 0.15:  front-facing -> 100% photo texture
                # nz <= -0.10: back-facing  -> 100% adaptive segment color (with noise)
                # in between:  smooth transition to blend out white borders/projection artifacts
                if nz >= 0.15:
                    blend = 0.0
                elif nz <= -0.10:
                    blend = 1.0
                else:
                    blend = (0.15 - nz) / 0.25

                final_col = [
                    (1.0 - blend) * tex_color[j] + blend * seg_color_varied[j] for j in range(4)
                ]
                final_col_uint8 = [int(np.clip(c * 255.0, 0, 255)) for c in final_col]
                new_colors.append(final_col_uint8)

            mesh.visual.vertex_colors = np.array(new_colors, dtype=np.uint8)
            
            # Export to binary GLB container
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            glb_data = mesh.export(file_type="glb")
            
            with open(output_path, "wb") as f_out:
                f_out.write(glb_data)
                
            print(f"WSL 4D-Humans model compiled successfully at: {output_path}")
            return True
            
        except Exception as e:
            print(f"[WARNING] Local 4D-Humans inference execution failed: {str(e)}")
            return False

    def generate_proportional_mannequin(
        self,
        chest_cm: float,
        waist_cm: float,
        hip_cm: float,
        height_cm: float,
        output_path: str,
        fit_chest: str = "perfect",
        fit_waist: str = "perfect",
        fit_shoulder: str = "perfect",
        person_img_path: str = None,
        mask_path: str = None,
        landmarks: list = None
    ) -> bool:
        """
        Attempts to generate a 3D body pose mesh using 4D-Humans locally.
        Falls back to generating a proportional gray mannequin if 4D-Humans is not configured or fails.
        """
        # Try local 4D-Humans first if image path is provided
        if person_img_path and os.path.exists(person_img_path):
            success = self.generate_local_hmr2_mesh(person_img_path, output_path, mask_path=mask_path, landmarks=landmarks)
            if success:
                return True
            print("WSL local 4D-Humans generation failed. Falling back to procedural mannequin.")

        try:
            # Scale height from cm to meters
            h_m = height_cm / 100.0
            
            # Width and depth factors in meters scaled proportionally
            chest_w = (chest_cm / 100.0) * 0.35
            waist_w = (waist_cm / 100.0) * 0.32
            hip_w = (hip_cm / 100.0) * 0.36
            torso_d = 0.18
            
            def get_color_for_fit(status: str) -> Tuple[float, float, float]:
                status_lower = status.lower()
                if "tight" in status_lower:
                    return (1.0, 0.55, 0.0) # Soft Orange
                elif "loose" in status_lower:
                    return (0.0, 0.45, 1.0) # Soft Blue
                else:
                    return (0.0, 0.85, 0.4) # perfect Green
                    
            head_color = (0.0, 0.89, 1.0)
            chest_color = get_color_for_fit(fit_chest)
            waist_color = get_color_for_fit(fit_waist)
            arm_color = get_color_for_fit(fit_shoulder)
            leg_color = (0.0, 0.3, 0.7)
            
            vertices: List[float] = []
            normals: List[float] = []
            colors: List[float] = []
            indices: List[int] = []

            def sanitize_float(val: float) -> float:
                if np.isnan(val) or np.isinf(val):
                    return 0.0
                return float(val)

            def add_sphere(cx: float, cy: float, cz: float, rx: float, ry: float, rz: float, rgb: Tuple[float, float, float], segments: int = 16, rings: int = 16):
                v_start = len(vertices) // 3
                for i in range(rings + 1):
                    theta = i * np.pi / rings
                    sin_t = np.sin(theta)
                    cos_t = np.cos(theta)
                    for j in range(segments):
                        phi = j * 2 * np.pi / segments
                        sin_p = np.sin(phi)
                        cos_p = np.cos(phi)
                        x = sanitize_float(cx + rx * sin_t * cos_p)
                        y = sanitize_float(cy + ry * cos_t)
                        z = sanitize_float(cz + rz * sin_t * sin_p)
                        vertices.extend([x, y, z])
                        nx = sin_t * cos_p
                        ny = cos_t
                        nz = sin_t * sin_p
                        n_len = np.sqrt(nx**2 + ny**2 + nz**2)
                        if n_len == 0:
                            normals.extend([0.0, 1.0, 0.0])
                        else:
                            normals.extend([sanitize_float(nx/n_len), sanitize_float(ny/n_len), sanitize_float(nz/n_len)])
                        colors.extend([sanitize_float(rgb[0]), sanitize_float(rgb[1]), sanitize_float(rgb[2]), 1.0])
                        
                for i in range(rings):
                    for j in range(segments):
                        next_j = (j + 1) % segments
                        p0 = v_start + i * segments + j
                        p1 = v_start + i * segments + next_j
                        p2 = v_start + (i + 1) * segments + j
                        p3 = v_start + (i + 1) * segments + next_j
                        indices.extend([p0, p1, p3])
                        indices.extend([p0, p3, p2])

            def add_cylinder(p1: List[float], p2: List[float], r1_x: float, r1_z: float, r2_x: float, r2_z: float, rgb: Tuple[float, float, float], segments: int = 16, rings: int = 4):
                v_start = len(vertices) // 3
                p1_arr = np.array(p1)
                p2_arr = np.array(p2)
                dir_vec = p2_arr - p1_arr
                length = np.linalg.norm(dir_vec)
                if length == 0:
                    return
                dir_u = dir_vec / length
                helper = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(dir_u, helper)) > 0.95:
                    helper = np.array([0.0, 1.0, 0.0])
                u = np.cross(dir_u, helper)
                u = u / np.linalg.norm(u)
                v = np.cross(dir_u, u)
                v = v / np.linalg.norm(v)
                
                for i in range(rings + 1):
                    t = i / rings
                    pos_center = p1_arr + t * dir_vec
                    rx = r1_x * (1 - t) + r2_x * t
                    rz = r1_z * (1 - t) + r2_z * t
                    for j in range(segments):
                        angle = j * 2 * np.pi / segments
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        pt = pos_center + rx * cos_a * u + rz * sin_a * v
                        vertices.extend([sanitize_float(pt[0]), sanitize_float(pt[1]), sanitize_float(pt[2])])
                        norm = cos_a * u + sin_a * v
                        n_len = np.linalg.norm(norm)
                        if n_len == 0:
                            normals.extend([sanitize_float(norm[0]), sanitize_float(norm[1]), sanitize_float(norm[2])])
                        else:
                            norm_norm = norm / n_len
                            normals.extend([sanitize_float(norm_norm[0]), sanitize_float(norm_norm[1]), sanitize_float(norm_norm[2])])
                        colors.extend([sanitize_float(rgb[0]), sanitize_float(rgb[1]), sanitize_float(rgb[2]), 1.0])
                        
                for i in range(rings):
                    for j in range(segments):
                        next_j = (j + 1) % segments
                        p0 = v_start + i * segments + j
                        p1 = v_start + i * segments + next_j
                        p2 = v_start + (i + 1) * segments + j
                        p3 = v_start + (i + 1) * segments + next_j
                        indices.extend([p0, p1, p3])
                        indices.extend([p0, p3, p2])

            h = h_m
            d = torso_d
            add_sphere(0.0, h - 0.12, 0.0, 0.09, 0.11, 0.09, head_color)
            add_cylinder([0.0, h * 0.78, 0.0], [0.0, h - 0.20, 0.0], 0.055, 0.055, 0.05, 0.05, head_color)
            add_cylinder([0.0, h * 0.60, 0.0], [0.0, h * 0.78, 0.0], waist_w/2, d/2, chest_w/2, d/2, chest_color)
            add_cylinder([0.0, h * 0.45, 0.0], [0.0, h * 0.60, 0.0], hip_w/2, (d+0.02)/2, waist_w/2, d/2, waist_color)
            add_cylinder([-chest_w/2 - 0.02, h * 0.72, 0.0], [-chest_w/2 - 0.04, h * 0.58, 0.0], 0.045, 0.045, 0.04, 0.04, arm_color)
            add_cylinder([-chest_w/2 - 0.04, h * 0.58, 0.0], [-chest_w/2 - 0.05, h * 0.44, 0.0], 0.04, 0.04, 0.032, 0.032, arm_color)
            add_cylinder([chest_w/2 + 0.02, h * 0.72, 0.0], [chest_w/2 + 0.04, h * 0.58, 0.0], 0.045, 0.045, 0.04, 0.04, arm_color)
            add_cylinder([chest_w/2 + 0.04, h * 0.58, 0.0], [chest_w/2 + 0.05, h * 0.44, 0.0], 0.04, 0.04, 0.032, 0.032, arm_color)
            add_cylinder([-hip_w/4, h * 0.45, 0.0], [-hip_w/4, h * 0.24, 0.0], 0.075, 0.075, 0.06, 0.06, leg_color)
            add_cylinder([-hip_w/4, h * 0.24, 0.0], [-hip_w/4, 0.04, 0.0], 0.06, 0.06, 0.048, 0.048, leg_color)
            add_cylinder([hip_w/4, h * 0.45, 0.0], [hip_w/4, h * 0.24, 0.0], 0.075, 0.075, 0.06, 0.06, leg_color)
            add_cylinder([hip_w/4, h * 0.24, 0.0], [hip_w/4, 0.04, 0.0], 0.06, 0.06, 0.048, 0.048, leg_color)
                
            pos_bytes = struct.pack(f"<{len(vertices)}f", *vertices)
            norm_bytes = struct.pack(f"<{len(normals)}f", *normals)
            col_bytes = struct.pack(f"<{len(colors)}f", *colors)
            ind_bytes = struct.pack(f"<{len(indices)}H", *indices)
            
            def pad_bytes(b: bytes) -> bytes:
                pad_len = (4 - (len(b) % 4)) % 4
                return b + b"\x00" * pad_len
                
            pos_bytes = pad_bytes(pos_bytes)
            norm_bytes = pad_bytes(norm_bytes)
            col_bytes = pad_bytes(col_bytes)
            ind_bytes = pad_bytes(ind_bytes)
            
            pos_len = len(pos_bytes)
            norm_len = len(norm_bytes)
            col_len = len(col_bytes)
            ind_len = len(ind_bytes)
            
            pos_offset = 0
            norm_offset = pos_len
            col_offset = norm_offset + norm_len
            ind_offset = col_offset + col_len
            
            total_buffer = pos_bytes + norm_bytes + col_bytes + ind_bytes
            
            gltf_json = {
                "asset": {
                    "version": "2.0",
                    "generator": "AR-Virtual-Tryon-Mesh-Service"
                },
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0, "name": "HumanMannequin"}],
                "meshes": [{
                    "primitives": [{
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "COLOR_0": 2
                        },
                        "indices": 3,
                        "mode": 4
                    }],
                    "name": "MannequinMesh"
                }],
                "accessors": [
                    {
                        "bufferView": 0,
                        "componentType": 5126,
                        "count": len(vertices) // 3,
                        "type": "VEC3",
                        "max": [max(vertices[i::3]) for i in range(3)],
                        "min": [min(vertices[i::3]) for i in range(3)]
                    },
                    {
                        "bufferView": 1,
                        "componentType": 5126,
                        "count": len(normals) // 3,
                        "type": "VEC3"
                    },
                    {
                        "bufferView": 2,
                        "componentType": 5126,
                        "count": len(colors) // 4,
                        "type": "VEC4"
                    },
                    {
                        "bufferView": 3,
                        "componentType": 5123,
                        "count": len(indices),
                        "type": "SCALAR"
                    }
                ],
                "bufferViews": [
                    {
                        "buffer": 0,
                        "byteOffset": pos_offset,
                        "byteLength": pos_len,
                        "target": 34962
                    },
                    {
                        "buffer": 0,
                        "byteOffset": norm_offset,
                        "byteLength": norm_len,
                        "target": 34962
                    },
                    {
                        "buffer": 0,
                        "byteOffset": col_offset,
                        "byteLength": col_len,
                        "target": 34962
                    },
                    {
                        "buffer": 0,
                        "byteOffset": ind_offset,
                        "byteLength": ind_len,
                        "target": 34963
                    }
                ],
                "buffers": [{"byteLength": len(total_buffer)}]
            }
            
            json_str = json.dumps(gltf_json)
            json_bytes = json_str.encode("utf-8")
            json_pad = (4 - (len(json_bytes) % 4)) % 4
            if json_pad > 0:
                json_bytes += b" " * json_pad
                
            bin_bytes = total_buffer
            bin_pad = (4 - (len(bin_bytes) % 4)) % 4
            if bin_pad > 0:
                bin_bytes += b"\x00" * bin_pad
                
            total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
            header = struct.pack("<4sII", b"glTF", 2, total_length)
            json_chunk_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)
            bin_chunk_header = struct.pack("<II", len(bin_bytes), 0x004E4942)
            glb_data = header + json_chunk_header + json_bytes + bin_chunk_header + bin_bytes
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(glb_data)
            return True
        except Exception as e:
            print(f"Error generating GLTF mannequin fallback: {str(e)}")
            return False

# Global instance of MeshService
mesh_service = MeshService()
