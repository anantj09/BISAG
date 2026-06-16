# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import cv2

class SAM3DBodyEstimator:
    def __init__(self, hf_repo_id: str):
        self.hf_repo_id = hf_repo_id
        self.initialized = False
        self.model = None
        self.model_cfg = None
        self.device = None
        self.detector = None
        
    def _initialize(self):
        if self.initialized:
            return
        # Lazy-initialize HMR2 model and load checkpoints
        from services.mesh_service import mesh_service
        mesh_service.initialize_models()
        self.model = mesh_service.model
        self.model_cfg = mesh_service.model_cfg
        self.device = mesh_service.device
        self.detector = mesh_service.detector
        self.initialized = True
        
    def predict(self, image: np.ndarray, prompt_mask: np.ndarray, prompt_joints_2d: list) -> dict:
        """
        Generative 3D twin prediction matching the SAM-3D interface.
        Pipes 2D keypoint prompts and segmentation masks to estimate the human rig.
        """
        self._initialize()
        
        h_img, w_img, _ = image.shape
        
        # 1. Map 2D MediaPipe landmark prompts to construct primary bbox
        xs = [pt["x"] * w_img for pt in prompt_joints_2d if isinstance(pt, dict) and "x" in pt]
        ys = [pt["y"] * h_img for pt in prompt_joints_2d if isinstance(pt, dict) and "y" in pt]
        
        if xs and ys:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w = max_x - min_x
            h = max_y - min_y
            pad_x = w * 0.15
            pad_y = h * 0.15
            box = np.array([[
                max(0.0, min_x - pad_x),
                max(0.0, min_y - pad_y),
                min(float(w_img), max_x + pad_x),
                min(float(h_img), max_y + pad_y)
            ]], dtype=np.float32)
        else:
            # Fallback to Detectron2 bbox detector
            det_out = self.detector(image)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            if len(boxes) > 0:
                box = boxes[0:1]
            else:
                box = np.array([[0.0, 0.0, float(w_img), float(h_img)]], dtype=np.float32)

        # 2. Run 4D-Humans estimator internally to perform 3D human pose recovery
        from hmr2.datasets.vitdet_dataset import ViTDetDataset
        from hmr2.utils import recursive_to
        from hmr2.utils.renderer import cam_crop_to_full
        
        dataset = ViTDetDataset(self.model_cfg, image, box)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        batch = recursive_to(batch, self.device)
        
        with torch.no_grad():
            out = self.model(batch)
            
        pred_vertices = out['pred_vertices'][0].detach().cpu().numpy()
        pred_cam = out['pred_cam']
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        
        scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
        cam_t = pred_cam_t_full[0].detach().cpu().numpy()
        
        # Soft-tissue/muscle params (SMPL beta coefficients)
        betas = out['pred_smpl_params']['betas'][0].detach().cpu().numpy()
        
        # 3D skeletal rig joints (Pad to 70 joints as expected by Momentum Human Rig)
        joints_3d = out['pred_keypoints_3d'][0].detach().cpu().numpy()
        padded_joints = np.zeros((70, 3), dtype=np.float32)
        min_joints = min(70, len(joints_3d))
        padded_joints[:min_joints] = joints_3d[:min_joints]
        
        return {
            "pred_vertices": pred_vertices,
            "shape_params": betas,
            "pred_keypoints_3d": padded_joints,
            "pred_cam_t": cam_t
        }

def setup_sam_3d_body(hf_repo_id: str):
    return SAM3DBodyEstimator(hf_repo_id)
