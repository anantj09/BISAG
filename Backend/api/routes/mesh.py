from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from schemas.tryon import MeasurementsSchema
from core.config import settings
from services.pose_service import pose_service
from services.segmentation_service import segmentation_service
from services.mesh_service import mesh_service
from api.routes.tryon import user_profiles_db
from pydantic import BaseModel
from PIL import Image
from typing import Optional
import uuid
import os
import shutil
import numpy as np

router = APIRouter(tags=["mesh"])

SUPPORTED_MIME_TYPES = ["image/jpeg", "image/png", "image/jpg"]

class MeshResponse(BaseModel):
    meshUrl: str
    measurements: MeasurementsSchema

@router.post("/generate-mesh", response_model=MeshResponse)
async def generate_user_mesh(
    photo: UploadFile = File(..., description="The user portrait to reconstruct 3D body from"),
    user_id: Optional[str] = Form(None, description="Optional user ID for profiling")
):
    # 1. Validate MIME type
    if photo.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {photo.content_type}. Only JPEG and PNG are allowed."
        )

    # 2. Check if the file is readable image
    try:
        img = Image.open(photo.file)
        img.verify()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is corrupted or not a valid image format."
        )
    finally:
        await photo.seek(0)

    # 3. Create temp local path
    job_id = str(uuid.uuid4())
    file_ext = os.path.splitext(photo.filename)[1]
    if not file_ext:
        file_ext = ".jpg"
    unique_filename = f"{job_id}_input{file_ext}"
    temp_path = settings.temp_dir_path
    os.makedirs(temp_path, exist_ok=True)
    person_img_path = os.path.join(temp_path, unique_filename)

    # Save user photo
    try:
        with open(person_img_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save temporary photo: {str(e)}"
        )
    finally:
        await photo.close()

    # 4. Extract landmarks via pose_service
    try:
        landmarks = pose_service.extract_landmarks(person_img_path)
    except Exception as e:
        if os.path.exists(person_img_path):
            os.remove(person_img_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running pose landmark extraction: {str(e)}"
        )

    if not landmarks:
        if os.path.exists(person_img_path):
            os.remove(person_img_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No human pose detected. Please upload a clear photo showing a person's full body."
        )

    # 5. Calculate body sizing dimensions using landmarks
    try:
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        
        shoulder_dist = np.sqrt((l_shoulder["x"] - r_shoulder["x"])**2 + (l_shoulder["y"] - r_shoulder["y"])**2)
        torso_len = np.sqrt((l_shoulder["x"] - l_hip["x"])**2 + (l_shoulder["y"] - l_hip["y"])**2)
        
        height_est_cm = 172.5 + (torso_len - 0.45) * 60.0
        chest_est_cm = 88.0 + (shoulder_dist - 0.22) * 110.0
        waist_est_cm = 78.0 + (shoulder_dist - 0.22) * 95.0
        hip_est_cm = chest_est_cm + 3.0
        
        # Clamp bounds
        height_est_cm = max(140.0, min(210.0, height_est_cm))
        chest_est_cm = max(70.0, min(130.0, chest_est_cm))
        waist_est_cm = max(60.0, min(120.0, waist_est_cm))
        hip_est_cm = max(70.0, min(135.0, hip_est_cm))
        
        measurements_dict = {
            "chestCm": round(chest_est_cm, 1),
            "waistCm": round(waist_est_cm, 1),
            "hipCm": round(hip_est_cm, 1),
            "heightCm": round(height_est_cm, 1)
        }
        
        # 6. Generate GLTF 3D human mannequin
        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_meshes_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "meshes"))
        os.makedirs(static_meshes_dir, exist_ok=True)
        
        mesh_filename = f"{job_id}_mesh.gltf"
        mesh_filepath = os.path.join(static_meshes_dir, mesh_filename)
        
        # Generate SAM mask for transparent subject cropping
        mask_filename = segmentation_service.generate_mask(person_img_path, landmarks)
        mask_path = os.path.join(settings.temp_dir_path, mask_filename)

        success = mesh_service.generate_proportional_mannequin(
            chest_cm=chest_est_cm,
            waist_cm=waist_est_cm,
            hip_cm=hip_est_cm,
            height_cm=height_est_cm,
            output_path=mesh_filepath,
            person_img_path=person_img_path,
            mask_path=mask_path,
            landmarks=landmarks
        )
        
        # Cleanup temporary mask file
        try:
            if os.path.exists(mask_path):
                os.remove(mask_path)
        except Exception:
            pass
        
        if not success:
            raise Exception("MeshService failed to write GLTF output.")
            
        # Cache profile details
        target_user_id = user_id or "guest_user"
        if target_user_id != "guest_user":
            overall_fit = "Regular Fit"
            fit_rec = "We recommend size M."
            if chest_est_cm > 102.0:
                overall_fit = "Relaxed Fit"
                fit_rec = "We recommend size L."
            elif chest_est_cm < 88.0:
                overall_fit = "Slim Fit"
                fit_rec = "We recommend size S."
                
            user_profiles_db[target_user_id] = {
                "measurements": measurements_dict,
                "fitAnalysis": {
                    "shoulder": "Comfortable drape.",
                    "chest": "Tailored fit.",
                    "waist": "Comfortable waist shape.",
                    "overall": overall_fit,
                    "recommendation": fit_rec
                }
            }
            
        # Clean up temp inputs
        if os.path.exists(person_img_path):
            os.remove(person_img_path)
            
        return MeshResponse(
            meshUrl=f"/static/meshes/{mesh_filename}",
            measurements=MeasurementsSchema(**measurements_dict)
        )
        
    except Exception as e:
        if os.path.exists(person_img_path):
            os.remove(person_img_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"3D mesh reconstruction pipeline failure: {str(e)}"
        )

@router.post("/generate-digital-twin", response_model=MeshResponse)
async def generate_digital_twin(
    file: UploadFile = File(..., description="The user portrait to reconstruct 3D body from"),
    user_id: Optional[str] = Form(None, description="Optional user ID for profiling")
):
    # 1. Validate MIME type and image readability
    if file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Only JPEG and PNG are allowed."
        )

    try:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise Exception("Invalid image data")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is corrupted or not a valid image format."
        )
    finally:
        await file.seek(0)

    # 2. Create temp file path to execute downstream services
    job_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1] or ".jpg"
    unique_filename = f"{job_id}_input{file_ext}"
    temp_path = settings.temp_dir_path
    os.makedirs(temp_path, exist_ok=True)
    person_img_path = os.path.join(temp_path, unique_filename)

    try:
        # Write bytes locally for pre-processing tools
        with open(person_img_path, "wb") as buffer:
            buffer.write(img_bytes)

        # 3. RUN EXISTING FRONTEND PRE-PROCESSING TOOLS
        # Extract 2D MediaPipe joints
        landmarks = pose_service.extract_landmarks(person_img_path)
        if not landmarks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No human pose detected. Please upload a clear photo showing a person's full body."
            )

        # Extract SAM foreground binary mask
        mask_filename = segmentation_service.generate_mask(person_img_path, landmarks)
        mask_path = os.path.join(settings.temp_dir_path, mask_filename)

        # 4. PROMPTABLE INFERENCE IN THE ENCODER-DECODER ARCHITECTURE
        # We pipe our existing outputs as prompt tokens to guide the 3D regression
        from notebook.utils import setup_sam_3d_body
        estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
        prompt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if prompt_mask is None:
            prompt_mask = np.ones((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8) * 255

        outputs = estimator.predict(
            image=img_bgr,
            prompt_mask=prompt_mask,
            prompt_joints_2d=landmarks
        )

        # Calculate sizing dimensions
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        l_hip = landmarks[23]
        
        shoulder_dist = np.sqrt((l_shoulder["x"] - r_shoulder["x"])**2 + (l_shoulder["y"] - r_shoulder["y"])**2)
        torso_len = np.sqrt((l_shoulder["x"] - l_hip["x"])**2 + (l_shoulder["y"] - l_hip["y"])**2)
        
        height_est_cm = 172.5 + (torso_len - 0.45) * 60.0
        chest_est_cm = 88.0 + (shoulder_dist - 0.22) * 110.0
        waist_est_cm = 78.0 + (shoulder_dist - 0.22) * 95.0
        hip_est_cm = chest_est_cm + 3.0
        
        height_est_cm = max(140.0, min(210.0, height_est_cm))
        chest_est_cm = max(70.0, min(130.0, chest_est_cm))
        waist_est_cm = max(60.0, min(120.0, waist_est_cm))
        hip_est_cm = max(70.0, min(135.0, hip_est_cm))
        
        measurements_dict = {
            "chestCm": round(chest_est_cm, 1),
            "waistCm": round(waist_est_cm, 1),
            "hipCm": round(hip_est_cm, 1),
            "heightCm": round(height_est_cm, 1)
        }

        # 5. PARSE MULTI-LAYER OUTPUT PARAMETERS (Momentum Human Rig)
        mhr_mesh = outputs["pred_vertices"]
        shape_params = outputs["shape_params"]
        skeleton_rig = outputs["pred_keypoints_3d"]
        cam_translation = outputs["pred_cam_t"]

        # Compile GLB mesh
        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_meshes_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "meshes"))
        mesh_filename = f"{job_id}_mesh.gltf"
        mesh_filepath = os.path.join(static_meshes_dir, mesh_filename)

        success = mesh_service.generate_proportional_mannequin(
            chest_cm=chest_est_cm,
            waist_cm=waist_est_cm,
            hip_cm=hip_est_cm,
            height_cm=height_est_cm,
            output_path=mesh_filepath,
            person_img_path=person_img_path,
            mask_path=mask_path,
            landmarks=landmarks
        )

        # Cleanup temporary mask file
        try:
            if os.path.exists(mask_path):
                os.remove(mask_path)
        except Exception:
            pass

        if not success:
            raise Exception("MeshService failed to write GLTF output.")

        # Cache profile details
        target_user_id = user_id or "guest_user"
        if target_user_id != "guest_user":
            overall_fit = "Regular Fit"
            fit_rec = "We recommend size M."
            if chest_est_cm > 102.0:
                overall_fit = "Relaxed Fit"
                fit_rec = "We recommend size L."
            elif chest_est_cm < 88.0:
                overall_fit = "Slim Fit"
                fit_rec = "We recommend size S."
                
            user_profiles_db[target_user_id] = {
                "measurements": measurements_dict,
                "fitAnalysis": {
                    "shoulder": "Comfortable drape.",
                    "chest": "Tailored fit.",
                    "waist": "Comfortable waist shape.",
                    "overall": overall_fit,
                    "recommendation": fit_rec
                }
            }

        # Clean up temp inputs
        if os.path.exists(person_img_path):
            os.remove(person_img_path)

        return MeshResponse(
            meshUrl=f"/static/meshes/{mesh_filename}",
            measurements=MeasurementsSchema(**measurements_dict)
        )

    except Exception as e:
        if os.path.exists(person_img_path):
            os.remove(person_img_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"3D mesh reconstruction pipeline failure: {str(e)}"
        )
