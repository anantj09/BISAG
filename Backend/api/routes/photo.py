from fastapi import APIRouter, UploadFile, File, HTTPException, status
from schemas.photo import UploadPhotoResponse
from core.config import settings
from services.pose_service import pose_service
from services.segmentation_service import segmentation_service
from PIL import Image
import uuid
import os
import shutil

router = APIRouter(tags=["photo"])

SUPPORTED_MIME_TYPES = ["image/jpeg", "image/png", "image/jpg"]

@router.post("/upload-photo", response_model=UploadPhotoResponse, status_code=status.HTTP_201_CREATED)
async def upload_photo(file: UploadFile = File(..., description="The user image to upload (JPEG or PNG)")):
    # 1. Validate MIME type
    if file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Only JPEG and PNG are allowed."
        )

    # 2. Check if the file is actually a readable image using Pillow
    try:
        img = Image.open(file.file)
        img.verify()  # Verifies the file is an image without decoding fully
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is corrupted or not a valid image format."
        )
    finally:
        # Seek back to start of file after verifying
        await file.seek(0)

    # 3. Create unique UUID filename to avoid name collisions
    file_ext = os.path.splitext(file.filename)[1]
    if not file_ext:
        file_ext = ".jpg" if "jpeg" in file.content_type or "jpg" in file.content_type else ".png"
    
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    
    # Ensure temporary directory exists
    temp_path = settings.temp_dir_path
    os.makedirs(temp_path, exist_ok=True)
    
    file_path = os.path.join(temp_path, unique_filename)

    # 4. Save the file locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the image: {str(e)}"
        )
    finally:
        await file.close()

    # 5. Extract pose landmarks in real-time using MediaPipe
    try:
        landmarks = pose_service.extract_landmarks(file_path)
    except Exception as e:
        # Clean up temp file on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running pose landmark extraction: {str(e)}"
        )

    # 6. Validate that landmarks were actually found (human presence validation)
    if not landmarks:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No human pose detected. Please upload a clear photo showing a person's full body or pose."
        )

    # 7. Extract binary body segmentation mask using SAM
    try:
        mask_filename = segmentation_service.generate_mask(file_path, landmarks)
    except Exception as e:
        # Clean up temp photo on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running body segmentation: {str(e)}"
        )

    mock_job_id = str(uuid.uuid4())
    
    return UploadPhotoResponse(
        job_id=mock_job_id,
        filename=unique_filename,
        message="Photo uploaded, pose landmarks and segmentation mask generated successfully.",
        mask_filename=mask_filename,
        body_landmarks=landmarks
    )
