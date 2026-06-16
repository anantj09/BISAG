from pydantic import BaseModel, Field
from typing import Optional, List

class LandmarkPoint(BaseModel):
    x: float = Field(..., description="Normalized X coordinate (0.0 to 1.0)")
    y: float = Field(..., description="Normalized Y coordinate (0.0 to 1.0)")
    z: float = Field(..., description="Z coordinate representing landmark depth")
    visibility: float = Field(..., description="Confidence value of landmark visibility")

class UploadPhotoResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the asynchronous processing job")
    filename: str = Field(..., description="Saved filename of the uploaded photo")
    message: str = Field(..., description="Status description of the upload")
    mask_filename: Optional[str] = Field(default=None, description="Filename of the saved binary segmentation mask")
    body_landmarks: Optional[List[LandmarkPoint]] = Field(
        default=None, 
        description="Coordinates of extracted 33 body pose landmarks"
    )
