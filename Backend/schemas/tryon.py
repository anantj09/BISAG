from pydantic import BaseModel, Field
from typing import Optional

class TryOnResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the try-on job")
    jobId: str = Field(..., alias="jobId", description="CamelCase mapping for GSON/frontend compatibility")
    status: str = Field(..., description="Current status of the try-on processing job")
    output_image_url: Optional[str] = Field(default=None, description="Filename/URL of the generated clothed try-on image")
    outputImageUrl: Optional[str] = Field(default=None, alias="outputImageUrl", description="CamelCase output url")
    message: str = Field(..., description="Descriptive status of the job completion")

    class Config:
        populate_by_name = True

class MeasurementsSchema(BaseModel):
    chestCm: float = Field(0.0, alias="chestCm", description="Chest measurement in cm")
    waistCm: float = Field(0.0, alias="waistCm", description="Waist measurement in cm")
    hipCm: float = Field(0.0, alias="hipCm", description="Hip measurement in cm")
    heightCm: float = Field(0.0, alias="heightCm", description="Height measurement in cm")

    class Config:
        populate_by_name = True

class FitAnalysisSchema(BaseModel):
    shoulder: str = Field("", description="Shoulder fit assessment")
    chest: str = Field("", description="Chest fit assessment")
    waist: str = Field("", description="Waist fit assessment")
    overall: str = Field("", description="Overall fit score")
    recommendation: str = Field("", description="Recommended size recommendation")

class ResultResponse(BaseModel):
    status: str = Field(..., description="Job status: processing, completed, failed")
    outputImageUrl: Optional[str] = Field(None, alias="outputImageUrl", description="HTTP reachable static URL of try-on output image")
    output_image_url: Optional[str] = Field(None, description="Alternative snake_case for backward compatibility")
    meshUrl: Optional[str] = Field(None, alias="meshUrl", description="HTTP reachable static URL of generated 3D body GLTF model")
    mesh_url: Optional[str] = Field(None, description="Alternative snake_case for meshUrl")
    measurements: Optional[MeasurementsSchema] = Field(None, description="Extracted body size measurements")
    fitAnalysis: Optional[FitAnalysisSchema] = Field(None, alias="fitAnalysis", description="Intelligent size fit assessment")

    class Config:
        populate_by_name = True
