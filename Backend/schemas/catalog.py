from pydantic import BaseModel, Field
from typing import List, Optional

class ClothingItemSchema(BaseModel):
    id: str = Field(..., description="Unique ID of the catalog clothing item")
    name: str = Field(..., description="Descriptive display name of the item")
    category: str = Field(..., description="Category of the clothing, e.g. T-Shirt, Shirt, Dress")
    imageUrl: str = Field(..., alias="imageUrl", description="Statically served URL path to the transparent PNG")
    sizes: List[str] = Field(default=["S", "M", "L", "XL"], description="List of available sizes")
    brand: str = Field(..., description="Brand of the item")
    gender: str = Field(..., description="Target gender: men or women")

    class Config:
        populate_by_name = True
        json_encoders = {
            # Ensure alias serialization is supported
        }
