from fastapi import APIRouter, Query, HTTPException, status
from typing import List, Optional
from schemas.catalog import ClothingItemSchema
import os
import json

router = APIRouter(tags=["catalog"])

@router.get("/clothing-catalog", response_model=List[ClothingItemSchema])
def get_clothing_catalog(
    category: Optional[str] = Query(None, description="Filter catalog items by category, e.g. T-Shirt, Shirt"),
    gender: Optional[str] = Query(None, description="Filter catalog items by gender: men or women")
):
    """
    Exposes the catalog of preprocessed transparent clothing items.
    Allows filtering by category and gender for Jetpack Compose UI.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "clothing", "catalog.json"))
    
    if not os.path.exists(catalog_path):
        # Fallback: return empty list if catalog.json hasn't been generated yet
        return []
        
    try:
        with open(catalog_path, "r") as f:
            items = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read catalog database: {str(e)}"
        )
        
    filtered_items = []
    for item in items:
        # Filter by category (case-insensitive)
        if category and category.lower() != item.get("category", "").lower():
            continue
        # Filter by gender (case-insensitive)
        if gender and gender.lower() != item.get("gender", "").lower():
            continue
            
        filtered_items.append(item)
        
    return filtered_items
