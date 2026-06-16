from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
from schemas.catalog import ClothingItemSchema
from api.routes.tryon import user_profiles_db, jobs_db
import os
import json

router = APIRouter(tags=["recommendations"])

@router.get("/recommend/{userId}", response_model=List[ClothingItemSchema])
def get_personalized_recommendations(userId: str):
    """
    Serves highly personalized AI clothing recommendations.
    Uses pose measurements (Chest, Height) to rank catalog items.
    Fallback: Serves the best 4 Men's items + the best 4 Women's items (total 8 items) for new profiles.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "data", "clothing", "catalog.json"))
    
    if not os.path.exists(catalog_path):
        return []
        
    try:
        with open(catalog_path, "r") as f:
            catalog = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read catalog database: {str(e)}"
        )
        
    user_profile = user_profiles_db.get(userId)
    
    # 1. FALLBACK MODE: User is new or has no dynamic try-on history yet
    # Returns best 4 Men's items and best 4 Women's items (total 8 items)
    if not user_profile:
        men_items = [item for item in catalog if item.get("gender") == "men"][:4]
        women_items = [item for item in catalog if item.get("gender") == "women"][:4]
        return men_items + women_items

    # 2. DYNAMIC AI RECOMMENDATIONS MODE (Serves up to 10 customized ranked items)
    measurements = user_profile["measurements"]
    chest = measurements.get("chestCm", 90.0)
    height = measurements.get("heightCm", 170.0)
    
    # Classify body silhouette profile
    # - Slim Fit: (Chest < 90cm)
    # - Regular Fit: (90cm <= Chest <= 100cm)
    # - Relaxed/Oversized Fit: (Chest > 100cm)
    fit_profile = "Regular Fit"
    if chest > 100.0:
        fit_profile = "Relaxed Fit"
    elif chest < 90.0:
        fit_profile = "Slim Fit"
        
    # Detect target gender profile based on their last tried clothing item inside jobs_db
    preferred_gender = None
    for job in jobs_db.values():
        if job.get("user_id") == userId and job.get("status") == "completed":
            # Extract gender preference from catalog PNG name if possible
            out_img = job.get("output_image_url", "")
            if "women" in out_img.lower():
                preferred_gender = "women"
                break
            elif "men" in out_img.lower():
                preferred_gender = "men"
                break
                
    scored_items = []
    for item in catalog:
        item_gender = item.get("gender", "")
        item_category = item.get("category", "")
        item_brand = item.get("brand", "")
        
        score = 0
        
        # Priority A: Match preferred gender preference
        if preferred_gender and item_gender == preferred_gender:
            score += 10
            
        # Priority B: Fit profile matching
        if fit_profile == "Slim Fit":
            # Slim people look best in tailored/slim Shirts and fitted Tops
            if item_category in ["Shirt", "Top"]:
                score += 5
            if item_brand in ["Roadster", "Mast & Harbour"]:
                score += 3
        elif fit_profile == "Relaxed Fit":
            # Larger or athletic people look best in comfort T-Shirts and stretchable sportswear
            if item_category in ["T-Shirt"]:
                score += 5
            if item_brand in ["HRX by Hrithik Roshan", "WROGN"]:
                score += 4
        else: # Regular Fit
            # Regular fit is highly compatible, suggest balanced T-shirts/casual Shirts
            if item_category in ["T-Shirt", "Shirt"]:
                score += 4
            if item_brand in ["Myntra Select", "Mast & Harbour"]:
                score += 2
                
        # Height adjustments: Kurta and Dress garments drape beautifully on taller subjects
        if height > 175.0 and item_category in ["Dress", "Shirt"]:
            score += 3
            
        scored_items.append((score, item))
        
    # Sort catalog items by score descending
    scored_items.sort(key=lambda x: x[0], reverse=True)
    
    # Return top 10 recommended items
    recommendations = [item[1] for item in scored_items][:10]
    return recommendations
