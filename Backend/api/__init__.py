from fastapi import APIRouter
from api.routes.mesh import router as mesh_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(mesh_router)

