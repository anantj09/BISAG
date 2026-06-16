import sys
import numpy as np
# 1. NumPy compatibility for chumpy (older pickle files)
np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
np.unicode = str

import torch
if not hasattr(torch.load, '__monkeypatched__'):
    original_load = torch.load
    def safe_load(*args, **kwargs):
        try:
            return original_load(*args, **kwargs)
        except Exception as e:
            if kwargs.get('weights_only', True) is not False:
                if len(args) > 0:
                    f = args[0]
                    if hasattr(f, 'seek') and callable(f.seek):
                        try:
                            f.seek(0)
                        except Exception:
                            pass
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

from fastapi import FastAPI, Request, HTTPException as FastAPIHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
from core.config import settings
from api import api_router
import os
import time
import asyncio

# Rate limiting state: Client IP -> list of request timestamps
rate_limit_records = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure the temp upload directory exists
    temp_path = settings.temp_dir_path
    os.makedirs(temp_path, exist_ok=True)
    
    # Pre-initialize 4D-Humans networks and detectors to warm up the GPU/CPU memory
    try:
        from services.mesh_service import mesh_service
        mesh_service.initialize_models()
    except Exception as e:
        print(f"[STARTUP WARNING] Local 4D-Humans initialization deferred: {str(e)}")
        print("Model will attempt to lazy-load on the first incoming request.")
        
    # Define and start the asynchronous stale file clean up scavenger
    async def cleanup_scavenger():
        while True:
            try:
                current_time = time.time()
                retention_period = getattr(settings, "CACHE_RETENTION_SECONDS", 7200)
                
                # Scavenge temp upload folder
                if os.path.exists(temp_path):
                    for filename in os.listdir(temp_path):
                        file_path = os.path.join(temp_path, filename)
                        if os.path.isfile(file_path):
                            file_mtime = os.path.getmtime(file_path)
                            if current_time - file_mtime > retention_period:
                                os.remove(file_path)
                                
                # Scavenge static preprocessed result assets
                current_dir = os.path.dirname(os.path.abspath(__file__))
                results_dir = os.path.abspath(os.path.join(current_dir, "..", "data", "results"))
                if os.path.exists(results_dir):
                    for filename in os.listdir(results_dir):
                        file_path = os.path.join(results_dir, filename)
                        if os.path.isfile(file_path):
                            file_mtime = os.path.getmtime(file_path)
                            if current_time - file_mtime > retention_period:
                                os.remove(file_path)
            except Exception as e:
                # Robust logger fallback
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cleanup Scavenger Error: {str(e)}")
                
            # Run cleaner loop hourly
            await asyncio.sleep(3600)
            
    cleanup_task = asyncio.create_task(cleanup_scavenger())
    yield
    # Shutdown: Cancel the scavenger task cleanly
    cleanup_task.cancel()

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Backend API for AR-Based Virtual Clothing Try-On WebXR App",
    lifespan=lifespan
)

# 1. Register global Rate Limiting Middleware
@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    path = request.url.path
    if path in ["/api/v1/try-on", "/api/v1/upload-photo"]:
        # Bypass rate limiting in test suite unless X-Test-Rate-Limit is passed
        is_test = getattr(settings, "ENVIRONMENT", "dev") == "test"
        has_force_header = "x-test-rate-limit" in request.headers
        if is_test and not has_force_header:
            return await call_next(request)
            
        client_ip = request.client.host if request.client else "unknown_ip"
        current_time = time.time()
        
        if client_ip not in rate_limit_records:
            rate_limit_records[client_ip] = []
            
        # Filter request timestamps inside standard 60-second sliding window
        rate_limit_records[client_ip] = [t for t in rate_limit_records[client_ip] if current_time - t < 60]
        
        # Max limit from config
        max_requests = getattr(settings, "RATE_LIMIT_PER_MIN", 3)
        
        if len(rate_limit_records[client_ip]) >= max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too Many Requests. Heavy AI try-on pipeline is rate-limited to 3 requests per minute per IP to protect server resources."
                },
                headers={
                    "Retry-After": "60"
                }
            )
            
        rate_limit_records[client_ip].append(current_time)
        
    return await call_next(request)

# 2. CORS configurations for physical mobile connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Register global API router
app.include_router(api_router)

# 4. Mount central static catalog directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
os.makedirs(data_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=data_dir), name="static")

# 5. Global Exception Handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "HTTPException"}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc.errors()), "type": "RequestValidationError"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected server error occurred: {str(exc)}", "type": "GlobalException"}
    )

@app.get("/health", tags=["health"])
def health_check():
    return {
        "status": "ok",
        "message": "AR Try-On backend is running",
        "environment": settings.ENVIRONMENT
    }
