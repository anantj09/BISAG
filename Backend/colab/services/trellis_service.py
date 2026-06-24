import os
import gc
import sys
from PIL import Image
from typing import Optional

# Ensure the cloned TRELLIS repository path is added to sys.path so we can import trellis2 and o_voxel
if os.path.exists("/content/TRELLIS"):
    if "/content/TRELLIS" not in sys.path:
        sys.path.insert(0, "/content/TRELLIS")

class TrellisService:
    def __init__(self):
        self.pipeline = None
        self.initialized = False

    def initialize_model(self):
        """
        Lazy loads the Microsoft TRELLIS.2 pipeline and weights from Hugging Face.
        Requires a CUDA-compatible GPU.
        """
        if self.initialized:
            return

        print(">>> Initializing Microsoft TRELLIS.2 Image-to-3D Pipeline...")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. TRELLIS.2 requires a CUDA-enabled GPU.")

        os.environ['SPCONV_ALGO'] = 'native'
        
        # Load the SOTA TRELLIS.2 pipeline with proper half-precision to fit within Colab VRAM limits
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Loading pipeline weights with precision: {dtype}...")
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS.2-4B",
            torch_dtype=dtype
        )
        self.pipeline.cuda()
        self.initialized = True
        print("--- TRELLIS.2 Pipeline successfully loaded on GPU ---")

    def generate_mesh(self, input_image_path: str, output_glb_path: str, remove_bg: bool = True) -> bool:
        """
        Runs background removal and TRELLIS.2 inference to generate a 3D GLB model.
        """
        try:
            # 1. Ensure model is loaded
            self.initialize_model()
            
            import torch
            from rembg import remove
            import o_voxel
            
            # 2. Open input image
            print(f"Loading input image: {input_image_path}")
            img = Image.open(input_image_path)
            
            # 3. Perform background removal if requested (TRELLIS works best with transparent backgrounds)
            if remove_bg:
                print("Removing background using rembg...")
                img = remove(img)
                # Ensure transparent pixels are treated properly (convert to RGBA)
                img = img.convert("RGBA")
                
            # 4. Run TRELLIS.2 pipeline
            print("Running TRELLIS.2 model inference (this may take 10-20 seconds)...")
            with torch.no_grad():
                outputs = self.pipeline.run(img)
                
            # 5. Extract mesh and post-process to GLB
            print("Post-processing 3D Mesh data to GLB container...")
            
            # Extract first mesh from outputs list
            mesh = outputs[0]
            
            # Convert to standard GLB with textures using o_voxel.postprocess.to_glb
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], # standard unit bounding box
                decimation_target=150000,                  # Decimates triangles slightly for web optimization
                texture_size=2048,                          # High-resolution texture maps
                remesh=True,
                verbose=True
            )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
            
            # Save the GLB file
            glb.export(output_glb_path)
            print(f"--- Photorealistic GLB compiled successfully: {output_glb_path} ---")
            
            # 6. Garbage collection to prevent CUDA OOM
            del outputs, glb
            gc.collect()
            torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] TRELLIS.2 mesh generation pipeline failure: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# Global instance of TrellisService
trellis_service = TrellisService()
