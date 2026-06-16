import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/**
 * Loads a GLB/GLTF model, formats it according to 4D-Humans requirements,
 * and adds it to the target parent group.
 * 
 * @param {string} url - Object URL or network URL of the model.
 * @param {THREE.Group} parentGroup - The parent group (e.g. modelContainer) to attach the model.
 * @param {Function} onLoad - Callback triggered when loading completes, receiving the loaded model.
 * @param {Function} onProgress - Progress callback.
 * @param {Function} onError - Error callback.
 */
export function loadModel(url, parentGroup, onLoad, onProgress, onError) {
    const loader = new GLTFLoader();
    
    loader.load(
        url,
        (gltf) => {
            const model = gltf.scene;

            // 1. Centering
            const box = new THREE.Box3().setFromObject(model);
            const center = new THREE.Vector3();
            box.getCenter(center);
            model.position.sub(center);

            // 2. Scaling to Life Size (exactly 1.8 units / meters)
            const size = new THREE.Vector3();
            box.getSize(size);
            const height = size.y;
            const scaleFactor = 1.8 / (height || 1.8);
            model.scale.set(scaleFactor, scaleFactor, scaleFactor);

            // Ensure matrices are updated before computing the scaled bounding box
            model.updateMatrixWorld(true);

            // 3. Floor & Camera Alignment
            const scaledBox = new THREE.Box3().setFromObject(model);
            
            // Align base (minimum Y of scaled bounding box) to ground (Y = 0)
            model.position.y += -scaledBox.min.y;
            
            // Center the model horizontally relative to its parent group (X = 0, Z = 0)
            const currentCenter = new THREE.Vector3();
            new THREE.Box3().setFromObject(model).getCenter(currentCenter);
            model.position.x += -currentCenter.x;
            model.position.z += -currentCenter.z;

            // 4. Shadows & Material Settings
            model.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                    if (child.material) {
                        child.material.shadowSide = THREE.DoubleSide;
                        
                        // Optimize roughness and metalness to make materials look natural under lights
                        if (child.material.isMeshStandardMaterial) {
                            child.material.roughness = 0.55; // Prevent flat or excessively shiny look
                            child.material.metalness = 0.05; // Prevent metallic reflections on human skin/clothing
                        }
                    }
                }
            });

            // Add model to parentGroup
            parentGroup.add(model);

            if (onLoad) onLoad(model);
        },
        onProgress,
        onError
    );
}
