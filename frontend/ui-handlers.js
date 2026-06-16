import { modelContainer, onWindowResize } from './scene-setup.js';
import { loadModel } from './model-loader.js';

// DOM Elements
export const uploadContainer = document.getElementById('upload-container');
export const uploadBox = document.getElementById('upload-box');
export const fileInput = document.getElementById('file-input');
export const instructionsPanel = document.getElementById('instructions-panel');
export const measurementsPanel = document.getElementById('measurements-panel');
export const statusDot = document.getElementById('status-dot');
export const statusText = document.getElementById('status-text');
export const modelInfoText = document.getElementById('model-info-text');

const valHeight = document.getElementById('val-height');
const valChest = document.getElementById('val-chest');
const valWaist = document.getElementById('val-waist');
const valHip = document.getElementById('val-hip');

export let currentModel = null;

// Helper to update status indicators
export function updateStatus(text, stateClass) {
    if (statusText) statusText.textContent = text;
    if (statusDot) statusDot.className = "status-dot " + stateClass;
}

// Clean up existing model
export function removeCurrentModel() {
    if (currentModel) {
        modelContainer.remove(currentModel);
        currentModel.traverse((child) => {
            if (child.isMesh) {
                child.geometry.dispose();
                if (Array.isArray(child.material)) {
                    child.material.forEach(mat => mat.dispose());
                } else if (child.material) {
                    child.material.dispose();
                }
            }
        });
        currentModel = null;
        modelInfoText.textContent = "No model active";
        
        // Reset status indicator to standard ready state
        if ('xr' in navigator) {
            navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
                if (supported) {
                    updateStatus("Disconnected (Ready for VR)", "active");
                } else {
                    updateStatus("System Ready (Flat Mode)", "active");
                }
            }).catch(() => {
                updateStatus("System Ready (Flat Mode)", "active");
            });
        } else {
            updateStatus("System Ready (Flat Mode)", "active");
        }
    }
}

// Measurements UI panel handlers
export function showMeasurementsPanel(measurements) {
    if (!measurementsPanel) return;
    if (valHeight) valHeight.textContent = `${Math.round(measurements.heightCm)}cm`;
    if (valChest) valChest.textContent = `${Math.round(measurements.chestCm)}cm`;
    if (valWaist) valWaist.textContent = `${Math.round(measurements.waistCm)}cm`;
    if (valHip) valHip.textContent = `${Math.round(measurements.hipCm)}cm`;
    measurementsPanel.style.display = 'block';
}

export function hideMeasurementsPanel() {
    if (measurementsPanel) {
        measurementsPanel.style.display = 'none';
    }
}

// Model File Handler (now uploads photo to backend)
export function handleModelFile(file) {
    if (!file) return;

    // Validate that the file is an image (allow fallback to extension check if mime type is missing/generic on mobile)
    const isImage = file.type.startsWith('image/') || /\.(jpe?g|png|gif|webp|heic|heif)$/i.test(file.name);
    if (!isImage) {
        alert("Please upload a valid image file (PNG or JPEG).");
        return;
    }

    updateStatus("Uploading photo... 0%", "loading");
    removeCurrentModel();
    hideMeasurementsPanel();

    // Hide upload box while showing the scene
    uploadContainer.style.display = 'none';

    // Prepare Multipart form upload
    const formData = new FormData();
    formData.append('photo', file);

    const backendHost = window.location.hostname;
    const backendUrl = `http://${backendHost}:8000/api/v1/generate-mesh`;

    const xhr = new XMLHttpRequest();
    xhr.open('POST', backendUrl);

    // Track upload progress
    xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            updateStatus(`Uploading photo... ${percent}%`, "loading");
        }
    });

    // Handle completed request
    xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            try {
                const data = JSON.parse(xhr.responseText);
                const meshUrl = data.meshUrl;
                const measurements = data.measurements;
                
                // Resolve absolute URL dynamically based on host
                const absoluteMeshUrl = meshUrl.startsWith('http') ? meshUrl : `http://${backendHost}:8000${meshUrl}`;
                
                updateStatus("Downloading generated 3D twin...", "loading");

                loadModel(
                    absoluteMeshUrl,
                    modelContainer,
                    (model) => {
                        currentModel = model;
                        updateStatus("Model Rendered (1.8m Aligned)", "active");
                        
                        // Show measurements HUD
                        showMeasurementsPanel(measurements);

                        // Update model info text
                        modelInfoText.innerHTML = `
                            Model: Generated Digital Twin<br>
                            Source Photo: ${file.name}<br>
                            Polygons: ${countPolygons(model).toLocaleString()} faces
                        `;
                        
                        showLoadAnotherButton();
                    },
                    (progress) => {
                        if (progress.total > 0) {
                            const percent = Math.round((progress.loaded / progress.total) * 100);
                            updateStatus(`Downloading model... ${percent}%`, "loading");
                        }
                    },
                    (error) => {
                        console.error("Error loading generated model:", error);
                        updateStatus("Load Failed", "loading");
                        alert("Failed to render the generated 3D model. Verify the mesh file was compiled correctly.");
                        uploadContainer.style.display = 'flex';
                    }
                );
            } catch (err) {
                console.error("Failed to parse backend response:", err);
                triggerFallback("Failed to compile server response.");
            }
        } else {
            console.warn(`Server error (${xhr.status}):`, xhr.responseText);
            triggerFallback(`Server error (${xhr.status})`);
        }
    });

    // Handle connection failures
    xhr.addEventListener('error', () => {
        console.warn("XHR Connection failed, falling back.");
        triggerFallback("Network connection error");
    });

    xhr.addEventListener('timeout', () => {
        console.warn("XHR Upload timed out.");
        triggerFallback("Connection timed out");
    });

    // Helper to run fallback
    function triggerFallback(reason) {
        console.warn(`Reconstruction failed (${reason}), loading local demo model...`);
        updateStatus("Connection failed. Loading demo mesh...", "loading");
        
        const sampleUrl = './data/demo/9dc6216c-3ba3-403b-9fc8-1783c828b8c4_mesh.gltf';
        loadModel(
            sampleUrl,
            modelContainer,
            (model) => {
                currentModel = model;
                updateStatus("Model Loaded (Backend Offline)", "active");
                
                const mockMeasurements = {
                    heightCm: 172.5,
                    chestCm: 92.0,
                    waistCm: 80.0,
                    hipCm: 95.0
                };
                showMeasurementsPanel(mockMeasurements);

                modelInfoText.innerHTML = `
                    File: 9dc6216c_mesh.gltf (Demo)<br>
                    Note: Backend offline (${reason}).<br>
                    Polygons: ${countPolygons(model).toLocaleString()} faces
                `;
                
                showLoadAnotherButton();
            },
            null,
            (err) => {
                console.error("Demo fallback failed:", err);
                updateStatus("Load Failed", "loading");
                alert("Connection to backend failed and demo fallback could not be loaded.");
                uploadContainer.style.display = 'flex';
            }
        );
    }

    // Set status to processing once upload completes
    xhr.upload.addEventListener('load', () => {
        updateStatus("Running 3D AI Reconstruction...", "loading");
    });

    xhr.send(formData);
}

// Sample Model Loader
export function handleSampleModel() {
    updateStatus("Loading Demo Model...", "loading");
    removeCurrentModel();
    hideMeasurementsPanel();
    uploadContainer.style.display = 'none';

    const sampleUrl = './data/demo/9dc6216c-3ba3-403b-9fc8-1783c828b8c4_mesh.gltf';

    loadModel(
        sampleUrl,
        modelContainer,
        (model) => {
            currentModel = model;
            updateStatus("Model Rendered (1.8m Aligned)", "active");
            
            // Show measurements HUD with sample mock measurements
            const mockMeasurements = {
                heightCm: 172.5,
                chestCm: 92.0,
                waistCm: 80.0,
                hipCm: 95.0
            };
            showMeasurementsPanel(mockMeasurements);

            modelInfoText.innerHTML = `
                File: 9dc6216c_mesh.gltf (Demo)<br>
                Render Scale: Y=1.8m (Scaled)<br>
                Polygons: ${countPolygons(model).toLocaleString()} faces
            `;
            
            showLoadAnotherButton();
        },
        (progress) => {
            if (progress.total > 0) {
                const percent = Math.round((progress.loaded / progress.total) * 100);
                updateStatus(`Loading... ${percent}%`, "loading");
            }
        },
        (error) => {
            console.error("Error loading sample model:", error);
            updateStatus("Load Failed", "loading");
            alert("Failed to load demo mesh. Check console logs.");
            uploadContainer.style.display = 'flex';
        }
    );
}

// Helper to count faces/polygons in model
function countPolygons(object) {
    let count = 0;
    object.traverse((child) => {
        if (child.isMesh && child.geometry.index) {
            count += child.geometry.index.count / 3;
        } else if (child.isMesh && child.geometry.attributes.position) {
            count += child.geometry.attributes.position.count / 3;
        }
    });
    return count;
}

// Floating HUD "Load Another" button controller
export function showLoadAnotherButton() {
    let btn = document.getElementById('load-another-btn');
    if (!btn) {
        btn = document.createElement('button');
        btn.id = 'load-another-btn';
        btn.className = 'btn btn-secondary interactive';
        btn.textContent = 'Upload New Twin';
        btn.style.marginTop = '12px';
        btn.addEventListener('click', () => {
            removeCurrentModel();
            hideMeasurementsPanel();
            uploadContainer.style.display = 'flex';
            btn.style.display = 'none';
            // Reset container scale/rotation
            modelContainer.rotation.set(0, 0, 0);
            modelContainer.scale.set(1, 1, 1);

            // Force layout and scale correction on new upload trigger
            window.scrollTo(0, 0);
            document.body.scrollTop = 0;
            document.body.scrollLeft = 0;
            document.documentElement.scrollTop = 0;
            document.documentElement.scrollLeft = 0;
            onWindowResize();
        });
        document.querySelector('.status-panel').appendChild(btn);
    }
    btn.style.display = 'inline-flex';
}

// Event Listeners for UI interaction
export function setupUI() {
    // Load sample model
    const loadSampleBtn = document.getElementById('load-sample-btn');
    if (loadSampleBtn) {
        loadSampleBtn.addEventListener('click', (e) => {
            e.preventDefault();  // prevent label activation on mobile
            e.stopPropagation(); // prevent triggering fileInput click
            handleSampleModel();
        });
    }
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleModelFile(e.target.files[0]);
            // Clear input value so that the user can select the same file again if needed
            fileInput.value = '';
        }
    });

    // Drag and drop support
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('drag-over');
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('drag-over');
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleModelFile(e.dataTransfer.files[0]);
        }
    });

    // Collapsible Navigation Controls Panel
    if (instructionsPanel) {
        const instructionsList = instructionsPanel.querySelector('ul');
        const toggleIcon = instructionsPanel.querySelector('.toggle-icon');

        instructionsPanel.addEventListener('click', () => {
            const isCollapsed = instructionsPanel.classList.contains('collapsed');
            if (isCollapsed) {
                instructionsPanel.classList.remove('collapsed');
                instructionsList.style.display = 'block';
                toggleIcon.textContent = '▲';
            } else {
                instructionsPanel.classList.add('collapsed');
                instructionsList.style.display = 'none';
                toggleIcon.textContent = '▼';
            }
        });
    }
}
