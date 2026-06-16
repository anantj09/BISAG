import { initScene, renderer, camera, scene, controls, modelContainer } from './scene-setup.js';
import { setupUI, instructionsPanel } from './ui-handlers.js';
import { setupWebXR } from './xr-manager.js';
import { pollGamepad } from './gamepad.js';

// Get viewport canvas container
const canvasContainer = document.getElementById('canvas-container');

// Main Render Loop (used by Three.js WebXR)
function renderLoop() {
    // 1. Update controls in desktop mode
    if (controls.enabled) {
        controls.update();
    }

    // 2. Poll Bluetooth Gamepad controller inputs
    pollGamepad(modelContainer, renderer, instructionsPanel);

    // 3. Render frame
    renderer.render(scene, camera);
}

// Initialize application components
function startApp() {
    // 1. Initialize scene/camera/renderer
    initScene(canvasContainer);
    
    // 2. Set up WebXR Button state
    setupWebXR();
    
    // 3. Set up UI event listeners
    setupUI();
    
    // 4. Start Three.js Animation Loop
    renderer.setAnimationLoop(renderLoop);
}

// Launch application
startApp();
