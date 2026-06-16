import { renderer, camera, controls, onWindowResize } from './scene-setup.js';
import { updateStatus } from './ui-handlers.js';

export const vrButton = document.getElementById('enter-vr-btn');

// Set up WebXR Button state and action
export function setupWebXR() {
    if ('xr' in navigator) {
        navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
            if (supported) {
                vrButton.disabled = false;
                vrButton.classList.remove('btn-secondary');
                vrButton.addEventListener('click', onEnterVR);
                updateStatus("Disconnected (Ready for VR)", "active");
            } else {
                vrButton.textContent = "VR NOT SUPPORTED";
                updateStatus("System Ready (WebXR Simulator / Flat Mode)", "active");
            }
        }).catch((err) => {
            console.error("WebXR session check error:", err);
            vrButton.textContent = "VR ERROR";
        });
    } else {
        vrButton.textContent = "XR NOT AVAILABLE";
        updateStatus("System Ready (Flat Desktop Mode)", "active");
    }
}

// Handle VR Session Request
export function onEnterVR() {
    const sessionInit = { optionalFeatures: ['local-floor', 'bounded-floor'] };
    navigator.xr.requestSession('immersive-vr', sessionInit).then((session) => {
        renderer.xr.setSession(session);
        
        // Disable OrbitControls in VR session to prevent conflicts
        controls.enabled = false;
        updateStatus("Active Immersive VR Session", "active");

        session.addEventListener('end', () => {
            // Re-enable controls when exiting VR
            controls.enabled = true;
            
            // Re-align desktop camera
            camera.position.set(0, 1.6, 1.0);
            controls.target.set(0, 0.9, -1.5);
            controls.update();
            updateStatus("Disconnected (Ready for VR)", "active");

            // Force browser to recalculate viewport scale and scroll offset
            resetViewportMeta();
            window.scrollTo(0, 0);
            document.body.scrollTop = 0;
            document.body.scrollLeft = 0;
            document.documentElement.scrollTop = 0;
            document.documentElement.scrollLeft = 0;
            onWindowResize();
        });
    }).catch((err) => {
        console.error("Failed to start VR session:", err);
        alert("Could not start VR session: " + err.message);
    });
}

// Helper to reset viewport meta to fix scaling bugs on VR exit
export function resetViewportMeta() {
    let meta = document.querySelector('meta[name="viewport"]');
    if (!meta) {
        meta = document.createElement('meta');
        meta.name = 'viewport';
        document.head.appendChild(meta);
    }
    // Set to standard scale parameters
    meta.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, shrink-to-fit=no');
}
