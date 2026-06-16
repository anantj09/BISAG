/**
 * Gamepad Bluetooth controller handling for WebXR.
 * Polls gamepad inputs and applies modifications to the modelContainer.
 */

let prevButtons = [];

/**
 * Polls gamepad inputs and applies controls to the model container.
 * Should be called inside the main Three.js animation / render loop.
 * 
 * @param {THREE.Group} modelContainer - The container holding the formatted model.
 * @param {THREE.WebGLRenderer} renderer - The active Three.js WebGLRenderer.
 * @param {HTMLElement} instructionsPanel - The HTML panel overlay containing user controls instructions.
 */
export function pollGamepad(modelContainer, renderer, instructionsPanel) {
    if (!navigator.getGamepads) return;
    
    const gamepads = navigator.getGamepads();
    let activeGamepad = null;

    // Find the first available gamepad
    for (let i = 0; i < gamepads.length; i++) {
        if (gamepads[i] && gamepads[i].connected) {
            activeGamepad = gamepads[i];
            break;
        }
    }

    if (!activeGamepad) return;

    // 1. Deadzone check
    const deadzone = 0.1;

    // 2. Axis 0: Horizontal left stick -> Rotate model Container
    if (Math.abs(activeGamepad.axes[0]) > deadzone) {
        modelContainer.rotation.y += activeGamepad.axes[0] * 0.03;
    }

    // 3. Axis 1: Vertical stick (left or right) -> Scale model Container (limit 0.3x to 3.0x)
    if (Math.abs(activeGamepad.axes[1]) > deadzone) {
        // Up is negative, Down is positive on typical gamepad layout.
        // We invert it so pushing UP scales UP, pushing DOWN scales DOWN.
        const scaleDelta = -activeGamepad.axes[1] * 0.02;
        let newScale = modelContainer.scale.x + scaleDelta;
        newScale = Math.max(0.3, Math.min(3.0, newScale));
        modelContainer.scale.set(newScale, newScale, newScale);
    }

    // Initialize button state cache if lengths don't match
    if (prevButtons.length !== activeGamepad.buttons.length) {
        prevButtons = activeGamepad.buttons.map(() => false);
    }

    // Helper function to check if a button is currently pressed
    const isButtonPressed = (index) => {
        if (index >= activeGamepad.buttons.length) return false;
        const button = activeGamepad.buttons[index];
        return typeof button === 'object' ? button.pressed : button === 1.0;
    };

    // 4. Button 0 (A/Cross): Reset model rotation and scale
    if (isButtonPressed(0) && !prevButtons[0]) {
        modelContainer.rotation.set(0, 0, 0);
        modelContainer.scale.set(1, 1, 1);
    }

    // 5. Button 1 (B/Circle): Terminate the active VR session
    if (isButtonPressed(1) && !prevButtons[1]) {
        const session = renderer.xr.getSession();
        if (session) {
            session.end();
        }
    }

    // 6. Button 2 (X/Square): Toggle instructions panel UI overlay visibility
    if (isButtonPressed(2) && !prevButtons[2]) {
        if (instructionsPanel) {
            const currentDisplay = window.getComputedStyle(instructionsPanel).display;
            instructionsPanel.style.display = currentDisplay === 'none' ? 'block' : 'none';
        }
    }

    // Save button states for single-trigger tracking in the next frame
    prevButtons = activeGamepad.buttons.map((button) => {
        return typeof button === 'object' ? button.pressed : button === 1.0;
    });
}
