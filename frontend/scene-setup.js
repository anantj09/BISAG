import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export let scene, camera, renderer, controls, modelContainer;

export function initScene(canvasContainer) {
    // 1. Create Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080810);
    scene.fog = new THREE.FogExp2(0x080810, 0.08);

    // 2. Create Camera
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
    // Start camera at standard standing eye height (1.6m) looking towards model position (0, 0.9, -1.5)
    camera.position.set(0, 1.6, 1.0);

    // 3. Create WebGLRenderer with WebXR & Shadows enabled
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.xr.enabled = true;
    
    // Modern Color Space & High Dynamic Range (HDR) Tone Mapping
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0; // Natural exposure level
    
    canvasContainer.appendChild(renderer.domElement);

    // 4. Create OrbitControls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.9, -1.5); // Target model chest level
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.1; // Don't go below floor level
    controls.minDistance = 0.5;
    controls.maxDistance = 10;
    controls.update();

    // 5. Create Model Container Group at target Z = -1.5
    modelContainer = new THREE.Group();
    modelContainer.position.set(0, 0, -1.5);
    scene.add(modelContainer);

    // 6. Lighting Setup
    // Ambient Light (subtle fill)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.25);
    scene.add(ambientLight);

    // Hemisphere Light (natural sky/ground environment bounce)
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x080810, 0.45);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    // Key Directional Light with Shadows (side-angle)
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(3, 5, 3);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 15;
    dirLight.shadow.camera.left = -2;
    dirLight.shadow.camera.right = 2;
    dirLight.shadow.camera.top = 2.5;
    dirLight.shadow.camera.bottom = -1;
    dirLight.shadow.bias = -0.0005;
    dirLight.target = modelContainer; // Point directly at model container
    scene.add(dirLight);

    // Front Fill Directional Light (illuminates face and frontal details)
    const frontLight = new THREE.DirectionalLight(0xffffff, 0.7);
    frontLight.position.set(0, 1.5, 2); // Keep close but reduce intensity for natural fill
    frontLight.target = modelContainer; // Point directly at model container
    scene.add(frontLight);

    // Purple Point Light at back-left for sharp neon contour highlights
    const pointLight = new THREE.PointLight(0x7b61ff, 6.0, 10, 1.0);
    pointLight.position.set(-2, 2, -1);
    scene.add(pointLight);

    // 7. Ground / Scale Environment
    // Grid Helper
    const gridHelper = new THREE.GridHelper(30, 30, 0x7b61ff, 0x181828);
    gridHelper.position.y = 0;
    scene.add(gridHelper);

    // Shadow receiving ground plane
    const floorGeo = new THREE.PlaneGeometry(100, 100);
    const floorMat = new THREE.ShadowMaterial({ opacity: 0.5 });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    floor.receiveShadow = true;
    scene.add(floor);

    // 8. Resize handler
    window.addEventListener('resize', onWindowResize);
}

// Window resizing
export function onWindowResize() {
    if (!camera || !renderer) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}
