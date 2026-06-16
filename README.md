# Immersive 3D Digital Twin WebXR Viewer

An immersive, hardware-accelerated WebXR browser application integrated with a Python AI backend to reconstruct and inspect customized 3D human digital twins from single-view portrait images.

---

## 1. System Architecture & Workings

```mermaid
graph TD
    %% Frontend Interaction
    subgraph Frontend [Client Browser - Three.js & WebGL]
        UI[HTML/CSS HUD Dashboard] -- 1. Upload Photo / Trigger Demo --> Upload[XMLHttpRequest Upload Engine]
        Canvas[WebGL Canvas] -- 5. Render 3D Scene / OrbitControls --> Viewport[Screen Viewport]
        XR[WebXR Device API] -- 6. Split Stereo Projections --> VR[VR Headset / Mobile Cardboard]
        Gamepad[Gamepad API] -- 7. Bluetooth Controls --> Canvas
    end

    %% Backend Processing
    subgraph Backend [FastAPI Server - Python]
        API[FastAPI Router API] -- 2. Receives Multipart Image --> SegService[Segmentation Service]
        SegService -- 3. Runs SAM on CPU --> Crop[Person Crop & Mask]
        Crop -- 4. Runs HMR2 on GPU --> MeshGen[HMR2 Reconstruction Engine]
        MeshGen -- a. Predicts SMPL Parameters --> SMPL[SMPL Neutral Template .pkl]
        MeshGen -- b. Computes Vertex Distances --> Measurements[AI Body Sizing Measurements]
        MeshGen -- c. Formats Triangles --> GLTF[GLTF Mesh File Exporter]
    end

    %% Communications
    Upload -- HTTP POST --> API
    GLTF -- HTTP GET Mesh URL --> Upload
    Measurements -- JSON Response --> UI
```

### Frontend Technology Stack (Three.js, WebGL & WebXR)
* **WebGL:** Communicates directly with the GPU using compiled shaders to draw human meshes consisting of thousands of triangles at high framerates.
* **Three.js:** Exposes clean object-oriented APIs to manage the scene graph, lighting nodes (Ambient fill, Hemisphere sky bounce, Directional key, and Neon پوائنٹ spotlights), dynamic shadow maps, and material calibrations.
* **WebXR Device API:** Negotiates high-refresh-rate stereoscopic viewpoints, manages inter-pupillary distance (IPD) offsets inside VR headsets, and polls gamepad controller triggers.

### Backend Technology Stack (4D-Humans & SMPL)
* **Segmentation Service:** Uses **Detectron2** to locate the subject and **Segment Anything Model (SAM)** (run on CPU to optimize VRAM footprint) to extract a clean background mask.
* **Feature Extraction:** Feeds the cropped human bounding box through a **ViT-Det (Vision Transformer)** backbone to extract detailed pose and body shape representations.
* **SMPL Body Template:** Predicts 10 shape parameters ($\beta$ PCA coefficients) and 72 pose parameters ($\theta$ joint rotations), deforming the 6,890 template vertices of [basicModel_neutral_lbs_10_207_0_v1.0.0.pkl](file:///c:/Coding_files/Internship3rdYEAR/VR_development/internship_week3/Backend/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl) to generate a customized mesh.
* **AI Sizing Calculations:** Computes Euclidean distances between specific indices of the deforming body vertices to estimate Height, Chest, Waist, and Hip circumferences in centimeters.

---

## 2. Directory Layout & Module Maps

```bash
internship_week3/
│
├── frontend/                     # Client application (Static Web Server Root)
│   ├── data/demo/                # Offline fallback meshes
│   │   └── 9dc6216c..._mesh.gltf # Local mannequin fallback model
│   ├── index.html                # HTML DOM structure and Importmaps
│   ├── styles.css                # Cyberpunk overlays and mobile layouts
│   ├── viewer.js                 # App Entry Point & Animation loop
│   ├── scene-setup.js            # Three.js core setups (lights, cameras, renderer)
│   ├── xr-manager.js             # WebXR session triggers and mobile resets
│   ├── ui-handlers.js            # Event listeners and XHR upload progress engine
│   ├── model-loader.js           # GLTF meshes alignment and surface calibration
│   └── gamepad.js                # VR bluetooth joystick polling
│
├── Backend/                      # Python FastAPI server
│   ├── api/routes/mesh.py        # Mesh POST generation route
│   ├── services/                 # AI reconstruction orchestrators
│   │   ├── segmentation_service.py # Runs SAM background isolation (CPU)
│   │   └── mesh_service.py       # Interacts with HMR2 and exports GLTF
│   ├── data/                     # Neutral SMPL template weights (.pkl)
│   ├── weights/                  # Pre-trained deep learning checkpoint weights
│   ├── main.py                   # FastAPI initialization
│   └── requirements.txt          # Python package dependency configurations
│
└── README.md                     # Project documentation
```

---

## 3. WebXR & VR Code Integration Locations
WebXR session setups are implemented inside [viewer.js](file:///c:/Coding_files/Internship3rdYEAR/VR_development/internship_week3/frontend/viewer.js) and [xr-manager.js](file:///c:/Coding_files/Internship3rdYEAR/VR_development/internship_week3/frontend/xr-manager.js):

* **WebGLRenderer WebXR Activation:** Enabled via `renderer.xr.enabled = true` inside `initScene()` in [scene-setup.js](file:///c:/Coding_files/Internship3rdYEAR/VR_development/internship_week3/frontend/scene-setup.js).
* **Hardware Support Checks:** Queries `navigator.xr.isSessionSupported('immersive-vr')` inside `setupWebXR()` in [xr-manager.js](file:///c:/Coding_files/Internship3rdYEAR/VR_development/internship_week3/frontend/xr-manager.js) to enable the VR button.
* **VR Presentation Session:** Calls `navigator.xr.requestSession('immersive-vr')` and feeds it into the renderer via `renderer.xr.setSession(session)`.
* **Animation Loop Compatibility:** Handled using `renderer.setAnimationLoop(renderLoop)` in [viewer.js](file:///c:/Coding_files/Internship3rdYEAR/VR_development/internship_week3/frontend/viewer.js) as standard `requestAnimationFrame` is blocked under immersive WebXR.

---

## 4. Local Setup & Installation

### A. Start the Python Backend (Port 8000)
1. Navigate into the `Backend` directory:
   ```cmd
   cd Backend
   ```
2. Activate your virtual environment:
   * **PowerShell:** `.venv/Scripts/Activate.ps1`
   * **Command Prompt (cmd):** `.venv\Scripts\activate.bat`
3. Run the FastAPI development server bound to all local interfaces:
   ```cmd
   .venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### B. Start the Frontend Web Server (Port 8080)
1. Open a separate terminal and navigate into the `frontend` folder:
   ```cmd
   cd frontend
   ```
2. Relaunch the static file server:
   ```cmd
   npx serve -l 8080
   ```

---

## 5. Mobile & Headset Connectivity Configurations

Because WebXR sensors require a **Secure Context**, browsers block VR/AR components over raw HTTP unless accessed via `localhost` or configured specifically.

### Method 1: Google Chrome Flags Bypass (Wireless)
1. Connect both your phone and laptop to the **same local Wi-Fi network**.
2. Note your laptop's local IP address (e.g. `192.168.1.15`).
3. On your mobile phone, open Chrome and navigate to: **`chrome://flags`**
4. Search for: **"Insecure origins treated as secure"**.
5. Enable the flag and input your laptop's address:
   `http://192.168.1.15:8080`
6. Relaunch Chrome. Open the page `http://192.168.1.15:8080` to interact.

### Method 2: USB Port Forwarding (Recommended - Network Independent)
1. Connect your mobile device to your laptop with a USB cable (with **USB Debugging** enabled in developer options).
2. Open Chrome on your laptop and go to: **`chrome://inspect`**
3. Click **Port forwarding...** and add two configurations:
   * Port `8080` $\rightarrow$ `localhost:8080`
   * Port `8000` $\rightarrow$ `localhost:8000`
4. Now, open **`http://localhost:8080`** directly on your phone. Chrome treats `localhost` as secure automatically, requiring no flags or IP updates if you change networks.

---

## 6. WebXR VR Gamepad Controller Mapping

When wearing a VR headset, you can control the digital twin using your controller thumbsticks:

* **Left Stick X-Axis (Horizontal):** Rotates the human twin on the Y-axis.
* **Left Stick Y-Axis (Vertical):** Scales the human twin size (range restricted from `0.3x` to `3.0x` for physics accuracy).
* **Button A (Index 0):** Resets the twin's rotation and scale back to baseline.
* **Button B (Index 1):** Terminates the immersive session programmatically to return to flat mode.
* **Button X (Index 2):** Toggles the visibility of the on-screen controls panel.
