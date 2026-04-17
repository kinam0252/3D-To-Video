# 3D-To-Video

Generate **3D-consistent realistic videos** of human-object interactions.

**Pipeline:** 3D Motion Capture → Blender Rendering → VACE V2V Diffusion

![Pipeline](https://img.shields.io/badge/Pipeline-MoCap→Blender→V2V-blue)
![Datasets](https://img.shields.io/badge/Datasets-OMOMO%20|%20HUMOTO-green)
![License](https://img.shields.io/badge/License-Research-yellow)

## Overview

This project builds a pipeline to:
1. Load human-object interaction data (OMOMO/HUMOTO datasets)
2. Render 3D scenes in Blender with orbit cameras (multiview)
3. Convert rendered animations to photorealistic video using VACE V2V diffusion

### Supported Datasets

| Dataset | Format | Human | Objects | Demo Samples |
|---------|--------|-------|---------|--------------|
| **OMOMO** (InterAct) | SMPLX `.npz` + `.obj` | Separate SMPLX body mesh | Separate OBJ meshes, keyframed | 3 sequences |
| **HUMOTO** | Animated `.glb` | Integrated in GLB | Integrated in GLB | 2 sequences |

---

## Prerequisites

Before running the demo, you need the following installed on your system:

### 1. System Dependencies

- **Linux** (tested on Ubuntu 22.04/24.04)
- **NVIDIA GPU** with drivers (for V2V; rendering works on CPU)
- **FFmpeg** (for video encoding)

```bash
# Check FFmpeg is installed
ffmpeg -version
# If not: sudo apt install ffmpeg
```

### 2. Blender 5.1+

Download and extract Blender (no installation needed):

```bash
# Download Blender 5.1.0
wget https://download.blender.org/release/Blender5.1/blender-5.1.0-linux-x64.tar.xz
tar -xf blender-5.1.0-linux-x64.tar.xz

# Verify
./blender-5.1.0-linux-x64/blender --version
# Should print: Blender 5.1.0
```

> **Note:** Blender is used headlessly (`--background`). No display server needed.

### 3. SMPLX Body Models (for OMOMO pipeline)

1. Register at [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)
2. Download the SMPLX model files (`SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`, `SMPLX_NEUTRAL.npz`)
3. Place them in a directory, e.g.:

```
models/smplx/
├── SMPLX_MALE.npz
├── SMPLX_FEMALE.npz
└── SMPLX_NEUTRAL.npz
```

### 4. Python Environment

```bash
# Create conda environment
conda create -n 3d-to-video python=3.11 -y
conda activate 3d-to-video

# Install dependencies
pip install torch torchvision  # or: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install smplx scipy trimesh numpy huggingface_hub

# Verify key packages
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import smplx; print('smplx OK')"
python -c "from huggingface_hub import snapshot_download; print('huggingface_hub OK')"
```

**Tested versions:** Python 3.11, PyTorch 2.11.0, smplx 0.1.28, scipy 1.17.1, numpy 1.26.4, huggingface_hub 1.10.2

### 5. VACE (for V2V, optional)

Only needed if you want to generate realistic videos (not just Blender renders).

```bash
# Clone VACE
git clone https://github.com/ali-vilab/VACE.git
cd VACE

# Follow VACE's setup instructions to download Wan2.1-VACE-14B model weights
# Requires ~22GB VRAM (RTX 3090/4090/5090)
```

---

## Quick Start: Demo Pipeline

### Step 1: Clone the repository

```bash
git clone https://github.com/kinam0252/3D-To-Video.git
cd 3D-To-Video
```

### Step 2: Download sample data

```bash
conda activate 3d-to-video
python download_data.py --data_dir ./data
```

This downloads ~15MB of sample data from [HuggingFace](https://huggingface.co/datasets/kinam0252/3D-To-Video-samples):

| Type | Sequence | Description |
|------|----------|-------------|
| OMOMO | `sub12_woodchair_005` | Person interacting with wooden chair (264 frames) |
| OMOMO | `sub14_suitcase_010` | Person handling suitcase (160 frames) |
| OMOMO | `sub12_tripod_010` | Person setting up tripod (161 frames) |
| HUMOTO | `lifting_and_putting_down_dining_chair-368` | Person lifting dining chair |
| HUMOTO | `eating_from_plastic_bowl_with_spoon-596` | Person eating with spoon |

Expected output:
```
data/
├── omomo/
│   ├── sub12_woodchair_005/    # human.npz + object.npz
│   ├── sub14_suitcase_010/
│   ├── sub12_tripod_010/
│   └── objects/                # woodchair.obj, suitcase.obj, tripod.obj
└── humoto/
    ├── lifting_and_putting_down_dining_chair-368/   # *.glb
    └── eating_from_plastic_bowl_with_spoon-596/     # *.glb
```

### Step 3: Run the demo (render only)

```bash
bash run_demo.sh \
  --blender /path/to/blender-5.1.0-linux-x64/blender \
  --smplx_dir /path/to/models/smplx \
  --skip_v2v
```

This will:
1. **Precompute SMPLX vertices** for each OMOMO sequence (torch + smplx)
2. **Render** all 5 sequences in Blender with orbit camera (EEVEE engine, ~0.1s/frame)
3. Output rendered frames + MP4 videos

**Expected time:** ~5-10 minutes total for all 5 sequences.

### Step 4: Check outputs

```bash
ls output/demo/renders/
# Should show 5 directories:
# sub12_woodchair_005_orbit_right/
# sub14_suitcase_010_orbit_right/
# sub12_tripod_010_orbit_right/
# lifting_and_putting_down_dining_chair-368_orbit_left/
# eating_from_plastic_bowl_with_spoon-596_orbit_right/

# Each contains frame_NNNN.png files and an .mp4 video
ls output/demo/renders/sub12_woodchair_005_orbit_right/*.mp4
```

### Step 5 (Optional): Run V2V generation

Requires VACE setup and ~22GB VRAM GPU.

```bash
bash run_demo.sh \
  --blender /path/to/blender \
  --smplx_dir /path/to/models/smplx \
  --vace_dir /path/to/VACE
```

**Expected time:** ~12 minutes per sequence (1 hour total for 5 sequences).

---

## Pipeline Details

### Pipeline A: OMOMO (InterAct)

```
human.npz + object.npz + object.obj
        ↓ precompute_smplx.py (torch + smplx)
    vertices.npz (SMPLX → mesh vertices, Y-up → Z-up)
        ↓ render_interact.py (Blender)
    frame_0000.png ... frame_NNNN.png (orbit camera, 30fps)
        ↓ 49-frame subsampling + depth extraction + VACE V2V
    realistic_video.mp4
```

**Key technical details:**
- **Coordinate conversion:** SMPLX uses Y-up, Blender uses Z-up → `(x, y, z) → (x, -z, y)`
- **Shape keys:** Each frame is stored as a Blender shape key for smooth mesh animation
- **Object transforms:** Euler angles + translations keyframed per frame
- **Orbit camera:** Camera orbits around the scene center over the animation duration

### Pipeline B: HUMOTO

```
animation.glb (with embedded skeleton + objects)
        ↓ render_humoto_full.py (Blender)
    frame_0000.png ... frame_NNNN.png (orbit camera)
        ↓ 49-frame subsampling + depth extraction + VACE V2V
    realistic_video.mp4
```

**Key technical details:**
- GLB contains pre-animated skeleton with integrated objects
- Render uses Cycles engine (higher quality than EEVEE, ~1s/frame)
- Camera auto-frames scene based on bounding box analysis

---

## Key Scripts

| Script | Description | Runs In |
|--------|-------------|---------|
| `download_data.py` | Download sample data from HuggingFace | Python |
| `precompute_smplx.py` | Compute SMPLX mesh vertices (Y-up → Z-up) | Python (torch) |
| `render_interact.py` | Render OMOMO sequences with orbit camera | Blender |
| `render_humoto_full.py` | Render HUMOTO GLB files with orbit camera | Blender |
| `run_demo.sh` | End-to-end demo pipeline orchestrator | Bash |

---

## Hardware Requirements

| Task | GPU | RAM | Time |
|------|-----|-----|------|
| Blender rendering (OMOMO, EEVEE) | Not required | 4GB+ | ~0.1s/frame |
| Blender rendering (HUMOTO, Cycles) | Optional (faster) | 8GB+ | ~1s/frame |
| V2V generation (VACE) | 22GB+ VRAM | 32GB+ | ~12 min/sequence |

---

## Troubleshooting

### "No module named smplx"
Make sure you activated the conda environment: `conda activate 3d-to-video`

### Blender crashes with "EEVEE" error
Ensure you're using Blender 5.1+. The engine name is `BLENDER_EEVEE` (not `BLENDER_EEVEE_NEXT`).

### V2V out of memory
VACE requires ~22GB VRAM. Use `--offload_model True --t5_cpu` (already set in `run_demo.sh`).

### Rendered frames look wrong (flipped/rotated)
The Y-up → Z-up conversion in `precompute_smplx.py` handles this. If adding new data, ensure `(x, y, z) → (x, -z, y)`.

---

## Acknowledgments

- [InterAct / OMOMO](https://github.com/jiashunwang/InterAct) — Human-object interaction motion capture dataset
- [HUMOTO](https://humoto.is.tue.mpg.de/) — Animated humanoid-object interaction dataset
- [VACE](https://github.com/ali-vilab/VACE) — Video-to-video diffusion model (Wan2.1-VACE-14B)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) — Expressive body model
- [Blender](https://www.blender.org/) — 3D rendering engine