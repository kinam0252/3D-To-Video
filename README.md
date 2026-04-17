# 3D-To-Video

Generate **3D-consistent realistic videos** of human-object interactions.

**Pipeline:** 3D Motion Capture → Blender Rendering → VACE V2V Diffusion

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

## Quick Start (One-Command Setup)

> Requires: **Linux**, **conda**, **FFmpeg** (`sudo apt install ffmpeg`), **wget**

```bash
# 1. Clone
git clone https://github.com/kinam0252/3D-To-Video.git
cd 3D-To-Video

# 2. Create Python environment
conda create -n 3d-to-video python=3.11 -y
conda activate 3d-to-video

# 3. Run setup (downloads Blender + data + SMPLX + HDRI)
bash setup.sh

# 4. Run demo
bash run_demo.sh --skip_v2v
```

That's it! The setup script handles everything:
- ✅ Downloads Blender 5.1.0 (~300MB) into the project directory
- ✅ Installs Python dependencies from `requirements.txt`
- ✅ Downloads sample data from HuggingFace (~120MB):
  - 3 OMOMO motion sequences + object meshes
  - 2 HUMOTO animation files (.glb)
  - SMPLX body model (SMPLX_MALE.npz)
  - HDRI environment map

### Check Outputs

```bash
ls output/demo/renders/
# sub12_woodchair_005_orbit_right/     (264 frames + .mp4)
# sub14_suitcase_010_orbit_right/      (160 frames + .mp4)
# sub12_tripod_010_orbit_right/        (161 frames + .mp4)
# lifting_and_putting_down_dining_chair-368_orbit_left/  (49 frames + .mp4)
# eating_from_plastic_bowl_with_spoon-596_orbit_right/   (49 frames + .mp4)
```

**Expected time:** ~5-10 minutes for all 5 sequences on any modern CPU.

---

## Manual Setup (Step by Step)

If `setup.sh` doesn't work or you want more control:

### 1. System Dependencies

```bash
# FFmpeg (for video encoding)
sudo apt install ffmpeg wget

# Verify
ffmpeg -version
```

### 2. Blender 5.1+

```bash
wget https://download.blender.org/release/Blender5.1/blender-5.1.0-linux-x64.tar.xz
tar -xf blender-5.1.0-linux-x64.tar.xz
./blender-5.1.0-linux-x64/blender --version  # Should print: Blender 5.1.0
```

> Blender runs headlessly (`--background`). No display server needed.

### 3. Python Environment

```bash
conda create -n 3d-to-video python=3.11 -y
conda activate 3d-to-video
pip install -r requirements.txt
# Or manually: pip install torch smplx scipy trimesh numpy huggingface_hub
```

### 4. Download Data

```bash
python download_data.py --data_dir ./data
```

This downloads from [HuggingFace](https://huggingface.co/datasets/kinam0252/3D-To-Video-samples) and automatically sets up:

```
data/                    # Downloaded from HuggingFace
├── omomo/
│   ├── sequences_canonical/
│   │   ├── sub12_woodchair_005/   # human.npz + object.npz
│   │   ├── sub14_suitcase_010/
│   │   └── sub12_tripod_010/
│   └── objects/                   # woodchair.obj, suitcase.obj, tripod.obj
└── humoto/
    ├── lifting_and_putting_down_dining_chair-368/  # *.glb
    └── eating_from_plastic_bowl_with_spoon-596/    # *.glb

models/smplx/            # Auto-copied from data/smplx/
└── SMPLX_MALE.npz

assets/hdri/             # Auto-copied from data/hdri/
└── pedestrian_overpass_1k.exr
```

### 5. Run Demo

```bash
bash run_demo.sh \
  --blender ./blender-5.1.0-linux-x64/blender \
  --smplx_dir ./models \
  --skip_v2v
```

### 6. (Optional) V2V Generation

Requires [VACE](https://github.com/ali-vilab/VACE) with Wan2.1-VACE-14B model (~22GB VRAM).

```bash
bash run_demo.sh \
  --blender ./blender-5.1.0-linux-x64/blender \
  --smplx_dir ./models \
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
        ↓ render_interact.py (Blender EEVEE)
    frame_0000.png ... frame_NNNN.png (orbit camera, 30fps)
        ↓ 49-frame subsampling + depth extraction + VACE V2V
    realistic_video.mp4
```

**Technical details:**
- **Coordinate conversion:** SMPLX uses Y-up, Blender uses Z-up → `(x, y, z) → (x, -z, y)`
- **Shape keys:** Each frame is stored as a Blender shape key for smooth mesh animation
- **Object transforms:** Euler angles + translations keyframed per frame
- **Orbit camera:** Camera orbits around the scene center over the animation

### Pipeline B: HUMOTO

```
animation.glb (with embedded skeleton + objects)
        ↓ render_humoto_full.py (Blender Cycles)
    frame_0000.png ... frame_NNNN.png (orbit camera)
        ↓ 49-frame subsampling + depth extraction + VACE V2V
    realistic_video.mp4
```

**Technical details:**
- GLB contains pre-animated skeleton with integrated objects
- Render uses Cycles engine (~1s/frame)
- Camera auto-frames scene based on bounding box analysis

---

## Project Structure

| File | Description |
|------|-------------|
| `setup.sh` | **One-command setup** (Blender + deps + data) |
| `download_data.py` | Download sample data + SMPLX + HDRI from HuggingFace |
| `run_demo.sh` | End-to-end demo pipeline (render + optional V2V) |
| `requirements.txt` | Python dependencies |
| `precompute_smplx.py` | Compute SMPLX mesh vertices (Y-up → Z-up) |
| `render_interact.py` | Blender script for OMOMO sequences |
| `render_humoto_full.py` | Blender script for HUMOTO GLB files |

---

## Hardware Requirements

| Task | GPU | RAM | Time |
|------|-----|-----|------|
| Blender rendering (OMOMO, EEVEE) | Not required | 4GB+ | ~0.1s/frame |
| Blender rendering (HUMOTO, Cycles) | Optional (faster) | 8GB+ | ~1s/frame |
| V2V generation (VACE) | ≥22GB VRAM | 32GB+ | ~12 min/sequence |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named smplx` | `conda activate 3d-to-video && pip install -r requirements.txt` |
| Blender EEVEE error | Use Blender 5.1+ (engine name is `BLENDER_EEVEE`, not `BLENDER_EEVEE_NEXT`) |
| V2V out of memory | Requires ≥22GB VRAM. `--offload_model True --t5_cpu` already set in `run_demo.sh` |
| Flipped/rotated renders | Y-up → Z-up handled by `precompute_smplx.py`. Verify `(x,y,z) → (x,-z,y)` |
| `setup.sh` Blender download fails | Manually download from [blender.org](https://www.blender.org/download/) and use `--blender /path/to/blender` |

---

## Acknowledgments

- [InterAct / OMOMO](https://github.com/jiashunwang/InterAct) — Human-object interaction data
- [HUMOTO](https://humoto.is.tue.mpg.de/) — Animated humanoid-object data
- [VACE](https://github.com/ali-vilab/VACE) — Video-to-video diffusion (Wan2.1-VACE-14B)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) — Expressive body model
- [Blender](https://www.blender.org/) — 3D rendering engine