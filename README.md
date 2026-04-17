# 3D-To-Video

Generate 3D-consistent realistic videos of human-object interactions.

**Pipeline:** 3D Motion Capture → Blender Rendering → VACE V2V Diffusion

## Overview

This project builds a pipeline to:
1. Load human-object interaction data (OMOMO/HUMOTO datasets)
2. Render 3D scenes in Blender with orbit cameras
3. Convert rendered animations to photorealistic video using VACE V2V

### Supported Datasets

| Dataset | Format | Human | Objects | Sequences |
|---------|--------|-------|---------|-----------|
| **OMOMO** (InterAct) | SMPLX .npz + .obj | Separate SMPLX body mesh | Separate OBJ meshes, keyframed | 3 demo samples |
| **HUMOTO** | Animated .glb | Integrated in GLB | Integrated in GLB | 2 demo samples |

## Quick Start

### 1. Setup Environment

```bash
conda create -n 3d-to-video python=3.11
conda activate 3d-to-video
pip install torch smplx scipy trimesh huggingface_hub
```

### 2. Download Sample Data

```bash
python download_data.py --data_dir ./data
```

This downloads from [HuggingFace](https://huggingface.co/datasets/kinam0252/3D-To-Video-samples):
- 3 OMOMO sequences (woodchair, suitcase, tripod) + object meshes
- 2 HUMOTO sequences (dining chair, eating)

### 3. Prerequisites

- **Blender 5.1+**: [Download](https://www.blender.org/download/)
- **SMPLX Models**: [Register & Download](https://smpl-x.is.tue.mpg.de/) — place `SMPLX_*.npz` in `./models/smplx/`
- **VACE Model** (for V2V): [Wan2.1-VACE-14B](https://github.com/ali-vilab/VACE)

### 4. Run Demo

```bash
# Render only (no GPU needed for V2V)
bash run_demo.sh --skip_v2v

# Full pipeline (requires ~22GB VRAM)
bash run_demo.sh --blender /path/to/blender --vace_dir /path/to/VACE --smplx_dir /path/to/smplx/models
```

## Pipeline Details

### Pipeline A: OMOMO (InterAct)

```
human.npz + object.npz + object.obj
        ↓ precompute_smplx.py (torch + smplx)
    vertices.npz (SMPLX → mesh vertices, Y-up → Z-up)
        ↓ render_interact.py (Blender)
    frame_0000.png ... frame_NNNN.png (orbit camera, 30fps)
        ↓ depth extraction + VACE V2V
    realistic_video.mp4
```

- **Coordinate conversion**: SMPLX uses Y-up, Blender uses Z-up → `(x,y,z) → (x,-z,y)`
- **Shape keys**: Each frame gets a Blender shape key for smooth mesh animation
- **Object transforms**: Euler angles + translations keyframed per frame

### Pipeline B: HUMOTO

```
animation.glb
        ↓ render_humoto_full.py (Blender)
    frame_0000.png ... frame_NNNN.png (orbit camera)
        ↓ depth extraction + VACE V2V
    realistic_video.mp4
```

## Key Scripts

| Script | Description |
|--------|-------------|
| `download_data.py` | Download sample data from HuggingFace |
| `precompute_smplx.py` | Compute SMPLX vertices outside Blender |
| `render_interact.py` | Blender script for OMOMO sequences |
| `render_humoto_full.py` | Blender script for HUMOTO GLB files |
| `run_demo.sh` | End-to-end demo pipeline |

## Hardware Requirements

- **Rendering only**: Any CPU (Blender EEVEE, ~0.1s/frame)
- **V2V generation**: NVIDIA GPU with ≥22GB VRAM (RTX 3090/4090/5090)
- **Full demo**: ~12 min per V2V job, ~1 hour for all 5 sequences

## Acknowledgments

- [InterAct / OMOMO](https://github.com/jiashunwang/InterAct) for human-object interaction data
- [HUMOTO](https://humoto.is.tue.mpg.de/) for animated humanoid data
- [VACE](https://github.com/ali-vilab/VACE) for video-to-video diffusion
- [SMPLX](https://smpl-x.is.tue.mpg.de/) for body model
