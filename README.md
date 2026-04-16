# 3D-To-Video: Photorealistic Human-Object Interaction Rendering

Multi-view video rendering of realistic human-object interactions using SMPL-X + Blender.

## Goal
Generate ground-truth multi-view videos of humans interacting with objects,
rendered from actual 3D scenes (not AI video generation) for 3D consistency.

## Architecture
```
SMPL-X (pose/shape) → OBJ mesh → Blender Cycles (GPU OptiX) → PNG/Video
     ↑                    ↑              ↑
  GRAB motions      SMPL-X UVs     RTX 5090 GPU
  body params       procedural     1080x1920
                    skin shader    256 samples
```

## Current State (v5)
- ✅ Isolated Blender environment (no conflict with other projects)
- ✅ SMPL-X posed mesh generation via Python (smplx package)
- ✅ Procedural skin shader: SSS, micro pores, wrinkle bump, roughness variation
- ✅ Procedural eyes with iris/pupil/sclera
- ✅ GPU OptiX rendering (~50s for 1080x1920)
- ✅ Studio lighting (5-point setup)
- ⬜ Hair system (particle hair needs careful vertex group)
- ⬜ Clothing meshes
- ⬜ GRAB motion sequences
- ⬜ Multi-view camera orbits
- ⬜ Object interaction rendering

## Quick Start
```bash
# Generate posed mesh (requires conda env)
conda activate 3d-to-video
python scripts/generate_posed_mesh.py

# Render
./run_blender.sh --python scripts/render_ultra_realistic.py
```

## Directory Structure
```
3D-To-Video/
├── scripts/
│   ├── generate_posed_mesh.py    # SMPL-X → OBJ with pose/shape
│   └── render_ultra_realistic.py # Blender render script
├── assets/
│   ├── humans/
│   │   ├── posed_smplx_male.obj  # Generated posed mesh
│   │   └── smplx_models/ → symlink to SMPL-X models
│   ├── motions/grab/ → symlink to GRAB dataset
│   ├── objects/
│   └── textures/SMPLitex/        # UV texture research
├── blender_env/                   # Isolated Blender config
│   ├── config/, scripts/, data/
├── output/renders/
├── run_blender.sh                # Headless Blender launcher
└── launch_blender.sh             # GUI Blender launcher
```

## Environment
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **Blender**: 5.1.0 with Cycles + OptiX
- **Conda**: `3d-to-video` (Python 3.11, PyTorch, smplx)
- **Data**: SMPL-X models, GRAB sequences (symlinked)
