#!/bin/bash
# Headless Blender runner (렌더링, 스크립트 실행용)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLENDER_BIN=~/Downloads/blender-5.1.0-linux-x64/blender

export BLENDER_USER_CONFIG="$PROJECT_DIR/blender_env/config"
export BLENDER_USER_SCRIPTS="$PROJECT_DIR/blender_env/scripts"
export BLENDER_USER_DATAFILES="$PROJECT_DIR/blender_env/data"

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate 3d-to-video

exec "$BLENDER_BIN" --background "$@"
