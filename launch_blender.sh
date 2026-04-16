#!/bin/bash
# 3D-To-Video 프로젝트 전용 Blender 런처
# 글로벌 ~/.config/blender/ 와 완전 격리됨

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLENDER_BIN=~/Downloads/blender-5.1.0-linux-x64/blender

# Blender 환경변수로 설정/스크립트/데이터 경로를 프로젝트 내부로 격리
export BLENDER_USER_CONFIG="$PROJECT_DIR/blender_env/config"
export BLENDER_USER_SCRIPTS="$PROJECT_DIR/blender_env/scripts"
export BLENDER_USER_DATAFILES="$PROJECT_DIR/blender_env/data"

# Conda 환경 활성화 (Blender 외부 스크립트용)
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate 3d-to-video

echo "=== 3D-To-Video Blender Environment ==="
echo "  Config:  $BLENDER_USER_CONFIG"
echo "  Scripts: $BLENDER_USER_SCRIPTS"
echo "  Data:    $BLENDER_USER_DATAFILES"
echo "  Conda:   3d-to-video ($(python --version))"
echo "========================================="

exec "$BLENDER_BIN" "$@"
