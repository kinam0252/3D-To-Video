#!/bin/bash
set -e

# ============================================================
# 3D-To-Video: One-command setup
# ============================================================
# This script sets up everything needed to run the demo:
#   1. Downloads & extracts Blender 5.1 (if not found)
#   2. Installs Python dependencies
#   3. Downloads sample data + SMPLX models + HDRI from HuggingFace
#
# Usage:
#   bash setup.sh [--blender /path/to/existing/blender]
#
# After setup, run:
#   bash run_demo.sh --skip_v2v
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BLENDER_VERSION="5.1.0"
BLENDER_URL="https://download.blender.org/release/Blender5.1/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_DIR="$SCRIPT_DIR/blender-${BLENDER_VERSION}-linux-x64"
BLENDER_BIN="$BLENDER_DIR/blender"
CUSTOM_BLENDER=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --blender) CUSTOM_BLENDER="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "============================================"
echo "  3D-To-Video Setup"
echo "============================================"
echo ""

# ---- Step 1: Blender ----
if [ -n "$CUSTOM_BLENDER" ]; then
  BLENDER_BIN="$CUSTOM_BLENDER"
  echo "[1/3] Using custom Blender: $BLENDER_BIN"
elif [ -x "$BLENDER_BIN" ]; then
  echo "[1/3] Blender already installed: $BLENDER_BIN"
else
  echo "[1/3] Downloading Blender ${BLENDER_VERSION}..."
  if ! command -v wget &> /dev/null; then
    echo "ERROR: wget not found. Install with: sudo apt install wget"
    exit 1
  fi
  wget -q --show-progress -O /tmp/blender.tar.xz "$BLENDER_URL"
  echo "  Extracting..."
  tar -xf /tmp/blender.tar.xz -C "$SCRIPT_DIR"
  rm /tmp/blender.tar.xz
  echo "  Blender installed: $BLENDER_BIN"
fi

# Verify Blender
if [ ! -x "$BLENDER_BIN" ]; then
  echo "ERROR: Blender not found at $BLENDER_BIN"
  exit 1
fi
echo "  $($BLENDER_BIN --version 2>/dev/null | head -1)"

# ---- Step 2: Python dependencies ----
echo ""
echo "[2/3] Installing Python dependencies..."
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
  echo "  Dependencies installed."
else
  echo "  WARNING: requirements.txt not found, installing manually..."
  pip install torch smplx scipy trimesh numpy huggingface_hub --quiet
fi

# Verify key packages
python -c "import torch, smplx, scipy, trimesh, numpy, huggingface_hub; print('  All packages OK')"

# ---- Step 3: Download data ----
echo ""
echo "[3/3] Downloading sample data (OMOMO + HUMOTO + SMPLX + HDRI)..."
python "$SCRIPT_DIR/download_data.py" --data_dir "$SCRIPT_DIR/data"

# ---- Done ----
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Blender:  $BLENDER_BIN"
echo "Data:     $SCRIPT_DIR/data"
echo "SMPLX:    $SCRIPT_DIR/models/smplx"
echo "HDRI:     $SCRIPT_DIR/assets/hdri"
echo ""
echo "To run the demo (render only):"
echo "  bash run_demo.sh --blender $BLENDER_BIN --smplx_dir $SCRIPT_DIR/models --skip_v2v"
echo ""
echo "Or simply (uses local Blender):"
echo "  bash run_demo.sh --skip_v2v"
