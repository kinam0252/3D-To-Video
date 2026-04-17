"""
Download sample data from HuggingFace for the 3D-To-Video pipeline.

Downloads:
  - OMOMO sequences (human.npz + object.npz + object meshes)
  - HUMOTO sequences (.glb animation files)
  - SMPLX body model (SMPLX_MALE.npz)
  - HDRI environment map (pedestrian_overpass_1k.exr)

Usage:
    python download_data.py [--data_dir ./data]
"""
import argparse, os, shutil

def main():
    parser = argparse.ArgumentParser(description="Download 3D-To-Video sample data")
    parser.add_argument("--data_dir", default="./data", help="Directory to save data")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    os.makedirs(data_dir, exist_ok=True)

    print("Downloading sample data from HuggingFace...")
    snapshot_download(
        repo_id="kinam0252/3D-To-Video-samples",
        repo_type="dataset",
        local_dir=data_dir,
    )

    # Set up SMPLX models in project-local directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    smplx_src = os.path.join(data_dir, "smplx")
    smplx_dst = os.path.join(script_dir, "models", "smplx")
    if os.path.isdir(smplx_src) and not os.path.isdir(smplx_dst):
        os.makedirs(os.path.dirname(smplx_dst), exist_ok=True)
        shutil.copytree(smplx_src, smplx_dst)
        print(f"  SMPLX models copied to: {smplx_dst}")
    elif os.path.isdir(smplx_dst):
        print(f"  SMPLX models already at: {smplx_dst}")

    # Set up HDRI in project-local directory
    hdri_src = os.path.join(data_dir, "hdri")
    hdri_dst = os.path.join(script_dir, "assets", "hdri")
    if os.path.isdir(hdri_src) and not os.path.isdir(hdri_dst):
        os.makedirs(os.path.dirname(hdri_dst), exist_ok=True)
        shutil.copytree(hdri_src, hdri_dst)
        print(f"  HDRI files copied to: {hdri_dst}")
    elif os.path.isdir(hdri_dst):
        print(f"  HDRI files already at: {hdri_dst}")

    print(f"\nData downloaded to: {data_dir}")

    omomo_dir = os.path.join(data_dir, "omomo", "sequences_canonical")
    humoto_dir = os.path.join(data_dir, "humoto")
    omomo_seqs = [d for d in os.listdir(omomo_dir) if os.path.isdir(os.path.join(omomo_dir, d))] if os.path.isdir(omomo_dir) else []
    humoto_seqs = [d for d in os.listdir(humoto_dir) if os.path.isdir(os.path.join(humoto_dir, d))] if os.path.isdir(humoto_dir) else []
    smplx_files = os.listdir(smplx_dst) if os.path.isdir(smplx_dst) else []
    hdri_files = os.listdir(hdri_dst) if os.path.isdir(hdri_dst) else []

    print(f"  OMOMO sequences: {len(omomo_seqs)} - {omomo_seqs}")
    print(f"  HUMOTO sequences: {len(humoto_seqs)} - {humoto_seqs}")
    print(f"  SMPLX models: {smplx_files}")
    print(f"  HDRI files: {hdri_files}")
    print("\n✅ All data ready. Run: bash run_demo.sh --skip_v2v")

if __name__ == "__main__":
    main()
