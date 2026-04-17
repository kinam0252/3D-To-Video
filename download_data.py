"""
Download sample data from HuggingFace for the 3D-To-Video pipeline.

Usage:
    python download_data.py [--data_dir ./data]
"""
import argparse, os

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

    data_dir = os.path.expanduser(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    print("Downloading sample data from HuggingFace...")
    snapshot_download(
        repo_id="kinam0252/3D-To-Video-samples",
        repo_type="dataset",
        local_dir=data_dir,
    )
    print(f"\nData downloaded to: {data_dir}")

    omomo_dir = os.path.join(data_dir, "omomo", "sequences_canonical")
    humoto_dir = os.path.join(data_dir, "humoto")
    omomo_seqs = os.listdir(omomo_dir) if os.path.isdir(omomo_dir) else []
    humoto_seqs = os.listdir(humoto_dir) if os.path.isdir(humoto_dir) else []
    print(f"  OMOMO sequences: {len(omomo_seqs)} - {omomo_seqs}")
    print(f"  HUMOTO sequences: {len(humoto_seqs)} - {humoto_seqs}")

    print("\nNOTE: SMPLX model files are required for OMOMO rendering.")
    print("  Download from: https://smpl-x.is.tue.mpg.de/")
    print("  Place SMPLX_*.npz files in: ./models/smplx/")

if __name__ == "__main__":
    main()
