"""
Precompute SMPLX vertices for InterAct/OMOMO sequences.
Runs OUTSIDE Blender using torch + smplx.

Usage (conda env 3d-to-video):
    python precompute_smplx.py \
        --sequence sub6_whitechair_024 \
        --data_dir ~/Repos/3D-To-Video/assets/datasets/interact_data/InterAct/omomo \
        --smplx_dir ~/Desktop/DATA/EgoX/SMPLX/models \
        --output_dir ~/Repos/3D-To-Video/output/interact_precomputed
"""
import argparse, os, numpy as np

def main():
    parser = argparse.ArgumentParser(description="Precompute SMPLX vertices")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--data_dir", required=True,
                        help="OMOMO data root (contains sequences_canonical/ and objects/)")
    parser.add_argument("--smplx_dir", required=True,
                        help="Path to SMPLX model directory (contains SMPLX_*.npz)")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    import torch
    import smplx

    data_dir = os.path.expanduser(args.data_dir)
    smplx_dir = os.path.expanduser(args.smplx_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load human data
    human_path = os.path.join(data_dir, "sequences_canonical", args.sequence, "human.npz")
    human = np.load(human_path)

    poses = human["poses"]        # (N, 156)
    betas = human["betas"]        # (16,)
    trans = human["trans"]        # (N, 3)
    gender = str(human["gender"]) if "gender" in human else "neutral"
    # Clean gender string
    gender = gender.strip().lower()
    if gender not in ("male", "female", "neutral"):
        gender = "neutral"

    N = poses.shape[0]
    print(f"Sequence: {args.sequence}")
    print(f"Frames: {N}, Gender: {gender}, Betas: {betas.shape}")

    # Split poses
    global_orient = poses[:, 0:3]       # (N, 3)
    body_pose = poses[:, 3:66]          # (N, 63)
    left_hand_pose = poses[:, 66:111]   # (N, 45)
    right_hand_pose = poses[:, 111:156] # (N, 45)

    # Create SMPLX model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smplx.create(
        model_path=smplx_dir,
        model_type="smplx",
        gender=gender,
        use_pca=False,
        num_betas=16,
        ext="npz",
        batch_size=N,
    ).to(device)

    # Run forward pass
    with torch.no_grad():
        output = model(
            global_orient=torch.tensor(global_orient, dtype=torch.float32, device=device),
            body_pose=torch.tensor(body_pose, dtype=torch.float32, device=device),
            left_hand_pose=torch.tensor(left_hand_pose, dtype=torch.float32, device=device),
            right_hand_pose=torch.tensor(right_hand_pose, dtype=torch.float32, device=device),
            betas=torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0).expand(N, -1),
            transl=torch.tensor(trans, dtype=torch.float32, device=device),
        )
        vertices = output.vertices.cpu().numpy()  # (N, 10475, 3)

    faces = model.faces.astype(np.int32)  # (F, 3)

    # Convert Y-up (SMPLX) to Z-up (Blender): swap Y and Z axes
    # (x, y, z)_yup -> (x, -z, y)_zup  (rotate -90 deg around X)
    vertices_zup = vertices.copy()
    vertices_zup[:, :, 0] = vertices[:, :, 0]   # X stays
    vertices_zup[:, :, 1] = -vertices[:, :, 2]  # new Y = -old Z
    vertices_zup[:, :, 2] = vertices[:, :, 1]   # new Z = old Y (height)
    vertices = vertices_zup

    out_path = os.path.join(output_dir, f"{args.sequence}_vertices.npz")
    np.savez_compressed(out_path, vertices=vertices, faces=faces)
    print(f"Saved: {out_path}")
    print(f"  vertices: {vertices.shape}, faces: {faces.shape}")

if __name__ == "__main__":
    main()
