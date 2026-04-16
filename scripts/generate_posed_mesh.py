"""Generate naturally-posed SMPL-X mesh v2 - corrected axes"""
import smplx, torch, numpy as np, os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = '/home/kinam/Desktop/DATA/EgoX/SMPLX/models'
NPZ_PATH = os.path.join(PROJECT_DIR, 'assets/humans/smplx_models/smplx/SMPLX_MALE.npz')
OUTPUT_OBJ = os.path.join(PROJECT_DIR, 'assets/humans/posed_smplx_male.obj')

def main():
    model = smplx.create(MODEL_PATH, model_type='smplx', gender='male',
                         use_pca=False, flat_hand_mean=True)

    # body_pose: 21 joints * 3 = 63
    # 0=L_Hip 1=R_Hip 2=Spine1 3=L_Knee 4=R_Knee 5=Spine2
    # 6=L_Ankle 7=R_Ankle 8=Spine3 9=L_Foot 10=R_Foot 11=Neck
    # 12=L_Collar 13=R_Collar 14=Head 15=L_Shoulder 16=L_Elbow 17=L_Wrist
    # 18=R_Shoulder 19=R_Elbow 20=R_Wrist
    bp = torch.zeros(1, 63, dtype=torch.float32)
    
    # Arms: CORRECTED signs - negative Z for left brings arm DOWN
    bp[0, 15*3+2] = -0.75   # L_Shoulder Z → arm down ~43°
    bp[0, 18*3+2] = 0.75    # R_Shoulder Z → arm down ~43°
    
    # Collar slight
    bp[0, 12*3+2] = -0.08
    bp[0, 13*3+2] = 0.08
    
    # Elbow slight bend
    bp[0, 16*3+1] = 0.2     # L_Elbow
    bp[0, 19*3+1] = -0.2    # R_Elbow
    
    # Natural leg stance
    bp[0, 0*3+2] = -0.03    # L_Hip splay
    bp[0, 1*3+2] = 0.03     # R_Hip splay
    bp[0, 3*3+0] = 0.03     # L_Knee bend
    bp[0, 4*3+0] = 0.03     # R_Knee bend

    # Shape: athletic/lean build (positive beta0 = thinner)
    betas = torch.zeros(1, 10, dtype=torch.float32)
    betas[0, 0] = 1.5    # thinner
    betas[0, 1] = -0.3   # slightly shorter proportions
    
    # Relaxed hands
    lhp = torch.zeros(1, 45, dtype=torch.float32)
    rhp = torch.zeros(1, 45, dtype=torch.float32)
    for i in range(15):
        lhp[0, i*3] = 0.2
        rhp[0, i*3] = 0.2

    out = model(betas=betas, body_pose=bp,
                left_hand_pose=lhp, right_hand_pose=rhp,
                expression=torch.zeros(1, 10))
    
    verts = out.vertices.detach().numpy()[0]
    faces = model.faces
    npz = np.load(NPZ_PATH, allow_pickle=True)
    vt = npz.get('vt', None)
    ft = npz.get('ft', None)
    
    height = verts[:,1].max() - verts[:,1].min()
    print(f'Verts: {verts.shape}, height={height:.3f}m (Y-up)')
    print(f'X: [{verts[:,0].min():.3f}, {verts[:,0].max():.3f}]')
    print(f'Y: [{verts[:,1].min():.3f}, {verts[:,1].max():.3f}]')
    
    with open(OUTPUT_OBJ, 'w') as f:
        f.write('# SMPL-X posed male - A-pose\n')
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        if vt is not None:
            for uv in vt:
                f.write(f'vt {uv[0]:.6f} {uv[1]:.6f}\n')
        for i, face in enumerate(faces):
            if ft is not None:
                uf = ft[i]
                f.write(f'f {face[0]+1}/{uf[0]+1} {face[1]+1}/{uf[1]+1} {face[2]+1}/{uf[2]+1}\n')
            else:
                f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
    print(f'Saved: {OUTPUT_OBJ}')

if __name__ == '__main__':
    main()
