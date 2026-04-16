"""
Batch Render Config Generator & Runner
=======================================
Generates combinatorial configs from avatars × motions × objects × scenes × cameras,
then renders them via Blender headless.

Usage:
    python3 batch_render.py --generate          # Generate configs only
    python3 batch_render.py --render            # Render all pending configs
    python3 batch_render.py --generate --render # Both
    python3 batch_render.py --render --max 5    # Render first 5
    python3 batch_render.py --status            # Show render status
"""
import json, os, sys, itertools, random, subprocess, argparse, time
from pathlib import Path

PROJECT_DIR = Path(os.path.expanduser("~/Repos/3D-To-Video"))
BATCH_DIR = PROJECT_DIR / "batch_configs"
BLENDER_BIN = os.path.expanduser("~/Downloads/blender-5.1.0-linux-x64/blender")
BLENDER_ENV = {
    "BLENDER_USER_CONFIG": str(PROJECT_DIR / "blender_env/config"),
    "BLENDER_USER_SCRIPTS": str(PROJECT_DIR / "blender_env/scripts"),
    "BLENDER_USER_DATAFILES": str(PROJECT_DIR / "blender_env/data"),
}

# ========== ASSET POOLS ==========
AVATARS = [
    {"id": "female_0", "file": "female_0.glb"},
    {"id": "female_1", "file": "female_1.glb"},
    {"id": "male_0", "file": "male_0.glb"},
    {"id": "male_1", "file": "male_1.glb"},
]

OBJECTS = [
    {
        "id": "backpack_black",
        "type": "backpack",
        "path": "assets/sketchfab/black_backpack/scene.gltf",
        "bone": "spine3",
        "size": 0.45,
        "rotation": [-90, 180, 0],
        "offset": [0.0, -0.20, -0.10],
        "hide_parts": ["daizi", "kouzi"],
    },
    {
        "id": "backpack_color",
        "type": "backpack",
        "path": "assets/sketchfab/backpack/scene.gltf",
        "bone": "spine3",
        "size": 0.45,
        "rotation": [-90, 180, 0],
        "offset": [0.0, -0.20, -0.10],
        "hide_parts": ["daizi", "kouzi"],
    },
    {
        "id": "cap",
        "type": "hat",
        "path": "assets/sketchfab/baseball_cap/scene.gltf",
        "bone": "head",
        "size": 0.25,
        "rotation": [0, 0, 0],
        "offset": [0.0, 0.1, 0.0],
        "hide_parts": [],
    },
    {
        "id": "none",
        "type": "none",
        "path": "",
        "bone": "",
        "size": 0,
        "rotation": [0, 0, 0],
        "offset": [0, 0, 0],
        "hide_parts": [],
    },
]

MOTIONS = [
    {"id": "phone_call", "action": "phone_call_1", "start": 600, "step": 6},
    {"id": "apple_eat", "action": "apple_eat_1", "start": 300, "step": 6},
    {"id": "camera_take", "action": "camera_takepicture_1", "start": 300, "step": 6},
    {"id": "cup_drink", "action": "cup_drink_1", "start": 300, "step": 6},
    {"id": "banana_peel", "action": "banana_peel_1", "start": 300, "step": 6},
]

SCENES = [
    {"id": "urban", "hdri": "urban_street.exr", "strength": 2.2, "preset": "urban", "fog": True},
    {"id": "park", "hdri": "kloofendal_48d_partly_cloudy_4k.exr", "strength": 3.0, "preset": "hdri_only", "fog": False},
    {"id": "overpass", "hdri": "pedestrian_overpass_4k.exr", "strength": 2.5, "preset": "hdri_only", "fog": False},
    {"id": "studio", "hdri": "studio_small_09_1k.exr", "strength": 2.0, "preset": "hdri_only", "fog": False},
    {"id": "autumn", "hdri": "autumn_park_1k.exr", "strength": 2.5, "preset": "hdri_only", "fog": False},
]

CAMERAS = [
    {"id": "orbit_behind", "start": 90, "sweep": 60, "direction": "left", "radius": 2.8, "height": 1.25, "lens": 65},
    {"id": "orbit_front", "start": 270, "sweep": 60, "direction": "right", "radius": 2.8, "height": 1.25, "lens": 65},
    {"id": "orbit_wide", "start": 90, "sweep": 120, "direction": "left", "radius": 3.5, "height": 1.3, "lens": 50},
    {"id": "orbit_close", "start": 90, "sweep": 45, "direction": "left", "radius": 2.0, "height": 1.2, "lens": 85},
]


def generate_config(avatar, obj, motion, scene, camera):
    """Generate a single render config dict."""
    name = f"{avatar['id']}_{motion['id']}_{obj['id']}_{scene['id']}_{camera['id']}"

    cfg = {
        "avatar": avatar["file"],
        "motion_subject": "s1",
        "motion_action": motion["action"],
        "motion_start_frame": motion["start"],
        "motion_step": motion["step"],
        "object_type": obj["type"],
        "object_path": obj["path"],
        "object_bone": obj["bone"],
        "object_size": obj["size"],
        "object_rotation": obj["rotation"],
        "object_offset": obj["offset"],
        "object_hide_parts": obj.get("hide_parts", []),
        "hdri": scene["hdri"],
        "hdri_strength": scene["strength"],
        "scene_preset": scene["preset"],
        "use_fog": scene["fog"],
        "orbit_start_deg": camera["start"],
        "orbit_sweep_deg": camera["sweep"],
        "orbit_direction": camera["direction"],
        "cam_radius": camera["radius"],
        "cam_height": camera["height"],
        "cam_lens": camera["lens"],
        "cam_dof_fstop": 4.0,
        "num_frames": 81,
        "fps": 24,
        "samples": 64,
        "resolution": 640,
        "use_denoising": True,
        "output_name": name,
    }
    return name, cfg


def generate_all(mode="full", max_combos=None, seed=42):
    """
    Generate batch configs.
    mode:
      - "full": all combinations
      - "sample": random subset
      - "diverse": one per avatar with varied combos
    """
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "full":
        combos = list(itertools.product(AVATARS, OBJECTS, MOTIONS, SCENES, CAMERAS))
    elif mode == "diverse":
        random.seed(seed)
        combos = []
        for avatar in AVATARS:
            for _ in range(5):
                combos.append((
                    avatar,
                    random.choice(OBJECTS),
                    random.choice(MOTIONS),
                    random.choice(SCENES),
                    random.choice(CAMERAS),
                ))
    else:
        combos = list(itertools.product(AVATARS, OBJECTS, MOTIONS, SCENES, CAMERAS))
        random.seed(seed)
        random.shuffle(combos)

    if max_combos:
        combos = combos[:max_combos]

    generated = []
    for avatar, obj, motion, scene, camera in combos:
        name, cfg = generate_config(avatar, obj, motion, scene, camera)
        cfg_path = BATCH_DIR / f"{name}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        generated.append(name)

    print(f"Generated {len(generated)} configs in {BATCH_DIR}/")
    for g in generated:
        print(f"  - {g}")
    return generated


def get_render_status():
    """Check which configs have been rendered."""
    if not BATCH_DIR.exists():
        return {"total": 0, "done": 0, "pending": 0}, [], []

    configs = sorted(BATCH_DIR.glob("*.json"))
    done = []
    pending = []

    for cfg_path in configs:
        name = cfg_path.stem
        render_dir = PROJECT_DIR / "output" / "renders" / name
        video_file = render_dir / "video.mp4"
        if video_file.exists() or (render_dir.exists() and any(render_dir.glob("*.png"))):
            done.append(name)
        else:
            pending.append(name)

    return {"total": len(configs), "done": len(done), "pending": len(pending)}, done, pending


def render_one(config_name):
    """Render a single config via Blender headless."""
    cfg_path = BATCH_DIR / f"{config_name}.json"
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return False

    env = os.environ.copy()
    env.update(BLENDER_ENV)

    cmd = [
        BLENDER_BIN, "--background", "--python",
        str(PROJECT_DIR / "render_pipeline.py"),
        "--", "--config", str(cfg_path)
    ]

    print("\n" + "=" * 60)
    print(f"Rendering: {config_name}")
    print("=" * 60)

    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  Done in {elapsed:.0f}s")
        return True
    else:
        print(f"  Failed ({elapsed:.0f}s)")
        err = result.stderr[-500:] if result.stderr else "No stderr"
        print(f"  {err}")
        return False


def render_batch(max_renders=None):
    """Render all pending configs."""
    status, done, pending = get_render_status()
    print(f"Status: {status['done']}/{status['total']} done, {status['pending']} pending")

    if not pending:
        print("Nothing to render!")
        return

    to_render = pending[:max_renders] if max_renders else pending
    print(f"Will render {len(to_render)} configs\n")

    success, fail = 0, 0
    for name in to_render:
        if render_one(name):
            success += 1
        else:
            fail += 1

    print("\n" + "=" * 60)
    print(f"Batch complete: {success} success, {fail} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate configs")
    parser.add_argument("--render", action="store_true", help="Render pending configs")
    parser.add_argument("--status", action="store_true", help="Show render status")
    parser.add_argument("--mode", default="diverse", choices=["full", "sample", "diverse"])
    parser.add_argument("--max", type=int, default=None, help="Max configs to generate/render")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.status:
        status, done, pending = get_render_status()
        print(f"Total:   {status['total']}")
        print(f"Done:    {status['done']}")
        print(f"Pending: {status['pending']}")
        if pending:
            print("\nPending configs:")
            for p in pending[:10]:
                print(f"  - {p}")
            if len(pending) > 10:
                print(f"  ... and {len(pending) - 10} more")
        sys.exit(0)

    if args.generate:
        generate_all(mode=args.mode, max_combos=args.max, seed=args.seed)

    if args.render:
        render_batch(max_renders=args.max)

    if not args.generate and not args.render and not args.status:
        parser.print_help()
