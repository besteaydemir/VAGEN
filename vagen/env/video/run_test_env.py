"""Simple runner for RoomEnv to test reset/step and image saving on headless nodes.

Usage (example):
    python run_test_env.py --pc path/to/pointcloud.ply --outdir ./imgs --steps 5

This script:
 - creates a RoomEnvConfig with image saving enabled
 - resets the env (deterministic with seed)
 - steps with a few homogeneous matrices centered near the point-cloud centroid
 - the env will save images to the configured directory
"""
import argparse
import os
import numpy as np
from .env_config import RoomEnvConfig
from .env import RoomEnv


def make_transform(translation=np.zeros(3), yaw=0.0):
    """Create a simple camera extrinsic: rotation around Z (yaw) + translation."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rz
    T[:3, 3] = translation
    return T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc", type=str, default="420673_laser_scan_cropped.ply", help="point cloud path (.ply/.pcd)")
    parser.add_argument("--outdir", type=str, default="./env_images", help="directory to save images")
    parser.add_argument("--steps", type=int, default=5, help="number of steps to take")
    parser.add_argument("--seed", type=int, default=42, help="reset seed for deterministic pose")
    args = parser.parse_args()

    cfg = RoomEnvConfig(point_cloud_path=args.pc, save_images=True, save_dir=args.outdir)
    env = RoomEnv(cfg)

    obs, _ = env.reset(seed=args.seed)
    print("Reset observation string:\n", obs["obs_str"])

    # create transforms around the centroid
    centroid = env.pc_centroid
    translations = [centroid + np.array([0.0, 0.0, 0.0])]
    for i in range(1, args.steps):
        # small lateral moves and small yaws
        t = centroid + np.array([0.1 * (i % 3 - 1), 0.05 * i, 0.0])
        yaw = (i * 0.2)
        translations.append((t, yaw))

    # step through and save images
    for i in range(args.steps):
        if i == 0:
            # already reset to initial pose, but call step to render again
            trans = translations[0]
            if isinstance(trans, tuple):
                T = make_transform(translation=trans[0], yaw=trans[1])
            else:
                T = make_transform(translation=trans)
        else:
            trans = translations[i]
            T = make_transform(translation=trans[0], yaw=trans[1])

        obs, reward, done, info = env.step(T)
        img_list = obs["multi_modal_data"][cfg.image_placeholder]
        print(f"Step {i}: reward={reward}, saved image index={env._img_counter - 1}")

    print("Images saved to:", os.path.abspath(args.outdir))


if __name__ == '__main__':
    main()
