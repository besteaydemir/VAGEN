# envs/room_env.py
from vagen.env.base.base_env import BaseEnv
import numpy as np

import os
os.environ["OPEN3D_CPU_RENDERING"] = "true"

import open3d as o3d
from typing import Dict
from .env_config import RoomEnvConfig
from .prompt import system_prompt, init_observation_template, action_template, format_prompt
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from PIL import Image

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FX = 600.0
DEFAULT_FY = 600.0
DEFAULT_CX = DEFAULT_WIDTH / 2
DEFAULT_CY = DEFAULT_HEIGHT / 2

class RoomEnv(BaseEnv):
    """
    Room Exploration Environment:
    Action = 4x4 homogeneous camera extrinsic matrix.
    Observation = rendered RGB image from the point cloud using CPU-only Open3D.
    """

    def __init__(self, config: RoomEnvConfig):
        super().__init__()
        self.config = config

        # Load or generate point cloud
        if self.config.point_cloud_path:
            self.point_cloud = o3d.io.read_point_cloud(self.config.point_cloud_path)
        else:
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector(np.random.rand(1000, 3) - 0.5)
            self.point_cloud.paint_uniform_color([0.1, 0.7, 0.9])

        # Compute centroid for initialization
        pts = np.asarray(self.point_cloud.points)
        self.pc_centroid = pts.mean(axis=0) if pts.size > 0 else np.array([0.0,0.0,0.0], dtype=np.float64)

        # Rendering parameters
        self.width = getattr(self.config, 'width', DEFAULT_WIDTH)
        self.height = getattr(self.config, 'height', DEFAULT_HEIGHT)
        fx = getattr(self.config, 'fx', DEFAULT_FX)
        fy = getattr(self.config, 'fy', DEFAULT_FY)
        cx = self.config.cx if (getattr(self.config, 'cx', None) is not None) else self.width / 2.0
        cy = self.config.cy if (getattr(self.config, 'cy', None) is not None) else self.height / 2.0
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, cx, cy)
        
        # Material for point cloud
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.0
        self.agent_pose = None
        self.total_reward = 0
        self.reward = 0
        #self.format_prompt_func = format_prompt[self.config.prompt_format]
        self._img_counter = 0

        # Image saving folder
        if getattr(self.config, 'save_images', False):
            if not getattr(self.config, 'save_dir', None):
                self.config.save_dir = os.path.join('.', 'env_images')
            os.makedirs(self.config.save_dir, exist_ok=True)

    def reset(self, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        jitter = rng.uniform(-0.2, 0.2, size=(3,))
        t = self.pc_centroid + jitter
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = t
        self.agent_pose = pose

        self.total_reward = 0
        self.reward = 0
        return self._render(init_obs=True), {}

    def step(self, action_matrix: np.ndarray):
        if action_matrix.shape != (4, 4):
            raise ValueError("Action must be a 4x4 homogeneous matrix")
        self.agent_pose = action_matrix
        obs = self._render(init_obs=False)
        self.reward = 0.1
        self.total_reward += self.reward
        done = False
        info = {}
        return obs, self.reward, done, info

    def _render(self, init_obs=False) -> Dict:
        extr = self.agent_pose.astype(np.float64)
        obs_text = f"Agent pose:\n{extr}"

        # ---- Create a temporary renderer ----
        renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        renderer.scene.set_background([1, 1, 1, 1])  # white background

        # Material settings
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = getattr(self.config, 'point_size', 2.0)

        # Add point cloud
        renderer.scene.add_geometry("pcd", self.point_cloud, mat)

        # Camera setup
        eye = extr[:3, 3]
        center = self.pc_centroid
        up = np.array([0, 0, 1], dtype=np.float64)
        renderer.setup_camera(60.0, center, eye, up)

        # Render to image
        img = renderer.render_to_image()
        img_np = np.asarray(img)
        img_pil = convert_numpy_to_PIL(img_np)

        # Cleanup
        renderer.scene.clear_geometry()
        del renderer

        # Optionally save images
        if getattr(self.config, 'save_images', False):
            fname = f"{getattr(self.config, 'image_prefix','room')}_{self._img_counter:04d}.png"
            path = os.path.join(self.config.save_dir, fname)
            img_pil.save(path)
            self._img_counter += 1

        # Build observation string
        if init_obs:
            obs_str = init_observation_template(observation=obs_text) + "\n" # + self.format_prompt_func()
        else:
            obs_str = action_template(valid_action=["ExtrinsicMatrix"], observation=obs_text) + "\n" # + self.format_prompt_func()

        return {"obs_str": obs_str, "multi_modal_data": {self.config.image_placeholder: [img_pil]}}


    def system_prompt(self):
        return system_prompt() + "\n"  # self.format_prompt_func()

    def get_env_state(self):
        return {"agent_pose": self.agent_pose.copy()}

    def close(self):
        pass  # nothing to clean up, CPU rendering
