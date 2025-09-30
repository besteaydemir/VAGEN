# envs/env_config.py
from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields
from typing import Optional

@dataclass
class RoomEnvConfig(BaseEnvConfig):
    env_name: str = "room_exploration"

    # Rendering
    render_mode: str = "vision"       # "vision" (RGB image) or "text"
    image_placeholder: str = "room_img"

    # Prompt formatting
    prompt_format: str = "default"
    max_actions_per_step: int = 1

    # Point cloud path
    point_cloud_path: Optional[str] = None   # Path to .ply/.pcd file

    # Camera intrinsics (defaults chosen to match env module constants)
    width: int = 640
    height: int = 480
    fx: float = 600.0
    fy: float = 600.0
    cx: Optional[float] = None  # if None, defaults to width/2
    cy: Optional[float] = None  # if None, defaults to height/2

    # Reward shaping
    use_state_reward: bool = False
    format_reward: float = 0.0

    # Image saving (headless machines)
    save_images: bool = False
    save_dir: Optional[str] = None
    image_prefix: str = "room"

    def config_id(self) -> str:
        """Generate a string ID for this config."""
        id_fields = [
            "render_mode",
            "max_actions_per_step",
            "prompt_format",
            "point_cloud_path"
        ]
        id_str = ",".join([
            f"{field.name}={getattr(self, field.name)}"
            for field in fields(self)
            if field.name in id_fields
        ])
        return f"RoomEnvConfig({id_str})"


if __name__ == "__main__":
    config = RoomEnvConfig(point_cloud_path="420673_laser_scan_cropped.ply")
    print(config.config_id())
