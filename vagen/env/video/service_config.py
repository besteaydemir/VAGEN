from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass

@dataclass
class RoomServiceConfig(BaseServiceConfig):
    """Configuration for RoomService with support for state reward functionality."""

    # State reward configuration - matches the pattern from SokobanServiceConfig
    use_state_reward: bool = False
    top_strings_m: int = 1000  # Maximum number of strings to track for repetition detection
    top_strings_k: int = 5     # Top-k strings for repetition penalty