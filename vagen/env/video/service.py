from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from vagen.env.base.base_service import BaseService
from vagen.env.base.base_service_config import BaseServiceConfig
from vagen.server.serial import serialize_observation

from .env import RoomEnv
from .env_config import RoomEnvConfig
from vagen.env.utils.state_reward_text_utils import service_state_reward_wrapper_v3 as service_state_reward_wrapper
from vagen.env.utils.top_string_tracker import TopKStringTracker

class RoomService(BaseService):
    """
    Service class for Room environments.
    Implements batch operations with parallel processing for efficiency.
    """
    
    def __init__(self, config: BaseServiceConfig):
        """
        Initialize the RoomService.
        
        Args:
            config: Service configuration including max_workers and other settings
        """
        self.max_workers = config.get('max_workers', 10)
        self.environments = {}
        self.env_configs = {}
        self.config = config
        if self.config.use_state_reward:
            self.top_strings_tracker_grounding = TopKStringTracker(self.config.top_strings_m)
            self.top_strings_tracker_worldmodeling = TopKStringTracker(self.config.top_strings_m)

    def create_environments_batch(self, ids2configs: Dict[Any, Any]) -> None:
        """
        Create multiple Room environments in parallel.
        
        Args:
            ids2configs: A dictionary where each key is an environment ID and the corresponding
                        value is the configuration for that environment.
                Each config should contain:
                - env_name: Should be "room"
                - env_config: Room specific configuration
        """
        def create_single_env(env_id, config):
            env_name = config.get('env_name', 'room')
            if env_name != 'room':
                return env_id, None, f"Expected environment type 'room', got '{env_name}'"
            
            try:
                env_config_dict = config.get('env_config', {})
                env_config = RoomEnvConfig(**env_config_dict)
                env = RoomEnv(env_config)
                return env_id, (env, env_config), None
            except Exception as e:
                return env_id, None, str(e)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(create_single_env, env_id, config): env_id 
                for env_id, config in ids2configs.items()
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error creating environment {env_id}: {error}")
                    continue
                
                env, env_config = result
                self.environments[env_id] = env
                self.env_configs[env_id] = env_config
    
    def reset_batch(self, ids2seeds: Dict[Any, Any]) -> Dict[Any, Tuple[Any, Any]]:
        """
        Reset multiple Room environments in parallel.
        
        Args:
            ids2seeds: A dictionary where each key is an environment ID and the corresponding
                     value is a seed value (or None for using default seeding behavior).
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, info)
        """
        results = {}
        
        def reset_single_env(env_id, seed):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                observation, info = env.reset(seed=seed)
                serialized_observation = serialize_observation(observation)
                return env_id, (serialized_observation, info), None
            except Exception as e:
                return env_id, None, str(e)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(reset_single_env, env_id, seed): env_id 
                for env_id, seed in ids2seeds.items()
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error resetting environment {env_id}: {error}")
                    results[env_id] = ({}, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    @service_state_reward_wrapper
    def step_batch(self, ids2actions: Dict[Any, Any]) -> Dict[Any, Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple Room environments in parallel.
        
        Args:
            ids2actions: A dictionary where each key is an environment ID and the corresponding
                       value is the action to execute in that environment.
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, reward, done, info)
        """
        results = {}
        
        def step_single_env(env_id, action):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                observation, reward, done, info = env.step(action)
                serialized_observation = serialize_observation(observation)
                return env_id, (serialized_observation, reward, done, info), None
            except Exception as e:
                return env_id, None, str(e)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(step_single_env, env_id, action): env_id 
                for env_id, action in ids2actions.items()
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error stepping environment {env_id}: {error}")
                    results[env_id] = ({}, 0.0, True, {"error": error})
                else:
                    results[env_id] = result
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[Any, float]:
        """
        Compute the total reward for multiple Room environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its computed total reward
        """
        results = {}
        
        def compute_reward_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                return env_id, env.total_reward, None
            except Exception as e:
                return env_id, None, str(e)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(compute_reward_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error computing reward for environment {env_id}: {error}")
                    results[env_id] = 0.0
                else:
                    results[env_id] = result
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[Any, str]:
        """
        Get system prompts for multiple Room environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its corresponding system prompt string
        """
        results = {}
        
        def get_system_prompt_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, None, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                return env_id, env.system_prompt(), None
            except Exception as e:
                return env_id, None, str(e)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(get_system_prompt_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error getting system prompt for environment {env_id}: {error}")
                    results[env_id] = ""
                else:
                    results[env_id] = result
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple Room environments and clean up resources in parallel.
        
        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
        """
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        def close_single_env(env_id):
            try:
                if env_id not in self.environments:
                    return env_id, f"Environment {env_id} not found"
                
                env = self.environments[env_id]
                env.close()
                return env_id, None
            except Exception as e:
                return env_id, str(e)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(close_single_env, env_id) for env_id in env_ids]
            
            for future in as_completed(futures):
                env_id, error = future.result()
                if error:
                    print(f"Error closing environment {env_id}: {error}")
        
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
