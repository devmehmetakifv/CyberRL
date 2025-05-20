"""
Configuration utilities for CyberRL
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save a configuration to a file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration to
    """
    logger = logging.getLogger(__name__)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {config_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge a base configuration with an override configuration
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    if override_config is None:
        return base_config
    
    merged_config = base_config.copy()
    
    for key, override_value in override_config.items():
        if (
            key in merged_config and
            isinstance(merged_config[key], dict) and
            isinstance(override_value, dict)
        ):
            # Recursively merge nested dictionaries
            merged_config[key] = merge_configs(merged_config[key], override_value)
        else:
            # Override or add the key
            merged_config[key] = override_value
    
    return merged_config


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration for CyberRL
    
    Returns:
        Default configuration dictionary
    """
    return {
        "environment": {
            "target_network": "192.168.1.0/24",
            "max_steps": 100,
            "real_execution": False,
            "render_mode": "ansi",
            "reward_config": {
                "discover_host": 5.0,
                "open_port": 1.0,
                "service_identified": 2.0,
                "os_identified": 3.0,
                "vulnerability_found": 10.0,
                "successful_exploit": 50.0,
                "privilege_escalation": 75.0,
                "lateral_movement": 20.0,
                "timeout_penalty": -1.0,
                "failed_action": -2.0,
                "redundant_action": -0.5
            }
        },
        "agent": {
            "model_type": "dqn",  # Options: dqn, conv_dqn, hybrid_dqn
            "hidden_dim": 256,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.995,
            "buffer_size": 100000,
            "batch_size": 64,
            "target_update_freq": 10,
            "device": "cuda"  # Options: cuda, cpu
        },
        "training": {
            "num_episodes": 5000,
            "eval_freq": 100,
            "save_freq": 500,
            "max_steps_per_episode": 100,
            "early_stopping_patience": 20,
            "checkpoint_dir": "models/checkpoints",
            "log_dir": "logs",
            "use_wandb": False,
            "wandb_project": "cyberrl"
        }
    } 