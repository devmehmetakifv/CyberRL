#!/usr/bin/env python3
"""
Inference script for running a trained CyberRL agent
"""

import os
import argparse
import json
import torch
import numpy as np
import logging
from datetime import datetime

from cyberrl.environment import PentestEnv
from cyberrl.agent import DQNAgent
from cyberrl.utils.logger import setup_logger
from cyberrl.utils.config import load_config, get_default_config
from cyberrl.utils.visualization import visualize_network, plot_action_distribution


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run CyberRL penetration testing agent")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--target", type=str, default=None,
                        help="Target network (overrides config)")
    parser.add_argument("--real-execution", action="store_true", 
                        help="Execute commands in the real network environment")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during execution")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save outputs (defaults to timestamped directory)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum steps per episode (overrides config)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu, overrides config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_agent(model_path, config, target_network=None, real_execution=False, 
              render=False, output_dir=None, max_steps=None, log_level="INFO",
              device=None, num_episodes=1):
    """
    Run the trained agent
    
    Args:
        model_path: Path to the trained model
        config: Configuration dictionary
        target_network: Target network (overrides config)
        real_execution: Whether to execute commands in the real network
        render: Whether to render the environment
        output_dir: Directory to save outputs
        max_steps: Maximum steps per episode
        log_level: Logging level
        device: Device to use (overrides config)
        num_episodes: Number of episodes to run
        
    Returns:
        reports: List of penetration test reports for each episode
    """
    # Setup logger
    logger = setup_logger("infer", output_dir, log_level)
    logger.info(f"Inference output directory: {output_dir}")
    
    # Override config values if specified
    if target_network is not None:
        config["environment"]["target_network"] = target_network
    
    if max_steps is not None:
        config["environment"]["max_steps"] = max_steps
    
    if device is not None:
        config["agent"]["device"] = device
    
    # Determine render mode
    render_mode = "human" if render else None
    
    # Create environment
    logger.info("Creating environment...")
    env = PentestEnv(
        target_network=config["environment"]["target_network"],
        max_steps=config["environment"]["max_steps"],
        real_execution=real_execution,
        render_mode=render_mode,
        log_level=log_level,
        reward_config=config["environment"]["reward_config"]
    )
    
    # Create agent
    logger.info("Creating agent...")
    agent = DQNAgent(
        env=env,
        action_space=env.action_space,
        memory_size=config["agent"]["buffer_size"],
        batch_size=config["agent"]["batch_size"],
        gamma=config["agent"]["gamma"],
        epsilon_start=config["agent"]["epsilon_min"],  # Use minimum epsilon for inference
        epsilon_final=config["agent"]["epsilon_min"],
        epsilon_frames=config["agent"]["epsilon_frames"],  # Won't be used during inference
        target_update_freq=config["agent"]["target_update_freq"],
        lr=config["agent"]["learning_rate"],
        device=config["agent"]["device"],
        grad_clip_value=config["agent"]["grad_clip_value"],
        model_type=config["agent"]["model_type"],
        hidden_dim=config["agent"]["hidden_dim"]
    )
    
    # Load trained model
    logger.info(f"Loading trained model from {model_path}...")
    agent.load(model_path)
    
    # Set epsilon to minimum for greedy action selection
    agent.epsilon = agent.epsilon_min
    
    reports = []
    total_rewards = []
    
    # Run episodes
    for episode in range(1, num_episodes + 1):
        logger.info(f"Starting episode {episode}/{num_episodes}...")
        
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Episode loop
        while not done and step < config["environment"]["max_steps"]:
            step += 1
            
            # Select action (greedy)
            with torch.no_grad():
                action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            done = terminated or truncated
            
            logger.info(f"Step {step}: Action={info['action_type']}, "
                       f"Target={info['target_host']}:{info['port']}, "
                       f"Success={info['success']}, Reward={reward:.2f}")
            
            if render:
                logger.info(f"Result: {info['message']}")
        
        # Generate report
        report = env.get_report()
        reports.append(report)
        total_rewards.append(episode_reward)
        
        # Save report
        report_path = os.path.join(output_dir, f"report_episode_{episode}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Episode {episode} completed with total reward {episode_reward:.2f}")
        logger.info(f"Report saved to {report_path}")
        
        # Visualize network
        visualize_network(
            report['hosts'],
            title=f"Network State - Episode {episode}",
            save_path=os.path.join(output_dir, f"network_vis_episode_{episode}.png")
        )
    
    # Save action distribution
    action_history = env.get_action_history()
    plot_action_distribution(
        action_history,
        title="Agent Action Distribution",
        save_path=os.path.join(output_dir, "action_distribution.png")
    )
    
    # Save summary
    summary = {
        "num_episodes": num_episodes,
        "avg_reward": sum(total_rewards) / num_episodes,
        "total_hosts_discovered": len(report['hosts']),
        "total_hosts_exploited": len([h for h, info in report['hosts'].items() if info['exploited']]),
        "hosts_with_admin_privileges": len([h for h, info in report['hosts'].items() 
                                         if info['privilege_level'] == 2]),
        "environment_config": {
            "target_network": config["environment"]["target_network"],
            "max_steps": config["environment"]["max_steps"],
            "real_execution": real_execution
        }
    }
    
    # Save summary
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    return reports


def main():
    """Main function"""
    args = parse_args()
    
    # Set seeds for reproducibility
    set_seed(args.seed)
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file {args.config} not found. Using default configuration.")
        config = get_default_config()
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join("results", f"infer-{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the agent
    run_agent(
        model_path=args.model,
        config=config,
        target_network=args.target,
        real_execution=args.real_execution,
        render=args.render,
        output_dir=output_dir,
        max_steps=args.max_steps,
        log_level=args.log_level,
        device=args.device
    )
    
    print(f"Inference completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 