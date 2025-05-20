#!/usr/bin/env python3
"""
Training script for CyberRL agent
"""

import os
import argparse
import json
import torch
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

from cyberrl.environment import PentestEnv
from cyberrl.agent import DQNAgent
from cyberrl.utils.logger import setup_logger, TrainingLogger
from cyberrl.utils.config import load_config, save_config, get_default_config
from cyberrl.utils.visualization import plot_learning_curve, plot_action_distribution, visualize_network


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train CyberRL penetration testing agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save outputs (defaults to timestamped directory)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
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


def train(config, output_dir, log_level="INFO", device=None):
    """
    Train the CyberRL agent
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save outputs
        log_level: Logging level
        device: Device to use (cuda or cpu, overrides config)
    """
    # Setup logger
    logger = setup_logger("train", output_dir, log_level)
    logger.info(f"Training output directory: {output_dir}")
    
    # Setup training logger for metrics
    training_logger = TrainingLogger(
        log_dir=os.path.join(output_dir, "logs"),
        use_wandb=config["training"]["use_wandb"]
    )
    
    # Initialize wandb if enabled
    if config["training"]["use_wandb"]:
        training_logger.setup_wandb(
            project_name=config["training"]["wandb_project"],
            config=config
        )
    
    # Override device if specified
    if device is not None:
        config["agent"]["device"] = device
    
    # Create environment
    logger.info("Creating environment...")
    env = PentestEnv(
        target_network=config["environment"]["target_network"],
        max_steps=config["environment"]["max_steps"],
        real_execution=config["environment"]["real_execution"],
        render_mode=config["environment"]["render_mode"],
        log_level=log_level,
        reward_config=config["environment"]["reward_config"]
    )
    
    # Create agent
    logger.info("Creating agent...")
    agent = DQNAgent(
        env=env,  # Pass the environment directly
        action_space=env.action_space, # Pass the action_space from env
        memory_size=config["agent"]["buffer_size"],
        batch_size=config.get('agent', {}).get('batch_size', 64), # Use .get for safety
        gamma=config["agent"]["gamma"],
        epsilon_start=config["agent"]["epsilon_start"],
        epsilon_final=config["agent"]["epsilon_min"],  # Mapped from epsilon_min
        epsilon_frames=config["agent"]["epsilon_frames"],
        target_update_freq=config["agent"]["target_update_freq"],
        lr=config["agent"]["learning_rate"],
        device=config["agent"]["device"],
        grad_clip_value=config["agent"]["grad_clip_value"],
        model_type=config["agent"]["model_type"],
        hidden_dim=config["agent"]["hidden_dim"]
    )
    
    # Training parameters
    num_episodes = config["training"]["num_episodes"]
    max_steps = config["training"]["max_steps_per_episode"]
    eval_freq = config["training"]["eval_freq"]
    save_freq = config["training"]["save_freq"]
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize metrics
    episode_rewards = []
    best_eval_reward = float("-inf")
    no_improvement_count = 0
    
    # Training loop
    logger.info(f"Starting training for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        # Episode loop
        for step in range(1, max_steps + 1):
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, next_state, reward, terminated or truncated)
            
            # Update agent
            loss = agent.update()
            
            # Log metrics
            if loss is not None:
                training_logger.log_step(episode * max_steps + step, {
                    "loss": loss,
                    "reward": reward,
                    "epsilon": agent.epsilon_by_frame(agent.frame)
                })
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # Log episode metrics
        episode_rewards.append(episode_reward)
        training_logger.log_episode(episode, {
            "reward": episode_reward,
            "steps": step,
            "epsilon": agent.epsilon_by_frame(agent.frame)
        })
        
        # Evaluation
        if episode % eval_freq == 0:
            eval_reward, eval_success_rate, exploited_hosts = evaluate(env, agent, num_episodes=5)
            
            training_logger.log_eval(episode, {
                "reward": eval_reward,
                "success_rate": eval_success_rate,
                "exploited_hosts": exploited_hosts
            })
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(checkpoint_dir, "best_model.pt"))
                no_improvement_count = 0
                logger.info(f"New best model with evaluation reward: {best_eval_reward:.2f}")
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= config["training"]["early_stopping_patience"]:
                logger.info(f"No improvement for {no_improvement_count} evaluations. Stopping early.")
                break
        
        # Save checkpoint
        if episode % save_freq == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{episode}.pt"))
            
            # Also save the episode rewards and metrics
            plot_learning_curve(
                episode_rewards,
                title="CyberRL Learning Curve",
                save_path=os.path.join(output_dir, "learning_curve.png")
            )
            
            # Save action distribution
            action_history = env.get_action_history()
            plot_action_distribution(
                action_history,
                title="Agent Action Distribution",
                save_path=os.path.join(output_dir, "action_distribution.png")
            )
    
    # Final save
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"))
    
    # Save metrics
    training_logger.save_metrics(os.path.join(output_dir, "metrics.json"))
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    eval_reward, eval_success_rate, exploited_hosts = evaluate(
        env, agent, num_episodes=10, render=True, generate_report=True,
        report_path=os.path.join(output_dir, "final_report.json"),
        visualize_path=os.path.join(output_dir, "network_visualization.png")
    )
    
    logger.info(f"Final evaluation - Reward: {eval_reward:.2f}, Success rate: {eval_success_rate:.2f}")
    logger.info(f"Training completed. Results saved to {output_dir}")
    
    # Finish wandb if enabled
    if config["training"]["use_wandb"]:
        training_logger.finish_wandb()
    
    return {
        "best_eval_reward": best_eval_reward,
        "final_eval_reward": eval_reward,
        "final_success_rate": eval_success_rate,
        "exploited_hosts": exploited_hosts,
        "episodes_completed": episode
    }


def evaluate(env, agent, num_episodes=5, render=False, generate_report=False, 
             report_path=None, visualize_path=None):
    """
    Evaluate the agent
    
    Args:
        env: Environment
        agent: Agent
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        generate_report: Whether to generate a penetration test report
        report_path: Path to save the report
        visualize_path: Path to save the network visualization
        
    Returns:
        avg_reward: Average reward across episodes
        success_rate: Fraction of episodes where the agent reached a terminal state
        avg_exploited_hosts: Average number of hosts exploited
    """
    rewards = []
    success_count = 0
    exploited_hosts_counts = []
    
    # Store epsilon and set it to minimum for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = agent.epsilon_min
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Select action (greedy)
            with torch.no_grad():
                action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Render if requested
            if render:
                env.render()
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            done = terminated or truncated
            
            # Check if goal was reached
            if terminated:
                success_count += 1
        
        # Collect metrics
        rewards.append(episode_reward)
        exploited_hosts_counts.append(len(env.exploited))
    
    # Generate report for the last episode if requested
    if generate_report:
        report = env.get_report()
        
        if report_path:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
        
        if visualize_path:
            visualize_network(report['hosts'], save_path=visualize_path)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Calculate metrics
    avg_reward = sum(rewards) / num_episodes
    success_rate = success_count / num_episodes
    avg_exploited_hosts = sum(exploited_hosts_counts) / num_episodes
    
    return avg_reward, success_rate, avg_exploited_hosts


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
        output_dir = os.path.join("results", f"run-{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, os.path.join(output_dir, "config.yaml"))
    
    # Train the agent
    results = train(config, output_dir, args.log_level, args.device)
    
    # Save results summary
    with open(os.path.join(output_dir, "results_summary.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 