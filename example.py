#!/usr/bin/env python3
"""
Example script for CyberRL
"""

import os
import torch
import numpy as np
import logging
from datetime import datetime

from cyberrl.environment import PentestEnv
from cyberrl.agent import DQNAgent
from cyberrl.utils.logger import setup_logger
from cyberrl.utils.visualization import plot_learning_curve, visualize_network


def main():
    # Setup logger
    log_dir = "example_logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("example", log_dir)
    
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    logger.info("Creating environment...")
    env = PentestEnv(
        target_network="192.168.1.0/24",
        max_steps=50,
        real_execution=False,  # Simulation mode
        render_mode="human",   # Render the environment
    )
    
    # Create agent
    logger.info("Creating agent...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    agent = DQNAgent(
        env=env,
        action_space=env.action_space,
        memory_size=10000,
        batch_size=64,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.1,
        epsilon_frames=50000,
        target_update_freq=10,
        lr=0.001,
        device=device,
        grad_clip_value=1.0,
        model_type="dueling_dqn",
        hidden_dim=128
    )
    
    # Training parameters
    num_episodes = 5  # Just a few episodes for demonstration
    max_steps = env.max_steps
    
    # Initialize metrics
    episode_rewards = []
    
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
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            
            # Print step info
            if info['success']:
                print(f"Step {step}: {info['action_type']} on {info['target_host']}:{info['port']} was successful")
            else:
                print(f"Step {step}: {info['action_type']} on {info['target_host']}:{info['port']} failed")
            
            if terminated or truncated:
                break
        
        # Log episode metrics
        episode_rewards.append(episode_reward)
        logger.info(f"Episode {episode} finished with reward {episode_reward:.2f} in {step} steps")
    
    # Generate and save report
    report = env.get_report()
    
    # Save report
    os.makedirs("example_results", exist_ok=True)
    with open("example_results/report.json", 'w') as f:
        import json
        json.dump(report, f, indent=4)
    
    # Visualize learning curve
    plot_learning_curve(
        episode_rewards,
        title="Example Learning Curve",
        save_path="example_results/learning_curve.png"
    )
    
    # Visualize network
    visualize_network(
        report['hosts'],
        title="Example Network Visualization",
        save_path="example_results/network_visualization.png"
    )
    
    # Print summary
    print("\n=== Example Run Summary ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average reward: {sum(episode_rewards) / num_episodes:.2f}")
    print(f"Hosts discovered: {len(report['hosts'])}")
    print(f"Hosts with vulnerabilities: {len(report['hosts'])}")
    print(f"Exploited hosts: {sum(1 for h, info in report['hosts'].items() if info['exploited'])}")
    print(f"Results saved to example_results/")
    

if __name__ == "__main__":
    main() 