"""
Visualization utilities for CyberRL
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import json
import logging
from datetime import datetime


def plot_training_metrics(metrics_file: str, save_dir: Optional[str] = None):
    """
    Plot training metrics from a metrics file
    
    Args:
        metrics_file: Path to the metrics file (JSON format)
        save_dir: Directory to save plots to (if None, plots are just shown)
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(metrics_file):
        logger.error(f"Metrics file not found: {metrics_file}")
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Create save directory if it doesn't exist
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot episode metrics
    if "episode_metrics" in metrics and metrics["episode_metrics"]:
        plot_episode_metrics(metrics["episode_metrics"], save_dir)
    
    # Plot step metrics
    if "step_metrics" in metrics and metrics["step_metrics"]:
        plot_step_metrics(metrics["step_metrics"], save_dir)
    
    # Plot evaluation metrics
    if "eval_metrics" in metrics and metrics["eval_metrics"]:
        plot_eval_metrics(metrics["eval_metrics"], save_dir)


def plot_episode_metrics(episode_metrics: Dict[str, Dict], save_dir: Optional[str] = None):
    """
    Plot episode metrics
    
    Args:
        episode_metrics: Dictionary of episode metrics
        save_dir: Directory to save plots to (if None, plots are just shown)
    """
    # Convert the episode numbers from strings back to integers
    episodes = [int(ep) for ep in episode_metrics.keys()]
    
    # Get all unique metric names
    metric_names = set()
    for metrics in episode_metrics.values():
        metric_names.update(metrics.keys())
    
    # Plot each metric
    for metric in metric_names:
        plt.figure(figsize=(10, 6))
        
        # Extract values for this metric (skipping episodes that don't have it)
        values = []
        eps = []
        for ep, metrics in episode_metrics.items():
            if metric in metrics:
                eps.append(int(ep))
                values.append(metrics[metric])
        
        # Plot if we have data
        if values:
            plt.plot(eps, values, marker="o")
            plt.title(f"Episode {metric}")
            plt.xlabel("Episode")
            plt.ylabel(metric)
            plt.grid(True)
            
            # Add smoothed line if we have enough data
            if len(values) > 10:
                window_size = max(5, len(values) // 20)
                smoothed = pd.Series(values).rolling(window=window_size).mean().to_numpy()
                plt.plot(eps, smoothed, 'r-', linewidth=2, label=f"Moving avg (window={window_size})")
                plt.legend()
            
            if save_dir is not None:
                filename = os.path.join(save_dir, f"episode_{metric}.png")
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()


def plot_step_metrics(step_metrics: Dict[str, Dict], save_dir: Optional[str] = None):
    """
    Plot step metrics
    
    Args:
        step_metrics: Dictionary of step metrics
        save_dir: Directory to save plots to (if None, plots are just shown)
    """
    # Convert the step numbers from strings back to integers
    steps = [int(step) for step in step_metrics.keys()]
    
    # Get all unique metric names
    metric_names = set()
    for metrics in step_metrics.values():
        metric_names.update(metrics.keys())
    
    # Plot each metric
    for metric in metric_names:
        plt.figure(figsize=(10, 6))
        
        # Extract values for this metric (skipping steps that don't have it)
        values = []
        st = []
        for step, metrics in step_metrics.items():
            if metric in metrics:
                st.append(int(step))
                values.append(metrics[metric])
        
        # Plot if we have data
        if values:
            plt.plot(st, values, marker=".", alpha=0.5)
            plt.title(f"Step {metric}")
            plt.xlabel("Step")
            plt.ylabel(metric)
            plt.grid(True)
            
            # Add smoothed line if we have enough data
            if len(values) > 10:
                window_size = max(10, len(values) // 50)
                smoothed = pd.Series(values).rolling(window=window_size).mean().to_numpy()
                plt.plot(st, smoothed, 'r-', linewidth=2, label=f"Moving avg (window={window_size})")
                plt.legend()
            
            if save_dir is not None:
                filename = os.path.join(save_dir, f"step_{metric}.png")
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()


def plot_eval_metrics(eval_metrics: Dict[str, Dict], save_dir: Optional[str] = None):
    """
    Plot evaluation metrics
    
    Args:
        eval_metrics: Dictionary of evaluation metrics
        save_dir: Directory to save plots to (if None, plots are just shown)
    """
    # Convert the step numbers from strings back to integers
    steps = [int(step) for step in eval_metrics.keys()]
    
    # Get all unique metric names
    metric_names = set()
    for metrics in eval_metrics.values():
        metric_names.update(metrics.keys())
    
    # Plot each metric
    for metric in metric_names:
        plt.figure(figsize=(10, 6))
        
        # Extract values for this metric (skipping steps that don't have it)
        values = []
        st = []
        for step, metrics in eval_metrics.items():
            if metric in metrics:
                st.append(int(step))
                values.append(metrics[metric])
        
        # Plot if we have data
        if values:
            plt.plot(st, values, marker="o")
            plt.title(f"Evaluation {metric}")
            plt.xlabel("Step")
            plt.ylabel(metric)
            plt.grid(True)
            
            if save_dir is not None:
                filename = os.path.join(save_dir, f"eval_{metric}.png")
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()


def plot_learning_curve(
    rewards: List[float], 
    window_size: int = 20,
    title: str = "Learning Curve",
    save_path: Optional[str] = None
):
    """
    Plot a learning curve with moving average
    
    Args:
        rewards: List of rewards
        window_size: Window size for moving average
        title: Plot title
        save_path: Path to save the plot (if None, plot is just shown)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw rewards
    plt.plot(rewards, alpha=0.3, label="Raw rewards")
    
    # Plot moving average
    if len(rewards) > window_size:
        smoothed = pd.Series(rewards).rolling(window=window_size).mean().to_numpy()
        plt.plot(smoothed, 'r-', linewidth=2, label=f"Moving avg (window={window_size})")
    
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_network(
    hosts: Dict[str, Dict],
    title: str = "Network Visualization",
    save_path: Optional[str] = None
):
    """
    Visualize network state from penetration testing environment
    
    Args:
        hosts: Dictionary of host information
        title: Plot title
        save_path: Path to save the plot (if None, plot is just shown)
    """
    try:
        import networkx as nx
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes for each host
        for host, info in hosts.items():
            # Node attributes
            attrs = {
                'exploited': info.get('exploited', False),
                'privilege_level': info.get('privilege_level', 0),
                'num_open_ports': len(info.get('open_ports', [])),
                'num_vulns': sum(len(vulns) for vulns in info.get('vulnerabilities', {}).values())
            }
            G.add_node(host, **attrs)
        
        # Add edges between hosts (if lateral movement is possible)
        # This is a simplified representation for visualization
        for host1 in hosts:
            for host2 in hosts:
                if host1 != host2:
                    # Add edge if both hosts are discovered and one is exploited
                    if hosts[host1].get('exploited', False) or hosts[host2].get('exploited', False):
                        G.add_edge(host1, host2, weight=0.5)
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Define node colors based on exploitation status and privilege level
        node_colors = []
        for node in G.nodes():
            if G.nodes[node]['exploited']:
                if G.nodes[node]['privilege_level'] == 2:  # Admin privileges
                    node_colors.append('red')
                else:  # User privileges
                    node_colors.append('orange')
            elif G.nodes[node]['num_vulns'] > 0:
                node_colors.append('yellow')
            else:
                node_colors.append('green')
        
        # Define node sizes based on number of open ports
        node_sizes = [100 + 50 * G.nodes[node]['num_open_ports'] for node in G.nodes()]
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=node_sizes, font_weight='bold')
        
        # Add legend
        plt.figure(figsize=(1, 2))
        plt.scatter([], [], c='red', s=100, label='Exploited (Admin)')
        plt.scatter([], [], c='orange', s=100, label='Exploited (User)')
        plt.scatter([], [], c='yellow', s=100, label='Vulnerable')
        plt.scatter([], [], c='green', s=100, label='Discovered')
        plt.legend()
        
        plt.title(title)
        
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    except ImportError:
        logging.warning("networkx package not installed. Cannot visualize network.")


def plot_action_distribution(
    action_history: List[Dict],
    title: str = "Action Type Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of action types taken by the agent
    
    Args:
        action_history: List of action dictionaries from environment
        title: Plot title
        save_path: Path to save the plot (if None, plot is just shown)
    """
    # Count action types
    action_counts = {}
    for action in action_history:
        action_type = action['action_type']
        if action_type in action_counts:
            action_counts[action_type] += 1
        else:
            action_counts[action_type] = 1
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(action_counts.keys(), action_counts.values())
    plt.title(title)
    plt.xlabel("Action Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
    # Also plot success rate by action type
    success_counts = {}
    total_counts = {}
    
    for action in action_history:
        action_type = action['action_type']
        success = action['success']
        
        if action_type not in total_counts:
            total_counts[action_type] = 0
            success_counts[action_type] = 0
            
        total_counts[action_type] += 1
        if success:
            success_counts[action_type] += 1
    
    # Calculate success rates
    success_rates = {
        action_type: success_counts[action_type] / total_counts[action_type]
        for action_type in total_counts
    }
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(success_rates.keys(), success_rates.values())
    plt.title(f"{title} - Success Rates")
    plt.xlabel("Action Type")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path.replace('.png', '_success_rates.png'))
        plt.close()
    else:
        plt.show() 