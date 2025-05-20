"""
Logger utility for CyberRL
"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Setup a logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set log level
    level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    logger.setLevel(level_dict.get(level, logging.INFO))
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
    )
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    
    # File handler - include timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{timestamp}.log")
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


class TrainingLogger:
    """
    Logger specifically for tracking training metrics
    """
    
    def __init__(self, log_dir: str = "logs", use_wandb: bool = False):
        """
        Initialize the training logger
        
        Args:
            log_dir: Directory to store logs
            use_wandb: Whether to use Weights & Biases for tracking
        """
        self.logger = setup_logger("training", log_dir)
        self.use_wandb = use_wandb
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                self.logger.warning("wandb not installed. Falling back to local logging only.")
                self.use_wandb = False
        
        # Initialize metrics dictionaries
        self.episode_metrics = {}
        self.step_metrics = {}
        self.eval_metrics = {}
        
    def log_episode(self, episode: int, metrics: dict):
        """
        Log metrics for an episode
        
        Args:
            episode: Episode number
            metrics: Dictionary of metrics to log
        """
        # Format the metrics as a string for logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Episode {episode} - {metrics_str}")
        
        # Store metrics
        self.episode_metrics[episode] = metrics
        
        # Log to wandb if enabled
        if self.use_wandb:
            metrics_dict = {f"episode/{k}": v for k, v in metrics.items()}
            metrics_dict["episode"] = episode
            self.wandb.log(metrics_dict)
    
    def log_step(self, step: int, metrics: dict):
        """
        Log metrics for a training step
        
        Args:
            step: Step number
            metrics: Dictionary of metrics to log
        """
        # Only log steps at regular intervals to avoid flooding logs
        if step % 1000 == 0:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.debug(f"Step {step} - {metrics_str}")
        
        # Store metrics
        self.step_metrics[step] = metrics
        
        # Log to wandb if enabled
        if self.use_wandb:
            metrics_dict = {f"step/{k}": v for k, v in metrics.items()}
            metrics_dict["step"] = step
            self.wandb.log(metrics_dict)
    
    def log_eval(self, step: int, metrics: dict):
        """
        Log evaluation metrics
        
        Args:
            step: Current step or episode
            metrics: Dictionary of evaluation metrics
        """
        # Format the metrics as a string for logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Evaluation at step {step} - {metrics_str}")
        
        # Store metrics
        self.eval_metrics[step] = metrics
        
        # Log to wandb if enabled
        if self.use_wandb:
            metrics_dict = {f"eval/{k}": v for k, v in metrics.items()}
            metrics_dict["step"] = step
            self.wandb.log(metrics_dict)
    
    def save_metrics(self, path: str):
        """
        Save all collected metrics to a file
        
        Args:
            path: Path to save metrics to
        """
        import json
        
        metrics = {
            "episode_metrics": self.episode_metrics,
            "step_metrics": self.step_metrics,
            "eval_metrics": self.eval_metrics
        }
        
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        self.logger.info(f"Metrics saved to {path}")
    
    def setup_wandb(self, project_name: str, config: dict = None):
        """
        Setup Weights & Biases tracking
        
        Args:
            project_name: Name of the W&B project
            config: Configuration dictionary to log
        """
        if not self.use_wandb:
            return
        
        self.wandb.init(project=project_name, config=config)
        self.logger.info(f"Weights & Biases tracking initialized for project {project_name}")
    
    def finish_wandb(self):
        """Finish the wandb run"""
        if self.use_wandb:
            self.wandb.finish()
            self.logger.info("Weights & Biases tracking finished") 