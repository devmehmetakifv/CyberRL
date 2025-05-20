"""
DQN Agent for Penetration Testing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from gymnasium import spaces

# Define a transition for experience replay
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    """Experience replay buffer to store and sample transitions"""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, next_state, reward, done):
        """Store a transition in the buffer"""
        self.buffer.append(Transition(state, action, next_state, reward, done))
        
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions from the buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return the current size of the buffer"""
        return len(self.buffer)


class DQNetwork(nn.Module):
    """Deep Q-Network for penetration testing agent"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the DQN
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Size of hidden layers
        """
        super(DQNetwork, self).__init__()
        
        # Define network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)


class DQNAgent:
    """DQN Agent for penetration testing"""
    
    def __init__(self,
                 state_space,
                 action_space,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 hidden_dim: int = 256,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 10,
                 log_level: str = "INFO"):
        """
        Initialize the DQN agent
        
        Args:
            state_space: The observation space of the environment
            action_space: The action space of the environment
            device: Device to run the neural networks on
            hidden_dim: Size of hidden layers in the Q-network
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which to decay epsilon
            buffer_size: Size of the experience replay buffer
            batch_size: Number of samples to use per update
            target_update_freq: How often to update the target network
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger("DQNAgent")
        log_level_dict = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, 
                         "WARNING": logging.WARNING, "ERROR": logging.ERROR}
        self.logger.setLevel(log_level_dict.get(log_level, logging.INFO))
        
        self.device = device
        self.state_space = state_space
        self.action_space = action_space
        
        # Determine state and action dimensions
        self.state_dim = self._get_flattened_dim(state_space)
        
        # For the penetration testing environment, we need to handle a Dict action space
        # We'll focus on the action_type for now, the full action will be constructed later
        # The DQN will predict which action type to take
        if isinstance(action_space, spaces.Dict) and 'action_type' in action_space.spaces:
            self.action_dim = action_space.spaces['action_type'].n
        else:
            raise ValueError("Unsupported action space. Expected Dict with 'action_type'")
        
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        
        # Initialize Q-networks
        self.q_network = DQNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_network = DQNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Set target network to evaluation mode
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Tracking variables
        self.update_count = 0
        self.losses = []
        
    def _get_flattened_dim(self, space) -> int:
        """Calculate the flattened dimension of an observation space"""
        if isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, spaces.Dict):
            return sum(self._get_flattened_dim(subspace) for subspace in space.spaces.values())
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
    
    def _flatten_state(self, state):
        """Flatten a possibly hierarchical state into a 1D tensor"""
        if isinstance(state, dict):
            # If the state is a dict, flatten each component and concatenate
            flattened = []
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    flattened.append(value.flatten())
                elif isinstance(value, dict):
                    flattened.append(self._flatten_state(value).flatten())
                else:
                    flattened.append(np.array([value]))
            return np.concatenate(flattened)
        elif isinstance(state, np.ndarray):
            return state.flatten()
        else:
            return np.array([state])
    
    def select_action(self, state) -> Dict[str, Any]:
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state observation
            
        Returns:
            Action dictionary with action_type, target_host, port, and extra_params
        """
        # Convert the state to a flat tensor
        flat_state = self._flatten_state(state)
        state_tensor = torch.FloatTensor(flat_state).to(self.device).unsqueeze(0)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: Random action type
            action_type = random.randint(0, self.action_dim - 1)
        else:
            # Exploit: Best action type based on current Q-values
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_type = q_values.max(1)[1].item()
        
        # Now construct the full action dictionary with additional parameters
        # Random values for other components
        target_host = random.randint(0, 255)  # Random host in a /24 network
        
        # For ports, prefer common ports
        common_ports = [21, 22, 23, 25, 53, 80, 139, 443, 445, 3306, 3389, 8080, 8443]
        if random.random() < 0.8:  # 80% chance to pick a common port
            port = random.choice(common_ports)
        else:
            port = random.randint(1, 65535)
            
        # Generate exploit parameters based on action type
        extra_params = self._generate_extra_params(action_type)
        
        # Construct the complete action
        action = {
            'action_type': action_type,
            'target_host': target_host,
            'port': port,
            'extra_params': extra_params
        }
        
        return action
    
    def _generate_extra_params(self, action_type: int) -> str:
        """Generate appropriate extra parameters based on the action type"""
        # This would be more sophisticated in a real implementation
        # with context-aware parameter generation
        
        exploit_types = [
            "webapps/wordpress/wp_admin_shell_upload",
            "unix/ftp/vsftpd_234_backdoor",
            "multi/http/apache_mod_cgi_bash_env_exec",
            "multi/mysql/mysql_udf_payload",
            "windows/smb/ms17_010_eternalblue"
        ]
        
        priv_esc_types = [
            "linux_sudo_technique", 
            "windows_kernel_exploit",
            "setuid_binary_exploit", 
            "cron_job_modification"
        ]
        
        lateral_movement_types = [
            "pass_the_hash",
            "ssh_key_reuse",
            "service_account_abuse",
            "smb_psexec"
        ]
        
        # Generate based on action type
        if action_type == 5:  # EXPLOIT_ATTEMPT
            return random.choice(exploit_types)
        elif action_type == 6:  # PRIVILEGE_ESCALATION
            return random.choice(priv_esc_types)
        elif action_type == 7:  # LATERAL_MOVEMENT
            return random.choice(lateral_movement_types)
        else:
            return ""  # No extra params for other action types
    
    def update(self) -> Optional[float]:
        """
        Update the Q-network using a batch from the replay buffer
        
        Returns:
            Loss value if an update was performed, None otherwise
        """
        # Skip if buffer doesn't have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None
            
        # Sample a batch from the buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create tensors for each component
        # First, flatten and convert each state in the batch
        state_batch = torch.FloatTensor(
            np.array([self._flatten_state(s) for s in batch.state])
        ).to(self.device)
        
        # For actions, we're only interested in the action_type
        # Extract action_type from the action dictionaries
        action_batch = torch.LongTensor(
            np.array([a['action_type'] for a in batch.action])
        ).to(self.device)
        
        # Get rewards and convert to tensor
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        
        # Process next states, handling terminal states
        non_final_mask = torch.BoolTensor(
            [not done for done in batch.done]
        ).to(self.device)
        
        non_final_next_states = [s for s, done in zip(batch.next_state, batch.done) if not done]
        non_final_next_states_tensor = torch.FloatTensor(
            np.array([self._flatten_state(s) for s in non_final_next_states])
        ).to(self.device) if non_final_next_states else None
        
        # Compute current Q values
        q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next state values using the target network
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states_tensor is not None and len(non_final_next_states_tensor) > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_network(
                    non_final_next_states_tensor
                ).max(1)[0]
        
        # Compute the expected Q values
        expected_q_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute the loss
        criterion = nn.SmoothL1Loss()  # Huber loss for stability
        loss = criterion(q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        # Update the target network if it's time
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Track the loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def save(self, path: str):
        """
        Save the agent's model and training state
        
        Args:
            path: File path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'losses': self.losses
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load the agent's model and training state
        
        Args:
            path: File path to load the model from
        """
        if not torch.cuda.is_available() and self.device == 'cuda':
            # If loading on CPU from a model saved on GPU
            checkpoint = torch.load(path, map_location='cpu')
        else:
            checkpoint = torch.load(path)
            
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
        self.losses = checkpoint['losses']
        
        self.logger.info(f"Model loaded from {path}")
    
    def store_transition(self, state, action, next_state, reward, done):
        """
        Store a transition in the replay buffer
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether the episode is done
        """
        self.replay_buffer.push(state, action, next_state, reward, done) 