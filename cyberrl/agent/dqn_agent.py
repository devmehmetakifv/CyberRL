#!/usr/bin/env python3
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
import torch.nn.functional as F

# Import the model architectures
from cyberrl.models.dqn_model import DuelingDQNetwork, PenTestConvDQN, HybridPenTestDQN

# Define a transition for experience replay
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, next_state, reward, done):
        """
        Store a transition
        
        Args:
            state: Current state
            action: Action taken (as an index)
            next_state: Next state
            reward: Reward received
            done: Whether the episode ended
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = Transition(state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for penetration testing"""
    
    def __init__(
        self, 
        env, 
        action_space: spaces.Space,
        memory_size: int = 10000,
        batch_size: int = 32,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_frames: int = 10000,
        target_update_freq: int = 1000,
        lr: float = 0.0001,
        device: str = "cpu",
        grad_clip_value: float = 1.0,
        model_type: str = "dueling_dqn",
        hidden_dim: int = 512
    ):
        """
        Initialize the DQN Agent
        
        Args:
            env: The environment
            action_space: Action space from the environment
            memory_size: Size of the replay buffer
            batch_size: Mini-batch size for training
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_final: Final epsilon after annealing
            epsilon_frames: Number of frames over which to anneal epsilon
            target_update_freq: Frequency of target network updates
            lr: Learning rate
            device: Device to use for tensor operations
            grad_clip_value: Maximum gradient value for clipping
            model_type: Type of DQN model to use ('dqn', 'dueling_dqn', 'conv_dqn', 'hybrid_dqn')
            hidden_dim: Size of hidden layers in the neural network
        """
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_frames = epsilon_frames
        self.target_update_freq = target_update_freq
        self.device = device
        self.grad_clip_value = grad_clip_value
        self.model_type = model_type
        
        # Initialize frame counter and update counter
        self.frame = 0
        self.update_count = 0
        
        # Setup logger
        self.logger = logging.getLogger("DQNAgent")
        self.logger.setLevel(logging.INFO)
        
        # Calculate state_dim based on actual environment observations
        initial_obs, _ = self.env.reset(seed=42)  # Use fixed seed for consistency
        processed_initial_obs = self._preprocess_state(initial_obs)
        self.state_dim = len(processed_initial_obs)
        self.logger.info(f"Calculated state dimension: {self.state_dim}")
        
        # For the penetration testing environment, we need to handle a Dict action space
        # Instead of just predicting action_type, we'll use a discretized combined action space
        # that maps a single action index to a full action dictionary including target_host, port, and extra_params
        if isinstance(action_space, spaces.Dict) and 'action_type' in action_space.spaces:
            # Create a discretized action space that combines all action components
            # This is a simplified approach - in a full implementation, we would define a more 
            # sophisticated mapping between action indices and full action dictionaries
            
            # Get the dimensions of each action component
            action_type_dim = action_space.spaces['action_type'].n
            
            # For target_host, we'll discretize to 256 possible values (0-255 for the last octet in a /24 network)
            target_host_dim = 256
            
            # For ports, we'll use common ports plus some random ones, total of 20
            self.common_ports = [21, 22, 23, 25, 53, 80, 139, 443, 445, 3306, 3389, 8080, 8443]
            self.uncommon_ports = [1024, 2048, 4444, 5000, 6667, 8000, 9000]
            port_dim = len(self.common_ports) + len(self.uncommon_ports)
            
            # For exploits/techniques
            self.exploit_types = [
                "webapps/wordpress/wp_admin_shell_upload",
                "unix/ftp/vsftpd_234_backdoor",
                "multi/http/apache_mod_cgi_bash_env_exec",
                "multi/mysql/mysql_udf_payload",
                "windows/smb/ms17_010_eternalblue"
            ]
            
            self.priv_esc_types = [
                "linux_sudo_technique", 
                "windows_kernel_exploit",
                "setuid_binary_exploit", 
                "cron_job_modification"
            ]
            
            self.lateral_movement_types = [
                "pass_the_hash",
                "ssh_key_reuse",
                "service_account_abuse",
                "smb_psexec"
            ]
            
            # Calculate total extra_params options
            extra_params_dim = max(len(self.exploit_types), len(self.priv_esc_types), len(self.lateral_movement_types))
            
            # Our action space will be action_type-dependent
            # For some actions, target_host, port, and extra_params don't matter
            # For others, they do. We'll create an efficient mapping
            
            # Calculate total action space size
            # This is a simplified approach - we'd want a more efficient mapping in practice
            self.action_dim = action_type_dim * target_host_dim * port_dim * extra_params_dim
            
            # Create mappings between action index and full action dictionary
            self.setup_action_mappings(action_type_dim, target_host_dim, port_dim, extra_params_dim)
            
            print(f"Configured composite action space with {self.action_dim} possible actions")
        else:
            raise ValueError("Unsupported action space. Expected Dict with 'action_type'")
        
        self.logger.info(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        
        # Initialize Q-networks based on the model type
        if model_type == 'dueling_dqn' or model_type == 'dqn':
            self.q_network = DuelingDQNetwork(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(device)
            self.target_network = DuelingDQNetwork(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(device)
        elif model_type == 'conv_dqn':
            # For now, raise an error as this requires special state processing
            raise NotImplementedError("ConvDQN not yet implemented. Requires special state processing.")
        elif model_type == 'hybrid_dqn':
            # For now, raise an error as this requires special state processing
            raise NotImplementedError("HybridDQN not yet implemented. Requires special state processing.")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Copy weights from q_network to target_network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Set target network to evaluation mode
        self.target_network.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
        
        # Tracking variables
        self.losses = []
        
    def _preprocess_state(self, state: Union[np.ndarray, Dict]) -> np.ndarray:
        """
        Preprocess a state observation to be fed into the neural network.
        Flattens dictionary observations from PentestEnv into a single vector.
        
        Args:
            state: Raw state from the environment, could be a dictionary or array
            
        Returns:
            Flattened numpy array representing the state
        """
        # If state is already a numpy array, just return it
        if isinstance(state, np.ndarray):
            return state
        
        # If state is a dictionary from PentestEnv, flatten it systematically
        if isinstance(state, dict):
            flattened_components = []
            
            # Process discovered_hosts (binary vector of which hosts have been found)
            if 'discovered_hosts' in state:
                flattened_components.append(state['discovered_hosts'].flatten())
                
            # Process open_ports (2D matrix of host x port, only for ports < 1024)
            if 'open_ports' in state:
                flattened_components.append(state['open_ports'].flatten())
                
            # Process os_detected (operating systems detected on hosts)
            if 'os_detected' in state:
                flattened_components.append(state['os_detected'].flatten())
                
            # Process service_info (services detected on ports)
            if 'service_info' in state:
                flattened_components.append(state['service_info'].flatten())
                
            # Process vulnerabilities (detected vulnerabilities)
            if 'vulnerabilities' in state:
                flattened_components.append(state['vulnerabilities'].flatten())
                
            # Process exploited (which hosts have been successfully exploited)
            if 'exploited' in state:
                flattened_components.append(state['exploited'].flatten())
                
            # Process privilege_level (current privilege level on exploited hosts)
            if 'privilege_level' in state:
                flattened_components.append(state['privilege_level'].flatten())
                
            # Process network_map (connectivity between hosts)
            if 'network_map' in state:
                flattened_components.append(state['network_map'].flatten())
                
            # Process step_count (current step in the episode)
            if 'step_count' in state and isinstance(state['step_count'], (int, float, np.number)):
                flattened_components.append(np.array([state['step_count']]))
                
            # Concatenate all components into a single flat vector
            if flattened_components:
                try:
                    return np.concatenate(flattened_components)
                except ValueError as e:
                    self.logger.error(f"Error concatenating state components: {e}")
                    self.logger.error(f"Component shapes: {[c.shape for c in flattened_components]}")
                    return np.zeros(self.state_dim)
            else:
                self.logger.warning("No valid components found in state dictionary.")
                return np.zeros(self.state_dim)
                
        # Fallback - should not happen with proper environment implementation
        self.logger.warning("Received unexpected state format. Returning zero vector.")
        return np.zeros(self.state_dim)
    
    def select_action(self, state: Union[np.ndarray, Dict], epsilon: float = 0.0) -> dict:
        """
        Select an action based on the current state using an epsilon-greedy policy with action masking.
        Action masking filters out invalid actions based on the current state.
        
        Args:
            state: The current state observation
            epsilon: Exploration rate
            
        Returns:
            Complete action dictionary with action_type, target_host, port, and extra_params
        """
        # Preprocess the state for the neural network
        processed_state = self._preprocess_state(state)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Random action - still needs to be valid
            # We'll try random actions until we find a valid one
            valid_action = False
            max_tries = 50  # Prevent infinite loops
            tries = 0
            
            while not valid_action and tries < max_tries:
                action_idx = random.randint(0, self.action_dim - 1)
                action = self._map_index_to_action(action_idx)
                valid_action = self._is_action_valid(action, state)
                tries += 1
                
            if not valid_action:
                # Fallback to a default action (e.g., NMAP_SCAN on first host)
                self.logger.warning("Could not find valid random action, using fallback")
                action = {
                    'action_type': 0,  # Assume 0 is a scan action that's generally valid
                    'target_host': 0,
                    'port': 80,
                    'extra_params': ""
                }
        else:
            # Greedy action - use the network but mask out invalid actions
            with torch.no_grad():
                state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                # Create a mask for valid actions
                action_mask = torch.zeros_like(q_values, dtype=torch.bool)
                
                # Check each possible action for validity
                for action_idx in range(self.action_dim):
                    action = self._map_index_to_action(action_idx)
                    if self._is_action_valid(action, state):
                        action_mask[0, action_idx] = True
                
                # If no actions are valid (shouldn't happen), unmask all
                if not torch.any(action_mask):
                    self.logger.warning("No valid actions found, unmasking all actions")
                    action_mask.fill_(True)
                
                # Apply the mask
                masked_q_values = q_values.clone()
                masked_q_values[~action_mask] = float('-inf')
                
                # Select the best valid action
                action_idx = masked_q_values.argmax().item()
                action = self._map_index_to_action(action_idx)
        
        return action
    
    def _is_action_valid(self, action: dict, state: Dict) -> bool:
        """
        Check if an action is valid given the current state.
        
        Args:
            action: Action dictionary to check
            state: Current environment state
            
        Returns:
            True if the action is valid, False otherwise
        """
        action_type = action['action_type']
        target_host = action['target_host']
        port = action['port']
        extra_params = action['extra_params']
        
        # Check if the target host exists in discovered_hosts
        if 'discovered_hosts' in state and isinstance(state['discovered_hosts'], np.ndarray):
            # If the target host is not discovered, the action is invalid
            if target_host >= len(state['discovered_hosts']) or not state['discovered_hosts'][target_host]:
                return False
        
        # Check if action_type is valid for the specific port
        if action_type in [1, 2, 5]:  # Actions that target specific ports (e.g., SERVICE_SCAN, VULNERABILITY_SCAN, EXPLOIT)
            if 'open_ports' in state and isinstance(state['open_ports'], np.ndarray):
                # For high ports (>1023), we can't validate them directly from open_ports
                if port < 1024:
                    # Check if target_host index is valid
                    if target_host < state['open_ports'].shape[0] and port < state['open_ports'].shape[1]:
                        # If port is not open on the target host, action is invalid
                        if not state['open_ports'][target_host][port]:
                            return False
                    else:
                        # Invalid indices into open_ports array
                        return False
                else:
                    # For high ports, we'll be permissive since they aren't tracked
                    # In a production system, we'd check against a more complete port scan result
                    pass  # Allow the action for high ports
        
        # Check if action_type is valid for the current exploitation status
        if action_type in [6, 7]:  # PRIVILEGE_ESCALATION, LATERAL_MOVEMENT
            if 'exploited' in state and isinstance(state['exploited'], np.ndarray):
                # These actions require the host to be already exploited
                if target_host < len(state['exploited']) and not state['exploited'][target_host]:
                    return False
        
        # Check if privilege escalation is valid based on current privilege level
        if action_type == 6:  # PRIVILEGE_ESCALATION
            if 'privilege_level' in state and isinstance(state['privilege_level'], np.ndarray):
                # Can't escalate privileges if already at maximum level (assuming max level is 2)
                if target_host < len(state['privilege_level']) and state['privilege_level'][target_host] >= 2:
                    return False
                    
        # Check if lateral movement makes sense based on network connectivity
        if action_type == 7:  # LATERAL_MOVEMENT
            if 'network_map' in state and isinstance(state['network_map'], np.ndarray):
                # Check if there's at least one host to move to
                connected_hosts = False
                for i in range(len(state['discovered_hosts'])):
                    if i != target_host and state['discovered_hosts'][i]:
                        if target_host < state['network_map'].shape[0] and i < state['network_map'].shape[1]:
                            if state['network_map'][target_host, i]:
                                connected_hosts = True
                                break
                if not connected_hosts:
                    return False
        
        # If we've passed all checks, the action is valid
        return True
    
    def _map_index_to_action(self, action_idx: int) -> dict:
        """
        Maps a flat action index to a complete action dictionary with all components.
        
        Args:
            action_idx: The flat action index
            
        Returns:
            Action dictionary with action_type, target_host, port, and extra_params
        """
        # Ensure action_idx is within bounds
        action_idx = max(0, min(action_idx, self.action_dim - 1))
        
        # Decompose the flat index into component indices
        # Total number of port * extra_params combinations
        port_extra_params = self.port_dim * self.extra_params_dim
        
        # Total number of target_host * port * extra_params combinations
        target_port_extra = self.target_host_dim * port_extra_params
        
        # Calculate component indices
        action_type = action_idx // target_port_extra
        remainder = action_idx % target_port_extra
        
        target_host = remainder // port_extra_params
        remainder = remainder % port_extra_params
        
        port_idx = remainder // self.extra_params_dim
        extra_params_idx = remainder % self.extra_params_dim
        
        # Ensure indices are within bounds of their respective ranges
        action_type = max(0, min(action_type, self.action_type_dim - 1))
        target_host = max(0, min(target_host, self.target_host_dim - 1))
        port_idx = max(0, min(port_idx, self.port_dim - 1))
        extra_params_idx = max(0, min(extra_params_idx, self.extra_params_dim - 1))
        
        # Convert indices to actual values
        # For port, use either common or uncommon ports
        all_ports = self.common_ports + self.uncommon_ports
        port = all_ports[port_idx]
        
        # For extra_params, select based on action_type
        extra_params = self._get_extra_params(action_type, extra_params_idx)
        
        # Construct the full action dictionary
        action = {
            'action_type': int(action_type),
            'target_host': int(target_host),
            'port': int(port),
            'extra_params': extra_params
        }
        
        return action
    
    def _get_extra_params(self, action_type: int, extra_params_idx: int) -> str:
        """
        Get appropriate extra parameters based on action type and parameter index.
        
        Args:
            action_type: The type of action (e.g., exploit, scan)
            extra_params_idx: Index into the parameters list
            
        Returns:
            String containing the extra parameters
        """
        # Select appropriate parameters based on action type
        if action_type == 5:  # EXPLOIT_ATTEMPT
            if extra_params_idx < len(self.exploit_types):
                return self.exploit_types[extra_params_idx]
        elif action_type == 6:  # PRIVILEGE_ESCALATION
            if extra_params_idx < len(self.priv_esc_types):
                return self.priv_esc_types[extra_params_idx]
        elif action_type == 7:  # LATERAL_MOVEMENT
            if extra_params_idx < len(self.lateral_movement_types):
                return self.lateral_movement_types[extra_params_idx]
                
        return ""  # Default empty string for other action types
    
    def update(self) -> Optional[float]:
        """
        Sample from replay buffer and update the Q-network
        
        Returns:
            The loss value if update was performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample a batch from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors - states are already preprocessed
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done).astype(float)).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.q_network(state_batch)
        state_action_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch)
            next_state_values = next_q_values.max(1)[0]
            
            # Compute the expected Q values
            expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_value)
        
        self.optimizer.step()
        
        # Update target network if it's time
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
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
            'frame': self.frame,
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
        self.frame = checkpoint.get('frame', 0)  # Use get() to handle older models
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
    
    def setup_action_mappings(self, action_type_dim, target_host_dim, port_dim, extra_params_dim):
        """
        Create efficient mappings between action indices and full action dictionaries.
        This is a simplified implementation - a production version would use a more
        sophisticated approach to keep the action space reasonable.
        """
        # In a full implementation, we would use a more sophisticated mapping
        # For demonstration purposes, we'll just store the dimensions
        self.action_type_dim = action_type_dim
        self.target_host_dim = target_host_dim
        self.port_dim = port_dim
        self.extra_params_dim = extra_params_dim
        
        print(f"Action space components: {action_type_dim} action types, " 
              f"{target_host_dim} targets, {port_dim} ports, {extra_params_dim} parameter sets")

    def step(self, state: Union[np.ndarray, Dict]) -> Tuple[dict, np.ndarray, float, bool, dict]:
        """
        Take an action in the environment based on the current state.
        
        Args:
            state: The current state observation
            
        Returns:
            tuple: (action, next_state, reward, done, info)
        """
        # Select action using epsilon-greedy policy
        epsilon = self.epsilon_by_frame(self.frame)
        action = self.select_action(state, epsilon)
        
        # Take action in the environment
        next_state, reward, done, info = self.env.step(action)
        
        # Preprocess states for storage in replay buffer
        processed_state = self._preprocess_state(state)
        processed_next_state = self._preprocess_state(next_state)
        
        # Store transition in replay buffer
        # Convert action to index for efficient storage
        action_idx = self._map_action_to_index(action)
        self.replay_buffer.push(processed_state, action_idx, processed_next_state, reward, done)
        
        # Update frame count
        self.frame += 1
        
        return action, next_state, reward, done, info
        
    def _map_action_to_index(self, action: dict) -> int:
        """
        Maps a complete action dictionary back to a flat action index.
        This is the inverse of _map_index_to_action.
        
        Args:
            action: Action dictionary with action_type, target_host, port, and extra_params
            
        Returns:
            Flat action index
        """
        # Get component indices
        action_type = min(max(0, action['action_type']), self.action_type_dim - 1)
        target_host = min(max(0, action['target_host']), self.target_host_dim - 1)
        
        # Map port to index
        all_ports = self.common_ports + self.uncommon_ports
        try:
            port_idx = all_ports.index(action['port'])
        except ValueError:
            # If port is not in our list, find the closest one
            closest_idx = 0
            min_diff = float('inf')
            for i, p in enumerate(all_ports):
                diff = abs(p - action['port'])
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            port_idx = closest_idx
        
        # Ensure port_idx is within bounds
        port_idx = min(max(0, port_idx), self.port_dim - 1)
        
        # Map extra_params to index based on action type
        extra_params_idx = 0
        if action_type == 5 and action['extra_params'] in self.exploit_types:
            extra_params_idx = self.exploit_types.index(action['extra_params'])
        elif action_type == 6 and action['extra_params'] in self.priv_esc_types:
            extra_params_idx = self.priv_esc_types.index(action['extra_params'])
        elif action_type == 7 and action['extra_params'] in self.lateral_movement_types:
            extra_params_idx = self.lateral_movement_types.index(action['extra_params'])
        
        # Ensure extra_params_idx is within bounds
        extra_params_idx = min(max(0, extra_params_idx), self.extra_params_dim - 1)
        
        # Calculate flat index
        port_extra_params = self.port_dim * self.extra_params_dim
        target_port_extra = self.target_host_dim * port_extra_params
        
        # Combine indices into flat index
        action_idx = (action_type * target_port_extra + 
                      target_host * port_extra_params + 
                      port_idx * self.extra_params_dim + 
                      extra_params_idx)
        
        # Ensure the resulting index is within bounds of the action space
        action_idx = min(max(0, action_idx), self.action_dim - 1)
        
        return action_idx 

    def epsilon_by_frame(self, frame: int) -> float:
        """
        Calculate epsilon value based on frame number for exploration scheduling.
        
        Args:
            frame: Current frame number
            
        Returns:
            Current epsilon value for exploration
        """
        # Linear annealing from epsilon_start to epsilon_final over epsilon_frames frames
        # Then keep epsilon at epsilon_final for the rest of the training
        if frame < self.epsilon_frames:
            return self.epsilon_start - (frame / self.epsilon_frames) * (self.epsilon_start - self.epsilon_final)
        else:
            return self.epsilon_final 