"""
Neural Network Models for Penetration Testing RL Agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNetwork(nn.Module):
    """
    Deep Q-Network for penetration testing agent with advanced architecture
    for handling complex state representations
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the DQN model
        
        Args:
            state_dim: Dimension of the flattened state space
            action_dim: Dimension of the action space
            hidden_dim: Size of hidden layers
        """
        super(DQNetwork, self).__init__()
        
        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Advantage stream (action-dependent)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value stream (state-dependent)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        """
        Forward pass through the network using Dueling DQN architecture
        
        Args:
            x: Input tensor representing the state
            
        Returns:
            Q-values for each action
        """
        features = self.feature_layer(x)
        
        # Dueling DQN: split into value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class PenTestConvDQN(nn.Module):
    """
    Convolutional DQN for processing grid-like penetration testing state representations
    (e.g., network topology, port scans)
    """
    
    def __init__(self, input_channels: int, grid_size: int, action_dim: int):
        """
        Initialize the convolutional DQN
        
        Args:
            input_channels: Number of input channels in the state representation
            grid_size: Size of the grid (e.g., 256 for a /24 network)
            action_dim: Dimension of the action space
        """
        super(PenTestConvDQN, self).__init__()
        
        # Convolutional layers for processing grid-like data
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the flattened conv output
        # Depends on the input grid size and conv parameters
        conv_output_size = self._get_conv_output_size(input_channels, grid_size)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Dueling architecture: Value and Advantage streams
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def _get_conv_output_size(self, input_channels: int, grid_size: int) -> int:
        """
        Calculate the size of the flattened convolutional output
        
        Args:
            input_channels: Number of input channels
            grid_size: Size of the input grid
            
        Returns:
            Size of the flattened conv output
        """
        # Create a dummy input to forward through the conv layers
        dummy_input = torch.zeros(1, input_channels, grid_size)
        conv_output = self.conv_layers(dummy_input)
        return int(np.prod(conv_output.shape))
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor representing the state (must be properly shaped)
            
        Returns:
            Q-values for each action
        """
        # Process through convolutional layers
        conv_out = self.conv_layers(x)
        
        # Flatten the output
        flattened = conv_out.view(conv_out.size(0), -1)
        
        # Process through fully connected layers
        features = self.fc_layers(flattened)
        
        # Dueling architecture
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class HybridPenTestDQN(nn.Module):
    """
    Hybrid DQN that combines processing of network structure data and
    numerical features for a comprehensive penetration testing agent
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the hybrid DQN model
        
        Args:
            state_dim: Dimension of the flattened state space
            action_dim: Dimension of the action space
            hidden_dim: Size of hidden layers
        """
        super(HybridPenTestDQN, self).__init__()
        
        # Network for processing flat features
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM for capturing temporal dependencies in penetration testing steps
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Attention mechanism for focusing on important parts of the state
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layers using dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor representing the state
            hidden: Hidden state for LSTM (optional)
            
        Returns:
            Q-values for each action and the new hidden state
        """
        batch_size = x.size(0)
        
        # Process through feature network
        features = self.feature_network(x)
        
        # Reshape for LSTM - assume sequence length of 1 if not provided
        features = features.view(batch_size, 1, -1)
        
        # Process through LSTM
        if hidden is None:
            lstm_out, new_hidden = self.lstm(features)
        else:
            lstm_out, new_hidden = self.lstm(features, hidden)
        
        # Apply attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Dueling architecture
        value = self.value_stream(context)
        advantage = self.advantage_stream(context)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values, new_hidden 