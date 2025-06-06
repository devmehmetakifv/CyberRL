# CyberRL Agent Module

This module contains the reinforcement learning agents for the CyberRL framework.

## DQNAgent

The main agent class is `DQNAgent`, which implements a Deep Q-Network approach to penetration testing.

### Key Features

- Supports composite action spaces with action_type, target_host, port, and exploit parameters
- Uses Dueling DQN architecture for better value estimation
- Implements action masking to filter invalid actions
- Supports different neural network architectures via model_type parameter

### Action Space Size Considerations

The current composite action space approach creates a large action space (typically >200,000 actions). While the action masking approach efficiently filters invalid actions at runtime, the Q-network's output layer remains large, which has memory and computation implications:

- **Memory usage**: Large output layers require more parameters
- **Computational cost**: Forward and backward passes on large networks are slower
- **Training efficiency**: Learning meaningful Q-values for so many actions may be challenging

For optimal performance, especially on resource-constrained systems:

1. Use efficient network architectures (the default `dueling_dqn` is recommended)
2. Consider increasing batch size (e.g., 64 or 128) and using larger replay buffers
3. Adjust `epsilon_frames` based on the size of your action space (larger action spaces need more exploration frames)
4. Consider adding these parameters to your config files for fine-tuning

In future versions, we may implement more efficient approaches like parameterized DQN or hierarchical RL.

## Updating Training/Inference Scripts

If you are updating existing scripts to use the new DQNAgent interface, follow these guidelines:

### Old Interface

```python
# Old way to initialize the agent
state_dim = calculate_state_dim(env)
agent = DQNAgent(
    state_dim=state_dim, 
    action_space=env.action_space, 
    learning_rate=config['agent']['learning_rate'],
    epsilon_start=config['agent']['epsilon_start'],
    epsilon_min=config['agent']['epsilon_min'],
    epsilon_decay=config['agent']['epsilon_decay'],
    buffer_size=config['agent']['buffer_size']
)
```

### New Interface

```python
# New way to initialize the agent
agent = DQNAgent(
    env=env,  # Pass the environment directly
    action_space=env.action_space,
    memory_size=config['agent']['buffer_size'],
    batch_size=config.get('agent', {}).get('batch_size', 32),
    gamma=config.get('agent', {}).get('gamma', 0.99),
    epsilon_start=config.get('agent', {}).get('epsilon_start', 1.0),
    epsilon_final=config.get('agent', {}).get('epsilon_min', 0.1),  # renamed from epsilon_min
    epsilon_frames=config.get('agent', {}).get('epsilon_frames', 100000),  # new parameter
    target_update_freq=config.get('agent', {}).get('target_update_freq', 1000),
    lr=config.get('agent', {}).get('learning_rate', 0.0001),
    device=config.get('agent', {}).get('device', "cuda" if torch.cuda.is_available() else "cpu"),
    grad_clip_value=config.get('agent', {}).get('grad_clip_value', 1.0),
    model_type=config.get('agent', {}).get('model_type', "dueling_dqn"),
    hidden_dim=config.get('agent', {}).get('hidden_dim', 512)
)
```

### Key Changes

1. **State dimension calculation**: The agent now calculates `state_dim` internally from the environment, so you no longer need to pass it.

2. **Environment object**: Pass the entire environment object, not just its dimensions.

3. **Parameter name changes**:
   - `epsilon_min` → `epsilon_final`
   - `learning_rate` → `lr`
   - `buffer_size` → `memory_size`

4. **New parameters**:
   - `epsilon_frames`: Number of frames over which to anneal epsilon
   - `model_type`: Can be "dueling_dqn" (default), "dqn", "conv_dqn", or "hybrid_dqn"
   - `hidden_dim`: Size of hidden layers in the network
   - `grad_clip_value`: Maximum gradient magnitude for clipping

5. **Removed parameters**:
   - `epsilon_decay`: Replaced with linear annealing over `epsilon_frames`

### Training Loop

The training loop can remain largely the same, but make sure to use the agent's step method:

```python
state, _ = env.reset()
for step in range(total_steps):
    action, next_state, reward, done, info = agent.step(state)
    
    # Optionally update the network after each step or every N steps
    if step % update_frequency == 0:
        loss = agent.update()
    
    # Handle episode termination
    if done:
        state, _ = env.reset()
    else:
        state = next_state
```

### Updating Configuration Files

Add these new parameters to your YAML configuration files (e.g., `config/default.yaml` and `config/a100_gpu.yaml`):

```yaml
agent:
  # Existing parameters
  buffer_size: 100000
  batch_size: 64
  learning_rate: 0.0001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_min: 0.1  # Will be mapped to epsilon_final
  
  # New parameters
  model_type: "dueling_dqn"  # Options: "dueling_dqn", "dqn", "conv_dqn", "hybrid_dqn"
  hidden_dim: 512
  epsilon_frames: 100000  # Number of frames over which to anneal epsilon
  target_update_freq: 1000
  grad_clip_value: 1.0
```

For GPU configurations, you might want to increase batch size and memory:

```yaml
agent:
  # For A100 GPU
  buffer_size: 500000
  batch_size: 256
  hidden_dim: 1024
  # Other parameters as above
```

### Training Loop 