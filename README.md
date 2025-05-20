# CyberRL: Autonomous Penetration Testing Agent

A reinforcement learning-based agent for automated penetration testing operations. This agent can perform reconnaissance, vulnerability scanning, exploitation attempts, and log findings autonomously.

## Architecture

CyberRL uses a Deep Reinforcement Learning approach with the following components:

1. **Environment**: A simulated or real network environment where the agent performs penetration testing
2. **State Space**: Network topology, discovered hosts, open ports, services, vulnerabilities
3. **Action Space**: Reconnaissance commands, vulnerability scanning, exploitation attempts
4. **Reward Function**: Based on successful reconnaissance, vulnerability discovery, and exploitation
5. **Agent Architecture**: Deep Q-Network (DQN) with experience replay for stable learning

## Technical Requirements

- Python 3.8+
- PyTorch 2.0+
- Gym/Gymnasium for RL environment
- Network scanning tools (Nmap)
- Vulnerability scanning tools (Metasploit framework)
- GPU support: Compatible with A100, L4, T4 GPUs or v5e/v6e TPUs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CyberRL.git
cd CyberRL

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Train the agent
python train.py --config configs/default.yaml

# Run inference with a pre-trained model
python infer.py --model models/pretrained.pt --target 192.168.1.0/24
```

## Project Structure

- `cyberrl/`: Main package
  - `agent/`: RL agent implementation
  - `environment/`: Penetration testing environment
  - `models/`: Neural network models
  - `utils/`: Utility functions
- `configs/`: Configuration files
- `data/`: Training and evaluation data
- `models/`: Saved model checkpoints
- `scripts/`: Helper scripts
- `notebooks/`: Jupyter notebooks for analysis

## License

MIT 