#!/usr/bin/env python3
"""
Training script entry point for CyberRL
"""

import sys
import os

# Add the parent directory to sys.path to allow running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the main function from the training script
from train import main

if __name__ == "__main__":
    main() 