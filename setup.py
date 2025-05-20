#!/usr/bin/env python3
"""
Setup script for CyberRL
"""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README for long description
with open('README.md') as f:
    long_description = f.read()

setup(
    name="cyberrl",
    version="0.1.0",
    description="Reinforcement Learning for Automated Penetration Testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CyberRL Team",
    author_email="info@cyberrl.ai",
    url="https://github.com/cyberrl/cyberrl",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "cyberrl-train=cyberrl.scripts.train:main",
            "cyberrl-infer=cyberrl.scripts.infer:main",
        ],
    },
) 