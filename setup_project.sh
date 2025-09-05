#!/bin/bash

# setup_project.sh - Complete setup script for motion-language-control project
# Run this script in your PyCharm project root directory

set -e  # Exit on any error

echo "🚀 Setting up motion-language-control project..."

# Create main directory structure
echo "📁 Creating directory structure..."
directories=(
    "src"
    "src/agents"
    "src/environments"
    "src/models"
    "src/utils"
    "src/training"
    "configs"
    "data"
    "data/mocap"
    "data/instructions"
    "notebooks"
    "scripts"
    "tests"
    "external"
    "logs"
    "checkpoints"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo "  ✓ Created: $dir"
done

# Create __init__.py files for Python packages
echo "🐍 Creating Python package files..."
init_files=(
    "src/__init__.py"
    "src/agents/__init__.py"
    "src/environments/__init__.py"
    "src/models/__init__.py"
    "src/utils/__init__.py"
    "src/training/__init__.py"
    "tests/__init__.py"
)

for init_file in "${init_files[@]}"; do
    touch "$init_file"
    echo "  ✓ Created: $init_file"
done

# Create requirements.txt
echo "📦 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core ML/RL dependencies
torch>=2.0.0
torchvision
transformers>=4.21.0
tokenizers

# Reinforcement Learning
gymnasium[mujoco]>=0.29.0
stable-baselines3[extra]>=2.0.0
sb3-contrib

# MuJoCo and physics simulation
mujoco>=2.3.0
dm-control

# Data processing and utilities
numpy>=1.21.0
scipy
pandas
scikit-learn
matplotlib
seaborn
tqdm
pyyaml
omegaconf

# Experiment tracking and logging
wandb
tensorboard

# Motion and 3D processing (for MotionGPT integration)
trimesh
open3d
opencv-python

# Language processing (for MotionGPT)
spacy
nltk

# Development tools
pytest
black
flake8
jupyter
ipykernel

# Optional: for advanced motion processing
# smplx  # Uncomment if using SMPL body models
# pytorch3d  # Uncomment if using 3D transformations
EOF
echo "  ✓ Created: requirements.txt"

# Create environment.yml for conda
echo "🐍 Creating environment.yml..."
cat > environment.yml << 'EOF'
name: motion-lang-control
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.9
  - pip

  # Core ML frameworks
  - pytorch>=2.0.0
  - torchvision
  - cudatoolkit=11.8  # Adjust based on your GPU

  # Scientific computing
  - numpy>=1.21.0
  - scipy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn

  # MuJoCo and simulation
  - gymnasium
  - tqdm
  - pyyaml

  # Development tools
  - jupyter
  - ipykernel
  - pytest
  - black
  - flake8

  # pip-only dependencies
  - pip:
    - transformers>=4.21.0
    - tokenizers
    - stable-baselines3[extra]>=2.0.0
    - sb3-contrib
    - mujoco>=2.3.0
    - dm-control
    - wandb
    - tensorboard
    - omegaconf
    - trimesh
    - open3d
    - opencv-python
    - spacy
    - nltk
EOF
echo "  ✓ Created: environment.yml"

# Create setup_external.sh script
echo "🔗 Creating setup_external.sh..."
cat > scripts/setup_external.sh << 'EOF'
#!/bin/bash

# Script to set up external repositories for motion-language-control project

echo "Setting up external repositories..."

# Clone AnySkill
echo "Cloning AnySkill..."
cd external
if [ ! -d "anyskill" ]; then
    git clone https://github.com/jiemingcui/anyskill.git anyskill
    echo "  ✓ AnySkill cloned"
else
    echo "  ⚠️ AnySkill already exists, skipping..."
fi

# Clone MotionGPT
echo "Cloning MotionGPT..."
if [ ! -d "motiongpt" ]; then
    git clone https://github.com/OpenMotionLab/MotionGPT.git motiongpt
    echo "  ✓ MotionGPT cloned"
else
    echo "  ⚠️ MotionGPT already exists, skipping..."
fi

cd ..

echo "✅ External repositories setup complete!"
echo "💡 Note: Add external/ to your PYTHONPATH in PyCharm:"
echo "   Settings > Project > Python Interpreter > Show All > Show paths for selected interpreter"
echo "   Add: $(pwd)/external/anyskill"
echo "   Add: $(pwd)/external/motiongpt"
EOF

chmod +x scripts/setup_external.sh
echo "  ✓ Created: scripts/setup_external.sh"

# Create basic config template
echo "⚙️ Creating default config..."
cat > configs/default.yaml << 'EOF'
# Experiment Configuration Template
experiment:
  name: "motion_language_baseline"
  seed: 42

environment:
  name: "Humanoid-v4"
  max_episode_steps: 1000

model:
  motion_tokenizer:
    vocab_size: 512
    hidden_dim: 256
  language_encoder:
    model_name: "t5-small"
    freeze_weights: false

training:
  algorithm: "PPO"
  total_timesteps: 1000000
  learning_rate: 3e-4
  batch_size: 64

evaluation:
  eval_freq: 10000
  n_eval_episodes: 10

logging:
  use_wandb: true
  project_name: "motion-language-control"
  log_dir: "./logs"
EOF
echo "  ✓ Created: configs/default.yaml"

# Create README.md
echo "📝 Creating README.md..."
cat > README.md << 'EOF'
# Motion-Language Control

**Continuous Control from Open-Vocabulary Feedback**

## Overview
This project enables agents in MuJoCo environments to follow natural language instructions
without relying on vision-based processing, by combining MotionGPT with hierarchical RL.

## Setup

### 1. Create Environment
```bash
# Option A: Using conda (recommended)
conda env create -f environment.yml
conda activate motion-lang-control

# Option B: Using pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up External Repositories
```bash
bash scripts/setup_external.sh
```

### 3. Configure PyCharm
- Set interpreter to your conda environment or venv
- Mark `src/` as Sources Root
- Add external repos to PYTHONPATH:
  - `external/anyskill`
  - `external/motiongpt`

## Project Structure
```
motion-language-control/
├── src/                    # Core implementation
│   ├── agents/            # RL agents and policies
│   ├── environments/      # MuJoCo environment wrappers
│   ├── models/            # MotionGPT integration
│   ├── utils/             # Utilities and evaluation
│   └── training/          # Training scripts
├── external/              # External repositories
│   ├── anyskill/         # AnySkill repo
│   └── motiongpt/        # MotionGPT repo
├── configs/              # Experiment configurations
├── data/                 # Datasets and motion capture
├── notebooks/            # Analysis and visualization
└── scripts/              # Setup and utility scripts
```

## Usage
```bash
# Train a basic agent
python src/training/train_agent.py --config configs/default.yaml

# Evaluate trained model
python src/utils/evaluation.py --checkpoint checkpoints/model.pt
```

## References
- [MotionGPT: Human Motion as a Foreign Language](http://arxiv.org/abs/2306.14795) (NeurIPS 2023)
- [AnySkill: Learning Open-Vocabulary Physical Skill for Interactive Agents](https://anyskill.github.io) (CVPR 2024)

## License
[Add your license here]
EOF
echo "  ✓ Created: README.md"

# Create .gitignore
echo "🙈 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyTorch
*.pth
*.pt

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
logs/
checkpoints/
wandb/
data/raw/
*.log

# Large files (add specific patterns as needed)
*.zip
*.tar.gz
*.mp4
*.avi
EOF
echo "  ✓ Created: .gitignore"

# Create a basic starter script
echo "🎯 Creating starter script..."
cat > src/training/train_agent.py << 'EOF'
#!/usr/bin/env python3
"""
Basic training script for motion-language control agents
"""

import argparse
import yaml
from pathlib import Path

def load_config(config_path):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train motion-language control agent')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(f"Experiment: {config['experiment']['name']}")

    # TODO: Initialize environment
    # TODO: Initialize agent
    # TODO: Start training loop

    print("🚀 Training started!")
    print("⚠️  This is a placeholder - implement your training logic here")

if __name__ == "__main__":
    main()
EOF
echo "  ✓ Created: src/training/train_agent.py"

echo ""
echo "🎉 Project setup complete!"
echo ""
echo "Next steps:"
echo "1. 🔗 Set up external repos:    bash scripts/setup_external.sh"
echo "2. 🐍 Create environment:       conda env create -f environment.yml"
echo "3. 🔧 Activate environment:     conda activate motion-lang-control"
echo "4. 📦 Install dependencies:     pip install -r requirements.txt"
echo "5. 🚀 Start coding in src/!"
echo ""
echo "🎯 Project structure created successfully at: $(pwd)"