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
