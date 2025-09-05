# Motion-Language Control - ISA Project

**Continuous Control from Open-Vocabulary Feedback**

## Overview
This project enables agents in MuJoCo environments to follow natural language instructions
without relying on vision-based processing, by combining MotionGPT with hierarchical RL.


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
