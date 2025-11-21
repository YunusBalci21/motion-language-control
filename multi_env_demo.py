#!/usr/bin/env python3
"""
Multi-Environment Demo - Entry point
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from integration.multi_env_robot import MultiEnvRobot


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    robot = MultiEnvRobot(device=device)
    robot.interactive_session()


if __name__ == "__main__":
    main()