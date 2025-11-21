#!/usr/bin/env python3
"""
Conversational Robot Demo - Main entry point
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from integration.conversation_robot import ConversationalRobot


def demo_mode(robot: ConversationalRobot):
    """Run automated demo"""
    print("\n🎬 AUTOMATED DEMO MODE")
    print("=" * 60)

    demo_script = [
        ("Walk forward please", "That was great!"),
        ("Turn left", "Perfect turn"),
        ("Now walk forward again", "A bit too fast"),
        ("Clean the table", "Good cleaning motion"),
        ("Stop moving", None),
    ]

    import time
    for user_input, feedback in demo_script:
        print(f"\n{'=' * 60}")
        print(f"Demo User: {user_input}")
        time.sleep(1)

        result = robot.process_conversation_turn(user_input)
        time.sleep(2)

        if feedback and result["motion_plan"]:
            print(f"\nDemo User: {feedback}")
            robot.handle_feedback(
                user_input,
                result["motion_plan"].motion_sequence,
                feedback
            )
            time.sleep(1)

    print("\n" + "=" * 60)
    print("✓ Demo completed!")


def interactive_conversation_mode(robot: ConversationalRobot):
    """Run interactive conversation"""
    robot.interactive_session()


def main():
    parser = argparse.ArgumentParser(
        description="Conversational Robot Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained motion model')
    parser.add_argument('--env', type=str, default='Humanoid-v4',
                        help='MuJoCo environment')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--llm-model', type=str, default='deepseek',
                        help='LLM model')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive mode')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo mode')
    parser.add_argument('--test', action='store_true',
                        help='Run tests')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("🤖 Conversational Robot System")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model_path if args.model_path else 'None'}")
    print()

    # Initialize
    robot = ConversationalRobot(
        model_path=args.model_path,
        env_name=args.env,
        device=device,
        llm_model=args.llm_model
    )

    # Mode selection
    if args.test:
        from integration.conversation_robot import test_conversational_robot
        test_conversational_robot()
    elif args.demo:
        demo_mode(robot)
    else:
        # Default to interactive
        interactive_conversation_mode(robot)


if __name__ == "__main__":
    main()