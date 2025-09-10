#!/usr/bin/env python3
"""
Conversational Robot Demo
Main entry point for the conversational robotics system

User: "Hey, can you clean the table?"
    ↓
LLM (DeepSeek): [Chain of Thought] "I need to pick up objects, wipe surface..."
    ↓
MotionGPT: Generates action trajectory for cleaning
    ↓
Robot: Executes the cleaning motion
    ↓
User: "I like it" / "I don't like it, too aggressive"
    ↓
System: Learns from feedback, improves next time
"""

import sys
import argparse
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from integration.conversation_robot import ConversationalRobot


def interactive_conversation_mode(robot: ConversationalRobot):
    """Interactive conversation mode with the robot"""

    print(f"\n{'=' * 80}")
    print(" CONVERSATIONAL ROBOT SYSTEM")
    print(f"{'=' * 80}")
    print("Welcome! I'm your conversational robot assistant.")
    print("Tell me what you'd like me to do, and I'll try to help!")
    print()
    print("Examples:")
    print("  - 'Walk forward please'")
    print("  - 'Can you clean the table?'")
    print("  - 'Turn left and then dance'")
    print("  - 'Pick up that object'")
    print()
    print("After I execute actions, give me feedback like:")
    print("  - 'That was perfect!'")
    print("  - 'Too fast, be more gentle'")
    print("  - 'I don't like that movement'")
    print()
    print("Type 'quit' to exit, 'status' for robot info, 'help' for commands")
    print(f"{'=' * 80}")

    conversation_count = 0

    while True:
        try:
            # Get user input
            user_input = input(f"\n[{conversation_count + 1}] You: ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n Robot: Goodbye! Thanks for chatting with me.")
                robot.save_session()
                break

            elif user_input.lower() == 'status':
                status = robot.get_status()
                print("\n Robot Status:")
                for key, value in status.items():
                    if key != "robot_memory":
                        print(f"  {key}: {value}")
                continue

            elif user_input.lower() == 'help':
                print("\n Available Commands:")
                print("  - Ask me to do physical tasks (walk, turn, clean, etc.)")
                print("  - Give feedback after I execute actions")
                print("  - 'status' - Show robot status")
                print("  - 'quit' - Exit the conversation")
                continue

            # Check if this is feedback on previous execution
            if (conversation_count > 0 and
                    any(word in user_input.lower() for word in
                        ['good', 'bad', 'like', 'dislike', 'great', 'terrible', 'fast', 'slow', 'gentle',
                         'aggressive'])):

                print("\n Processing your feedback...")
                feedback_response = robot.process_feedback(user_input)
                print(f" Robot: {feedback_response['robot_response']}")

                if feedback_response.get('improvements_identified'):
                    print(f" I'll work on: {', '.join(feedback_response['improvements_identified'])}")

                continue

            # Process as new task request
            print("\n Processing your request...")
            response = robot.process_user_message(user_input)

            print(f" Robot: {response['robot_response']}")

            # Execute if ready
            if response["ready_to_execute"]:
                print(f"  Estimated duration: {response['estimated_duration']}")

                # Show execution plan
                if response["execution_plan"]:
                    plan = response["execution_plan"]
                    print(f" Plan: {len(plan.motion_steps)} steps")
                    for i, step in enumerate(plan.motion_steps):
                        print(f"  {i + 1}. {step.instruction} ({step.duration} steps)")

                # Ask for execution confirmation
                execute = input("\n  Execute this plan? (y/n): ").strip().lower()

                if execute in ['y', 'yes', '']:
                    print("\n Executing plan...")

                    execution_result = robot.execute_plan(show_live=True)

                    if execution_result["success"]:
                        print(" Execution completed successfully!")
                        print(f" Results: Similarity={execution_result['overall_metrics']['total_similarity']:.3f}, "
                              f"Success={execution_result['overall_metrics']['total_success_rate']:.3f}")
                        print("\n How did I do? Give me feedback!")
                    else:
                        print("  Execution had issues:")
                        print(f"   {execution_result['message']}")
                        print("\n Please give me feedback so I can improve!")
                else:
                    print("  Execution cancelled.")
            else:
                print(" I'm not sure what actions to take. Could you be more specific?")

            conversation_count += 1

        except KeyboardInterrupt:
            print("\n\n Robot: Goodbye! Session interrupted.")
            robot.save_session()
            break
        except Exception as e:
            print(f"\n Error: {e}")
            print("Let's try again!")


def demo_mode(robot: ConversationalRobot):
    """Automated demo mode showing the system capabilities"""

    print(f"\n{'=' * 80}")
    print(" CONVERSATIONAL ROBOT DEMO")
    print(f"{'=' * 80}")
    print("Running automated demo showing conversational interaction...")

    demo_scenarios = [
        {
            "user_message": "Hey robot, can you walk forward?",
            "feedback": "Great job! Very smooth movement.",
            "wait_time": 2
        },
        {
            "user_message": "Now turn left please",
            "feedback": "Too fast and jerky, please be more gentle next time.",
            "wait_time": 2
        },
        {
            "user_message": "Can you clean the table?",
            "feedback": "Good cleaning motion, but maybe a bit slower.",
            "wait_time": 3
        },
        {
            "user_message": "Dance for me!",
            "feedback": "I love the dancing! Very entertaining.",
            "wait_time": 3
        }
    ]

    for i, scenario in enumerate(demo_scenarios):
        print(f"\n{'=' * 60}")
        print(f"DEMO SCENARIO {i + 1}/{len(demo_scenarios)}")
        print(f"{'=' * 60}")

        # User request
        print(f" User: {scenario['user_message']}")
        time.sleep(1)

        # Robot processing
        print(" Robot: (Processing...)")
        response = robot.process_user_message(scenario["user_message"])
        print(f" Robot: {response['robot_response']}")

        # Show plan
        if response["ready_to_execute"]:
            print(f" Execution Plan: {len(response['execution_plan'].motion_steps)} steps")
            print(f"  Duration: {response['estimated_duration']}")

            # Simulate execution
            print(" Executing...")
            time.sleep(scenario["wait_time"])

            # Simulate successful execution
            print(" Execution completed!")

            # Process feedback
            print(f" User: {scenario['feedback']}")
            feedback_response = robot.process_feedback(scenario["feedback"])
            print(f" Robot: {feedback_response['robot_response']}")

            if feedback_response.get('improvements_identified'):
                print(f" Robot will work on: {', '.join(feedback_response['improvements_identified'])}")

        time.sleep(2)

    # Show final status
    print(f"\n{'=' * 60}")
    print("DEMO COMPLETED - FINAL STATUS")
    print(f"{'=' * 60}")

    status = robot.get_status()
    feedback_summary = robot.feedback_system.get_feedback_summary()

    print(f"Total conversations: {status['total_conversations']}")
    print(f"Total executions: {status['total_executions']}")
    print(f"Feedback received: {status['feedback_received']}")
    print(f"User satisfaction: {status['user_satisfaction']}")

    if "success_rate" in feedback_summary:
        print(f"Overall success rate: {feedback_summary['success_rate']}")

    print(f"Tasks completed: {status['robot_memory']['total_tasks_completed']}")


def main():
    parser = argparse.ArgumentParser(
        description="Conversational Robot Demo - Natural Language Robot Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and environment
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained motion model (.zip file)')
    parser.add_argument('--env', type=str, default='Humanoid-v4',
                        help='MuJoCo environment')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')

    # LLM settings
    parser.add_argument('--llm-model', type=str, default='deepseek',
                        choices=['deepseek', 'claude', 'gpt', 'local'],
                        help='LLM model to use')

    # Demo modes
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive conversation mode')
    parser.add_argument('--demo', action='store_true',
                        help='Run automated demo mode')
    parser.add_argument('--test', action='store_true',
                        help='Run system tests')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(" Conversational Robot System")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Device: {device}")
    print(f"LLM Model: {args.llm_model}")
    print(f"Motion Model: {args.model_path if args.model_path else 'None (demo mode)'}")

    # Initialize robot
    robot = ConversationalRobot(
        model_path=args.model_path,
        env_name=args.env,
        device=device,
        llm_model=args.llm_model
    )

    # Check if model is available for real execution
    if args.model_path and not Path(args.model_path).exists():
        print(f"  Warning: Model file not found: {args.model_path}")
        print("   Running in conversation-only mode (no real execution)")
        robot.current_model_path = None

    # Mode selection
    if args.test:
        print("\n Running system tests...")
        from integration.conversation_robot import test_conversational_robot
        test_conversational_robot()

    elif args.demo:
        demo_mode(robot)

    elif args.interactive or not any([args.demo, args.test]):
        # Default to interactive mode
        interactive_conversation_mode(robot)

    print("\n Session ended. Thank you for using Conversational Robot!")


if __name__ == "__main__":
    main()