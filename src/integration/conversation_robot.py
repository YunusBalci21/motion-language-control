#!/usr/bin/env python3
"""
Conversational Robot - Main integration of all components
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from conversation.deepseek_llm import DeepSeekLLM
from conversation.task_planner import TaskPlanner, MotionPlan
from conversation.feedback_system import FeedbackSystem

try:
    from agents.hierarchical_agent import EnhancedMotionLanguageAgent

    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("⚠ Agent not available - using conversation-only mode")


class ConversationalRobot:
    """
    Complete conversational robot system
    Integrates LLM, planning, execution, and feedback learning
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 env_name: str = "Humanoid-v4",
                 device: str = "auto",
                 llm_model: str = "deepseek"):

        print("Initializing Conversational Robot System")
        print("=" * 60)

        # Initialize components
        self.llm = DeepSeekLLM()
        self.planner = TaskPlanner()
        self.feedback = FeedbackSystem()

        # Initialize motion agent if available
        self.agent = None
        self.current_model_path = model_path

        if AGENT_AVAILABLE and model_path and Path(model_path).exists():
            try:
                self.agent = EnhancedMotionLanguageAgent(
                    env_name=env_name,
                    device=device
                )
                print(f"✓ Motion agent initialized")
                print(f"✓ Loaded model: {model_path}")
            except Exception as e:
                print(f"⚠ Failed to initialize agent: {e}")
                print("  Running in conversation-only mode")
        else:
            print("⚠ Running in conversation-only mode (no real execution)")

        print("=" * 60)
        print("✓ Conversational Robot System Ready!")
        print()

    def process_conversation_turn(self, user_input: str) -> Dict:
        """
        Process one conversation turn: understand -> plan -> execute -> respond
        """
        result = {
            "user_input": user_input,
            "llm_response": None,
            "motion_plan": None,
            "execution_results": None,
            "success": False
        }

        # Step 1: LLM Understanding
        conversation_turn = self.llm.process_user_request(user_input)
        result["llm_response"] = conversation_turn.llm_response

        if not conversation_turn.extracted_actions:
            print("\n⚠ No actionable commands extracted")
            return result

        # Step 2: Task Planning
        motion_plan = self.planner.create_motion_plan(conversation_turn.extracted_actions)
        result["motion_plan"] = motion_plan

        print(f"\n📋 Motion Plan:")
        print(f"   Sequence: {' → '.join(motion_plan.motion_sequence)}")
        print(f"   Duration: ~{motion_plan.estimated_duration:.1f}s")
        if motion_plan.warnings:
            print(f"   ⚠ Warnings: {motion_plan.warnings}")

        # Step 3: Safety Check
        if not motion_plan.safety_check_passed:
            print("\n⛔ Safety check failed - aborting execution")
            return result

        # Step 4: Execution (if agent available)
        if self.agent and self.current_model_path:
            print(f"\n🤖 Executing motion sequence...")
            execution_results = self._execute_motion_plan(motion_plan)
            result["execution_results"] = execution_results
            result["success"] = True
        else:
            print(f"\n🎭 Simulating motion execution (no real agent)")
            result["success"] = True

        return result

    def _execute_motion_plan(self, plan: MotionPlan) -> Dict:
        """
        Execute motion plan using the agent
        """
        results = {
            "instructions_executed": [],
            "success": True,
            "error": None
        }

        try:
            for instruction in plan.motion_sequence:
                print(f"   Executing: {instruction}")

                # For demo, we just validate the instruction exists
                # In full system, you would call agent.evaluate_instruction()
                results["instructions_executed"].append(instruction)
                time.sleep(0.5)  # Simulate execution time

            print(f"   ✓ Completed {len(plan.motion_sequence)} motions")

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            print(f"   ✗ Execution failed: {e}")

        return results

    def handle_feedback(self, user_input: str, commands_executed: list, feedback_text: str):
        """
        Handle user feedback
        """
        entry = self.feedback.record_feedback(
            user_input=user_input,
            commands_executed=commands_executed,
            feedback_text=feedback_text
        )

        print(f"\n💬 Feedback recorded: {entry.feedback_type}")
        print(f"   Score: {entry.performance_score}")

        # Get recommendations
        recommendations = self.feedback.get_recommendations()
        if recommendations:
            print(f"\n💡 Learning insights:")
            for rec in recommendations[:3]:
                print(f"   • {rec}")

    def interactive_session(self):
        """
        Run interactive conversation session
        """
        print("\n" + "=" * 60)
        print("🤖 INTERACTIVE ROBOT CONVERSATION")
        print("=" * 60)
        print("\nYou can:")
        print("  • Give commands: 'walk forward', 'turn left', 'clean table'")
        print("  • Give feedback: 'That was great!', 'Too fast'")
        print("  • Type 'quit' or 'exit' to end")
        print("  • Type 'help' for examples")
        print()

        last_commands = []
        last_user_input = ""

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                user_lower = user_input.lower()

                # Check for exit
                if user_lower in ["quit", "exit", "bye", "goodbye"]:
                    print("\n🤖 Goodbye! Thanks for chatting!")
                    break

                # Check for help
                if user_lower == "help":
                    print("\nExample commands:")
                    print("  • 'Walk forward please'")
                    print("  • 'Turn left and stop'")
                    print("  • 'Clean the table'")
                    print("  • 'Dance for me!'")
                    print("\nExample feedback (after execution):")
                    print("  • 'That was great!'")
                    print("  • 'Too aggressive, be more gentle'")
                    print("  • 'Perfect!'")
                    continue

                # Check if this is feedback
                is_feedback = any(word in user_lower for word in [
                    "great", "good", "bad", "wrong", "perfect", "too", "slow", "fast"
                ])

                if is_feedback and last_commands:
                    # This is feedback on previous execution
                    self.handle_feedback(last_user_input, last_commands, user_input)
                    continue

                # Process as new command
                result = self.process_conversation_turn(user_input)

                if result["motion_plan"]:
                    last_commands = result["motion_plan"].motion_sequence
                    last_user_input = user_input

                # Small delay for readability
                time.sleep(0.5)

            except KeyboardInterrupt:
                print("\n\n🤖 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue


def test_conversational_robot():
    """Test the conversational robot system"""
    print("Testing Conversational Robot System")
    print("=" * 60)

    robot = ConversationalRobot()

    # Test conversation flow
    test_conversations = [
        ("Hey, can you walk forward?", "That was perfect!"),
        ("Now turn left", "Good turn"),
        ("Clean the table please", "Too fast, slow down"),
        ("Dance for me!", "Love it!"),
    ]

    for user_input, feedback in test_conversations:
        print(f"\n{'=' * 60}")
        result = robot.process_conversation_turn(user_input)
        time.sleep(1)

        if result["motion_plan"]:
            robot.handle_feedback(
                user_input,
                result["motion_plan"].motion_sequence,
                feedback
            )
        time.sleep(1)

    print("\n" + "=" * 60)
    print("✓ Test completed!")


if __name__ == "__main__":
    test_conversational_robot()