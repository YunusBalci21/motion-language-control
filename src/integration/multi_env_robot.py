#!/usr/bin/env python3
"""
Multi-Environment Conversational Robot
Supports Humanoid, HalfCheetah, Ant, and manipulation environments
"""

import sys
from pathlib import Path
from typing import Optional, Dict

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from conversation.deepseek_llm import DeepSeekLLM
from conversation.task_planner import TaskPlanner
from conversation.feedback_system import FeedbackSystem

try:
    from agents.hierarchical_agent import EnhancedMotionLanguageAgent

    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


class MultiEnvRobot:
    """
    Multi-environment conversational robot
    """

    # Available environments and their capabilities
    ENVIRONMENTS = {
        "humanoid": {
            "name": "Humanoid-v4",
            "capabilities": ["walking", "waving", "dancing", "turning", "crouching"],
            "description": "Humanoid robot (walking, waving, dancing)"
        },
        "halfcheetah": {
            "name": "HalfCheetah-v4",
            "capabilities": ["running", "jumping", "walking"],
            "description": "Half-Cheetah (running, jumping)"
        },
        "ant": {
            "name": "Ant-v4",
            "capabilities": ["walking", "navigation", "turning"],
            "description": "Ant (navigation tasks)"
        },
        "walker": {
            "name": "Walker2d-v4",
            "capabilities": ["walking", "running"],
            "description": "Walker2d (bipedal locomotion)"
        }
    }

    def __init__(self, device: str = "auto"):
        self.device = device
        self.current_env = None
        self.agents: Dict[str, Optional[EnhancedMotionLanguageAgent]] = {}

        # Shared components
        self.llm = DeepSeekLLM()
        self.planner = TaskPlanner()
        self.feedback = FeedbackSystem()

        print("🤖 Multi-Environment Robot System Initialized")

    def list_environments(self):
        """Display available environments"""
        print("\n" + "=" * 60)
        print("AVAILABLE ENVIRONMENTS")
        print("=" * 60)
        for i, (key, info) in enumerate(self.ENVIRONMENTS.items(), 1):
            print(f"{i}. {info['description']}")
            print(f"   Capabilities: {', '.join(info['capabilities'])}")
        print("=" * 60)

    def select_environment(self, choice: str) -> bool:
        """Select environment by number or name"""
        # Try by number
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(self.ENVIRONMENTS):
                env_key = list(self.ENVIRONMENTS.keys())[idx]
                return self._load_environment(env_key)

        # Try by name
        choice_lower = choice.lower()
        for key in self.ENVIRONMENTS.keys():
            if key in choice_lower or choice_lower in key:
                return self._load_environment(key)

        print(f"⚠ Unknown environment: {choice}")
        return False

    def _load_environment(self, env_key: str) -> bool:
        """Load specific environment"""
        if env_key not in self.ENVIRONMENTS:
            return False

        env_info = self.ENVIRONMENTS[env_key]
        self.current_env = env_key

        print(f"\n🔄 Loading {env_info['description']}...")

        # Create agent if not exists
        if env_key not in self.agents:
            if AGENT_AVAILABLE:
                try:
                    agent = EnhancedMotionLanguageAgent(
                        env_name=env_info["name"],
                        device=self.device,
                        use_stability_focus=True
                    )
                    self.agents[env_key] = agent
                    print(f"✓ {env_info['name']} agent created")
                except Exception as e:
                    print(f"⚠ Failed to create agent: {e}")
                    self.agents[env_key] = None
            else:
                self.agents[env_key] = None

        print(f"✓ Active environment: {env_info['description']}")
        print(f"   Capabilities: {', '.join(env_info['capabilities'])}")

        return True

    def get_current_capabilities(self):
        """Get capabilities of current environment"""
        if self.current_env:
            return self.ENVIRONMENTS[self.current_env]["capabilities"]
        return []

    def process_command(self, user_input: str) -> Dict:
        """Process command with current environment"""
        if not self.current_env:
            print("\n⚠ No environment selected. Use 'switch [env]' to select one.")
            return {"success": False, "error": "No environment selected"}

        # Process with LLM
        conversation_turn = self.llm.process_user_request(user_input)

        if not conversation_turn.extracted_actions:
            return {"success": False, "error": "No commands extracted"}

        # Create motion plan
        motion_plan = self.planner.create_motion_plan(conversation_turn.extracted_actions)

        print(f"\n📋 Motion Plan:")
        print(f"   Environment: {self.ENVIRONMENTS[self.current_env]['description']}")
        print(f"   Sequence: {' → '.join(motion_plan.motion_sequence)}")
        print(f"   Duration: ~{motion_plan.estimated_duration:.1f}s")

        if motion_plan.warnings:
            print(f"   ⚠ Warnings: {motion_plan.warnings}")

        # Execute if agent available
        agent = self.agents.get(self.current_env)
        if agent:
            print(f"\n🤖 Executing on {self.current_env}...")
            # Here you would call agent.evaluate_instruction() for each motion
            print(f"   ✓ Simulated execution")
        else:
            print(f"\n🎭 Simulating execution (no real agent)")

        return {
            "success": True,
            "motion_plan": motion_plan,
            "conversation_turn": conversation_turn
        }

    def interactive_session(self):
        """Interactive session with environment switching"""
        print("\n" + "=" * 60)
        print("🤖 MULTI-ENVIRONMENT INTERACTIVE ROBOT")
        print("=" * 60)
        print("\nCommands:")
        print("  • 'list' - Show available environments")
        print("  • 'switch [env]' - Switch environment (e.g., 'switch humanoid')")
        print("  • 'status' - Show current environment")
        print("  • Give robot commands: 'walk forward', 'turn left'")
        print("  • Give feedback: 'That was great!', 'Too fast'")
        print("  • 'help' - Show examples")
        print("  • 'quit' - Exit")
        print()

        # Start with environment selection
        self.list_environments()
        print("\nSelect an environment (1-4):")
        choice = input("Choice: ").strip()

        if not self.select_environment(choice):
            print("⚠ Invalid choice, defaulting to Humanoid")
            self.select_environment("humanoid")

        last_commands = []
        last_user_input = ""

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                user_lower = user_input.lower()

                # System commands
                if user_lower in ["quit", "exit", "bye"]:
                    print("\n🤖 Goodbye!")
                    break

                elif user_lower == "list":
                    self.list_environments()
                    continue

                elif user_lower.startswith("switch "):
                    env_name = user_lower.replace("switch ", "").strip()
                    self.select_environment(env_name)
                    continue

                elif user_lower == "status":
                    if self.current_env:
                        env_info = self.ENVIRONMENTS[self.current_env]
                        print(f"\n📍 Current: {env_info['description']}")
                        print(f"   Capabilities: {', '.join(env_info['capabilities'])}")
                    else:
                        print("\n⚠ No environment selected")
                    continue

                elif user_lower == "help":
                    print("\nExamples:")
                    if self.current_env:
                        caps = self.get_current_capabilities()
                        print(f"  For {self.ENVIRONMENTS[self.current_env]['description']}:")
                        if "walking" in caps:
                            print("    • 'Walk forward please'")
                        if "running" in caps:
                            print("    • 'Run fast'")
                        if "jumping" in caps:
                            print("    • 'Jump up'")
                        if "dancing" in caps:
                            print("    • 'Dance for me'")
                    else:
                        print("  Select an environment first with 'switch [env]'")
                    continue

                # Check if feedback
                is_feedback = any(word in user_lower for word in [
                    "great", "good", "bad", "wrong", "perfect", "too", "slow", "fast"
                ])

                if is_feedback and last_commands:
                    entry = self.feedback.record_feedback(
                        last_user_input, last_commands, user_input
                    )
                    print(f"\n💬 Feedback: {entry.feedback_type} ({entry.performance_score:+.1f})")
                    continue

                # Process as robot command
                result = self.process_command(user_input)

                if result["success"] and result.get("motion_plan"):
                    last_commands = result["motion_plan"].motion_sequence
                    last_user_input = user_input

            except KeyboardInterrupt:
                print("\n\n🤖 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Environment Robot")
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("🤖 Multi-Environment Conversational Robot")
    print(f"Device: {device}")

    robot = MultiEnvRobot(device=device)
    robot.interactive_session()


if __name__ == "__main__":
    main()