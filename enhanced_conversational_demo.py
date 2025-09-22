#!/usr/bin/env python3
"""
Enhanced Conversational Robot Demo - COMPLETE FIXED VERSION
Implements the exact flow with immediate retry and stability improvements:
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
System: "Would you like me to try again with improvements?" → Immediate retry
"""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import Optional

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from conversation.deepseek_llm import DeepSeekLLM
from conversation.task_planner import TaskPlanner
from conversation.feedback_system import FeedbackSystem
from agents.hierarchical_agent import EnhancedMotionLanguageAgent
from models.motion_tokenizer import MotionTokenizer


class EnhancedConversationalRobot:
    """Enhanced conversational robot with immediate retry and stability improvements"""

    def __init__(self,
                 model_path: Optional[str] = None,
                 env_name: str = "Humanoid-v4",
                 device: str = "cuda"):

        print(f"\n{'=' * 80}")
        print(" ENHANCED CONVERSATIONAL ROBOT SYSTEM")
        print(f"{'=' * 80}")
        print("Implementing the complete conversational flow:")
        print("User Request → DeepSeek LLM → MotionGPT → Robot Execution → User Feedback → Learning")
        print("NEW: Immediate retry with improvements for negative feedback")
        print(f"{'=' * 80}")

        # Initialize all components
        print("🤖 Initializing DeepSeek LLM for chain-of-thought reasoning...")
        self.deepseek_llm = DeepSeekLLM()

        print("📋 Initializing Task Planner for motion sequencing...")
        self.task_planner = TaskPlanner()

        print("🧠 Initializing Feedback System for learning...")
        self.feedback_system = FeedbackSystem()

        print("🤖 Initializing MotionGPT Tokenizer...")
        self.motion_tokenizer = MotionTokenizer(device=device)

        print("🦾 Initializing Motion Execution Agent...")
        self.motion_agent = EnhancedMotionLanguageAgent(env_name=env_name, device=device)

        # System state
        self.current_model_path = model_path
        self.current_plan = None  # Store current plan for retry
        self.conversation_history = []
        self.execution_history = []
        self.learning_progress = {
            "total_conversations": 0,
            "successful_executions": 0,
            "user_satisfaction_score": 0.5,
            "learned_preferences": {}
        }

        print("✅ Enhanced Conversational Robot initialized successfully!")
        print(f"Model loaded: {model_path if model_path else 'None (demo mode)'}")
        print(f"Environment: {env_name}")
        print(f"Device: {device}")

    def process_user_request(self, user_input: str) -> dict:
        """
        Step 1: Process user request with DeepSeek LLM chain-of-thought reasoning
        """
        print(f"\n{'🗣️  USER REQUEST':<20} | {user_input}")
        print(f"{'=' * 80}")

        # DeepSeek LLM Processing with Chain-of-Thought
        print("🧠 DeepSeek LLM: Analyzing request with chain-of-thought reasoning...")

        conversation_turn = self.deepseek_llm.process_user_request(user_input)

        # Extract reasoning from LLM response
        try:
            if conversation_turn.llm_response.startswith('{'):
                llm_data = json.loads(conversation_turn.llm_response)
                chain_of_thought = llm_data.get("thought_process", "I understand what you want me to do.")
                motion_commands = llm_data.get("motion_commands", conversation_turn.extracted_actions)
                estimated_duration = llm_data.get("estimated_duration", "Unknown")
                safety_notes = llm_data.get("safety_notes", "Standard safety precautions")
            else:
                chain_of_thought = "I understand what you want me to do."
                motion_commands = conversation_turn.extracted_actions
                estimated_duration = "Unknown"
                safety_notes = "Standard safety precautions"
        except:
            chain_of_thought = "I understand what you want me to do."
            motion_commands = conversation_turn.extracted_actions
            estimated_duration = "Unknown"
            safety_notes = "Standard safety precautions"

        print(f"🤔 Chain-of-Thought: {chain_of_thought}")
        print(f"⚡ Extracted Actions: {motion_commands}")
        print(f"⏱️  Estimated Duration: {estimated_duration}")
        print(f"🛡️  Safety Notes: {safety_notes}")

        return {
            "conversation_turn": conversation_turn,
            "chain_of_thought": chain_of_thought,
            "motion_commands": motion_commands,
            "estimated_duration": estimated_duration,
            "safety_notes": safety_notes,
            "ready_for_planning": len(motion_commands) > 0
        }

    def generate_motion_trajectory(self, motion_commands: list, task_description: str) -> dict:
        """
        Step 2: MotionGPT generates action trajectory for the task
        """
        print(f"\n{'🎯 MOTIONGPT PLANNING':<20} | Generating action trajectory...")
        print(f"{'=' * 80}")

        # Task Planning (converts LLM commands to detailed motion sequence)
        print("📋 Task Planner: Converting commands to motion sequence...")
        execution_plan = self.task_planner.create_execution_plan(motion_commands, task_description)

        print(f"✅ Motion Trajectory Generated:")
        print(f"   📝 Task: {execution_plan.task_description}")
        print(f"   🎬 Motion Steps: {len(execution_plan.motion_steps)}")
        print(f"   ⏱️  Total Duration: {execution_plan.total_duration} steps")
        print(f"   🛡️  Safety Level: {execution_plan.safety_level}")

        # Show motion sequence
        print(f"\n📋 Detailed Motion Trajectory:")
        for i, step in enumerate(execution_plan.motion_steps[:5]):  # Show first 5 steps
            print(f"   {i + 1}. {step.instruction} ({step.duration} steps, priority: {step.priority:.2f})")

        if len(execution_plan.motion_steps) > 5:
            print(f"   ... and {len(execution_plan.motion_steps) - 5} more steps")

        # Apply user preferences from learning
        personalized_plan = self._apply_learned_preferences(execution_plan)

        # Store current plan for potential retry
        self.current_plan = personalized_plan

        return {
            "execution_plan": personalized_plan,
            "trajectory_generated": True,
            "motion_quality_preview": self._preview_motion_quality(personalized_plan)
        }

    def execute_robot_motion(self, execution_plan, show_live: bool = True) -> dict:
        """
        Step 3: Robot executes the motion with stability improvements
        """
        print(f"\n{'🤖 ROBOT EXECUTION':<20} | Executing motion trajectory...")
        print(f"{'=' * 80}")

        if not self.current_model_path:
            # Demo mode - simulate execution
            print("🎭 DEMO MODE: Simulating robot execution...")
            return self._simulate_robot_execution(execution_plan)

        # Real execution mode with stability improvements
        print("🦾 REAL EXECUTION: Running trained motion model...")

        execution_results = {}
        total_similarity = 0.0
        total_success_rate = 0.0

        try:
            # Execute each step in the plan
            for i, motion_step in enumerate(execution_plan.motion_steps):
                print(f"🎬 Step {i + 1}/{len(execution_plan.motion_steps)}: {motion_step.instruction}")

                # Execute single instruction using the motion system with stability focus
                step_result = self.motion_agent.evaluate_instruction(
                    instruction=motion_step.instruction,
                    model_path=self.current_model_path,
                    num_episodes=1,
                    language_reward_weight=0.7,
                    deterministic=True,
                    render=show_live,
                    record_video=False
                )

                execution_results[f"step_{i + 1}"] = {
                    "instruction": motion_step.instruction,
                    "similarity": step_result.get("mean_similarity", 0.0),
                    "success_rate": step_result.get("episode_success_rate", 0.0),
                    "reward": step_result.get("mean_total_reward", 0.0),
                    "quality": step_result.get("mean_motion_overall_quality", 0.0)
                }

                print(f"   ✅ Similarity: {step_result.get('mean_similarity', 0.0):.3f}")
                print(f"   🎯 Success: {step_result.get('episode_success_rate', 0.0):.3f}")
                print(f"   🏆 Quality: {step_result.get('mean_motion_overall_quality', 0.0):.3f}")

                total_similarity += step_result.get("mean_similarity", 0.0)
                total_success_rate += step_result.get("episode_success_rate", 0.0)

                time.sleep(0.5)  # Brief pause between steps

            # Calculate overall performance
            avg_similarity = total_similarity / len(execution_plan.motion_steps)
            avg_success_rate = total_success_rate / len(execution_plan.motion_steps)
            execution_successful = avg_similarity > 0.4 and avg_success_rate > 0.3

            print(f"\n🏁 EXECUTION COMPLETED!")
            print(f"   📊 Average Similarity: {avg_similarity:.3f}")
            print(f"   🎯 Average Success Rate: {avg_success_rate:.3f}")
            print(f"   ✅ Overall Success: {'YES' if execution_successful else 'PARTIAL'}")

            return {
                "execution_successful": execution_successful,
                "average_similarity": avg_similarity,
                "average_success_rate": avg_success_rate,
                "step_results": execution_results,
                "total_steps": len(execution_plan.motion_steps)
            }

        except Exception as e:
            print(f"❌ Execution error: {e}")
            return {
                "execution_successful": False,
                "error": str(e),
                "step_results": execution_results
            }

    def process_user_feedback_with_retry(self, feedback: str, execution_result: dict,
                                         original_request: str, motion_commands: list) -> dict:
        """
        ENHANCED: Process user feedback with immediate retry option
        """
        print(f"\n{'💬 USER FEEDBACK':<20} | {feedback}")
        print(f"{'=' * 80}")

        # Process feedback with the feedback system
        print("🧠 Processing feedback and updating preferences...")

        feedback_entry = self.feedback_system.process_feedback(
            task_description=original_request,
            motion_sequence=motion_commands,
            user_feedback=feedback,
            execution_metrics=execution_result
        )

        print(f"📊 Feedback Analysis:")
        print(f"   😊 Sentiment: {feedback_entry.sentiment}")
        print(f"   📈 Improvements: {feedback_entry.improvement_suggestions}")

        # Update learning progress
        self._update_learning_progress(feedback_entry, execution_result)

        # Generate personalized response
        robot_response = self._generate_learning_response(feedback_entry)
        print(f"🤖 Robot Learning Response: {robot_response}")

        # NEW: Offer immediate retry for negative feedback
        should_retry = False
        retry_result = None

        if feedback_entry.sentiment == "negative":
            print(f"\n🔄 IMMEDIATE IMPROVEMENT OPTION")
            retry_input = input("🤖 Would you like me to try that again with improvements? (y/n): ").strip().lower()

            if retry_input in ['y', 'yes', '']:
                should_retry = True
                print("🔄 Retrying with improved parameters...")

                # Apply improvements immediately
                improved_plan = self._apply_immediate_improvements(
                    self.current_plan,
                    feedback_entry.improvement_suggestions
                )

                # Re-execute with improvements
                retry_result = self.execute_robot_motion(improved_plan, show_live=True)

                print(f"\n🔄 RETRY COMPLETED!")
                print(f"   📊 Retry Similarity: {retry_result.get('average_similarity', 0):.3f}")
                print(f"   🎯 Retry Success: {retry_result.get('average_success_rate', 0):.3f}")

                # Ask for feedback on the retry
                retry_feedback = input("\n💭 How was the retry? (or press Enter to continue): ").strip()
                if retry_feedback:
                    print(f"🤖 Thank you! I'll remember: {retry_feedback}")
                    # Process retry feedback too
                    self.feedback_system.process_feedback(
                        task_description=f"{original_request} (retry)",
                        motion_sequence=motion_commands,
                        user_feedback=retry_feedback,
                        execution_metrics=retry_result
                    )

        # Get updated recommendations for future
        future_recommendations = self.feedback_system.get_personalized_recommendations(motion_commands)

        print(f"🔮 Future Improvements:")
        for improvement in future_recommendations.get("suggested_improvements", []):
            print(f"   • {improvement}")

        return {
            "feedback_processed": True,
            "sentiment": feedback_entry.sentiment,
            "improvements": feedback_entry.improvement_suggestions,
            "robot_response": robot_response,
            "retry_attempted": should_retry,
            "retry_result": retry_result,
            "future_recommendations": future_recommendations,
            "learning_progress": self.learning_progress
        }

    def _apply_immediate_improvements(self, execution_plan, improvements):
        """Apply improvements immediately to the execution plan"""
        import copy
        improved_plan = copy.deepcopy(execution_plan)

        for improvement in improvements:
            if "reduce_movement_speed" in improvement:
                # Make movements slower and more gentle
                for step in improved_plan.motion_steps:
                    step.duration *= 1.5  # 50% slower
                print("   🐌 Applied: Slower, gentler movements")

            elif "increase_movement_speed" in improvement:
                # Make movements faster
                for step in improved_plan.motion_steps:
                    step.duration *= 0.8  # 20% faster
                print("   ⚡ Applied: Faster movements")

            elif "adjust_style_to_gentle" in improvement:
                # Force gentle/safe execution
                improved_plan.safety_level = "high"
                print("   🤲 Applied: Gentler style")

        return improved_plan

    def _simulate_robot_execution(self, execution_plan) -> dict:
        """Simulate robot execution for demo mode with realistic metrics"""
        print("🎭 Simulating robot motion execution...")

        total_similarity = 0.0
        total_success = 0.0

        for i, step in enumerate(execution_plan.motion_steps):
            print(f"🎬 Step {i + 1}: {step.instruction}")
            # Use motion tokenizer to simulate execution
            try:
                # Create mock motion sequence for this step
                import numpy as np
                mock_motion = np.random.randn(20, 30) * 0.1  # 20 timesteps, 30 features

                # Compute simulated performance
                similarity = self.motion_tokenizer.compute_motion_language_similarity(
                    mock_motion, step.instruction
                )
                success_rate = self.motion_tokenizer.compute_success_rate(
                    mock_motion, step.instruction
                )

                print(f"   ✅ Simulated similarity: {similarity:.3f}")
                print(f"   🎯 Simulated success: {success_rate:.3f}")

                total_similarity += similarity
                total_success += success_rate

                time.sleep(1)  # Simulate execution time

            except Exception as e:
                print(f"   ⚠️ Simulation error: {e}")
                total_similarity += 0.5
                total_success += 0.5

        # Return simulated results
        avg_similarity = total_similarity / len(execution_plan.motion_steps)
        avg_success_rate = total_success / len(execution_plan.motion_steps)

        return {
            "execution_successful": True,
            "average_similarity": max(0.1, min(1.0, avg_similarity)),
            "average_success_rate": max(0.1, min(1.0, avg_success_rate)),
            "step_results": {},
            "total_steps": len(execution_plan.motion_steps),
            "simulation_mode": True
        }

    def _apply_learned_preferences(self, execution_plan):
        """Apply learned user preferences to the execution plan"""
        # Get current user preferences
        preferences = self.feedback_system.user_preferences

        # Modify plan based on learned preferences
        if preferences.preferred_speed == "slow":
            for step in execution_plan.motion_steps:
                step.duration *= 1.3  # Increase duration for slower movement
        elif preferences.preferred_speed == "fast":
            for step in execution_plan.motion_steps:
                step.duration *= 0.8  # Decrease duration for faster movement

        # Apply style preferences
        if preferences.preferred_style == "gentle":
            execution_plan.safety_level = "high"

        return execution_plan

    def _preview_motion_quality(self, execution_plan) -> dict:
        """Preview expected motion quality"""
        # Simple preview based on plan complexity and user preferences
        complexity_score = len(execution_plan.motion_steps) / 10.0
        safety_score = {"low": 0.9, "medium": 0.7, "high": 0.5, "very_high": 0.3}.get(
            execution_plan.safety_level, 0.7
        )

        return {
            "expected_smoothness": max(0.2, 0.8 - complexity_score),
            "expected_safety": safety_score,
            "expected_success_rate": max(0.3, 0.7 - complexity_score * 0.2)
        }

    def _update_learning_progress(self, feedback_entry, execution_results):
        """Update the system's learning progress"""
        self.learning_progress["total_conversations"] += 1

        if execution_results.get("execution_successful", False):
            self.learning_progress["successful_executions"] += 1

        # Update satisfaction score
        if feedback_entry.sentiment == "positive":
            self.learning_progress["user_satisfaction_score"] = min(1.0,
                                                                    self.learning_progress[
                                                                        "user_satisfaction_score"] + 0.1)
        elif feedback_entry.sentiment == "negative":
            self.learning_progress["user_satisfaction_score"] = max(0.0,
                                                                    self.learning_progress[
                                                                        "user_satisfaction_score"] - 0.1)

    def _generate_learning_response(self, feedback_entry) -> str:
        """Generate a learning response based on feedback"""
        if feedback_entry.sentiment == "positive":
            responses = [
                "Thank you! I'm glad you liked my performance. I'll remember what worked well.",
                "Great to hear! I'm learning what you prefer and will continue to improve.",
                "Wonderful! I'll make note of this successful approach for future tasks."
            ]
        elif feedback_entry.sentiment == "negative":
            improvements = ", ".join(feedback_entry.improvement_suggestions[:2])
            responses = [
                f"I apologize for not meeting your expectations. I'll work on: {improvements}.",
                f"Thank you for the feedback. I'm learning to be better at: {improvements}.",
                f"I understand. Let me improve by focusing on: {improvements}."
            ]
        else:
            responses = [
                "Thank you for the feedback. I'm continuously learning to serve you better.",
                "I appreciate your input. Every piece of feedback helps me improve."
            ]

        import random
        return random.choice(responses)

    def run_conversation_session(self):
        """Run the complete conversational session with retry functionality"""
        print(f"\n{'=' * 80}")
        print(" CONVERSATIONAL ROBOT SESSION STARTED")
        print(f"{'=' * 80}")
        print("🗣️  Tell me what you'd like me to do!")
        print("Examples: 'Hey, can you clean the table?', 'Walk forward please', 'Dance for me!'")
        print("Type 'quit' to exit, 'status' for system status")
        print("NEW: I'll offer to retry immediately if you're not satisfied!")
        print(f"{'=' * 80}")

        conversation_count = 0

        while True:
            try:
                # Get user input
                user_input = input(f"\n[{conversation_count + 1}] 🗣️  You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n🤖 Goodbye! Thanks for the great conversation!")
                    break

                elif user_input.lower() == 'status':
                    self._show_system_status()
                    continue

                # THE COMPLETE CONVERSATIONAL FLOW
                print(f"\n{'🔄 STARTING CONVERSATIONAL FLOW':<40}")

                # Step 1: User Request → DeepSeek LLM Chain-of-Thought
                request_result = self.process_user_request(user_input)

                if not request_result["ready_for_planning"]:
                    print("🤖 I'm not sure what actions to take. Could you be more specific?")
                    continue

                # Step 2: MotionGPT Trajectory Generation
                trajectory_result = self.generate_motion_trajectory(
                    request_result["motion_commands"],
                    user_input
                )

                # Ask for execution confirmation
                print(f"\n{'🤔 EXECUTION CONFIRMATION':<20}")
                execute = input("🚀 Execute this motion plan? (y/n): ").strip().lower()

                if execute not in ['y', 'yes', '']:
                    print("❌ Execution cancelled.")
                    continue

                # Step 3: Robot Execution
                execution_result = self.execute_robot_motion(
                    trajectory_result["execution_plan"],
                    show_live=True
                )

                # Step 4: Request User Feedback
                print(f"\n{'💭 FEEDBACK REQUEST':<20}")
                print("🤖 How did I do? Please give me feedback!")
                print("Examples: 'Great job!', 'Too fast, be more gentle', 'Perfect!'")

                feedback = input("💬 Your feedback: ").strip()

                if feedback:
                    # Step 5: Process Feedback and Learn (WITH RETRY OPTION)
                    feedback_result = self.process_user_feedback_with_retry(
                        feedback,
                        execution_result,
                        user_input,
                        request_result["motion_commands"]
                    )

                    if feedback_result.get("retry_attempted"):
                        print(f"\n🎓 Learning completed with immediate improvement!")
                    else:
                        print(f"\n🎓 Learning completed! System improved for next time.")
                else:
                    print("👍 No feedback provided, but I'm still learning!")

                conversation_count += 1
                print(f"\n{'✅ CONVERSATION FLOW COMPLETED':<40}")

            except KeyboardInterrupt:
                print("\n\n🤖 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Let's try again!")

    def _show_system_status(self):
        """Show current system status"""
        print(f"\n{'🔍 SYSTEM STATUS':<20}")
        print(f"{'=' * 60}")
        print(f"💬 Total conversations: {self.learning_progress['total_conversations']}")
        print(f"✅ Successful executions: {self.learning_progress['successful_executions']}")
        print(f"😊 User satisfaction: {self.learning_progress['user_satisfaction_score']:.1%}")
        print(f"🤖 Model loaded: {'Yes' if self.current_model_path else 'Demo mode'}")

        # Show learned preferences
        prefs = self.feedback_system.user_preferences
        print(f"🎯 Learned preferences:")
        print(f"   Speed: {prefs.preferred_speed}")
        print(f"   Style: {prefs.preferred_style}")
        print(f"   Safety sensitivity: {prefs.safety_sensitivity:.2f}")
        print(f"   Feedback count: {prefs.feedback_count}")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Conversational Robot Demo with Immediate Retry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained motion model (.zip file)')
    parser.add_argument('--env', type=str, default='Humanoid-v4',
                        help='MuJoCo environment')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--demo-mode', action='store_true',
                        help='Run in demo mode without trained model')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("🚀 Enhanced Conversational Robot Demo")
    print("=" * 50)
    print(f"Environment: {args.env}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path if args.model_path else 'Demo mode'}")
    print("NEW: Immediate retry functionality enabled!")

    # Initialize the enhanced robot
    robot = EnhancedConversationalRobot(
        model_path=args.model_path,
        env_name=args.env,
        device=device
    )

    # Check model availability
    if args.model_path and not Path(args.model_path).exists():
        print(f"⚠️  Warning: Model file not found: {args.model_path}")
        print("🎭 Running in demo mode (simulated execution)")
        robot.current_model_path = None

    # Run the conversational session
    robot.run_conversation_session()


if __name__ == "__main__":
    main()