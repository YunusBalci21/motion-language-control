#!/usr/bin/env python3
"""
Conversational Robot System
Integrates LLM, Task Planning, Motion Execution, and Feedback Learning
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

# Import conversation components
from conversation.llm_interface import LLMInterface, ConversationTurn
from conversation.task_planner import TaskPlanner, ExecutionPlan
from conversation.feedback_system import FeedbackSystem

# Import your existing motion system
from agents.hierarchical_agent import EnhancedMotionLanguageAgent, DirectMotionLanguageWrapper


class ConversationalRobot:
    """Main conversational robot system that integrates all components"""

    def __init__(self,
                 model_path: Optional[str] = None,
                 env_name: str = "Humanoid-v4",
                 device: str = "cuda",
                 llm_model: str = "deepseek"):

        print("Initializing Conversational Robot System")
        print("=" * 50)

        # Initialize all components
        self.llm_interface = LLMInterface(model_name=llm_model)
        self.task_planner = TaskPlanner()
        self.feedback_system = FeedbackSystem()

        # Initialize motion system (your existing code)
        self.motion_agent = EnhancedMotionLanguageAgent(env_name=env_name, device=device)
        self.current_model_path = model_path

        # State tracking
        self.current_conversation: Optional[ConversationTurn] = None
        self.current_plan: Optional[ExecutionPlan] = None
        self.execution_history: List[Dict] = []
        self.is_executing = False

        # Robot "memory"
        self.robot_memory = {
            "user_name": "Human",
            "session_start": time.time(),
            "total_tasks_completed": 0,
            "favorite_tasks": [],
            "recent_context": []
        }

        print("✓ LLM Interface ready")
        print("✓ Task Planner ready")
        print("✓ Feedback System ready")
        print("✓ Motion System ready")
        print(f"✓ Model loaded: {model_path if model_path else 'None (will need training)'}")
        print("\nConversational Robot initialized successfully!")

    def process_user_message(self, user_input: str) -> Dict:
        """Process user message and return response with execution plan"""

        print(f"\n{'=' * 60}")
        print(f"USER: {user_input}")
        print(f"{'=' * 60}")

        # Get conversation context
        context = self._build_conversation_context()

        # Process with LLM
        conversation_turn = self.llm_interface.process_user_request(user_input, context)
        self.current_conversation = conversation_turn

        # Create execution plan
        if conversation_turn.extracted_actions:
            plan = self.task_planner.create_execution_plan(
                conversation_turn.extracted_actions,
                task_description=user_input
            )
            self.current_plan = plan

            # Get personalized recommendations
            recommendations = self.feedback_system.get_personalized_recommendations(
                conversation_turn.extracted_actions
            )

            # Prepare response
            response = {
                "robot_response": self._generate_robot_response(conversation_turn, plan, recommendations),
                "execution_plan": plan,
                "recommendations": recommendations,
                "ready_to_execute": True,
                "estimated_duration": f"{plan.total_duration} steps (~{plan.total_duration / 100:.1f} seconds)"
            }
        else:
            response = {
                "robot_response": "I understand, but I'm not sure what actions to take. Could you be more specific?",
                "execution_plan": None,
                "recommendations": None,
                "ready_to_execute": False,
                "estimated_duration": "N/A"
            }

        # Update robot memory
        self._update_robot_memory(user_input, response)

        return response

    def execute_plan(self, max_steps: int = 1000, show_live: bool = True) -> Dict:
        """Execute the current plan using your motion system"""

        if not self.current_plan or not self.current_model_path:
            return {
                "success": False,
                "message": "No execution plan or trained model available",
                "execution_results": {}
            }

        print(f"\n{'=' * 60}")
        print("EXECUTING PLAN")
        print(f"{'=' * 60}")

        self.is_executing = True
        execution_results = {}

        try:
            # Execute each step in the plan
            for i, motion_step in enumerate(self.current_plan.motion_steps):
                print(f"\nStep {i + 1}/{len(self.current_plan.motion_steps)}: {motion_step.instruction}")

                # Execute single instruction using your existing system
                step_result = self._execute_single_instruction(
                    motion_step.instruction,
                    int(motion_step.duration),
                    show_live=show_live
                )

                execution_results[f"step_{i + 1}"] = {
                    "instruction": motion_step.instruction,
                    "planned_duration": motion_step.duration,
                    "actual_results": step_result
                }

                # Small pause between steps
                time.sleep(0.5)

            # Aggregate results
            total_similarity = np.mean([
                result["actual_results"].get("mean_similarity", 0)
                for result in execution_results.values()
            ])

            total_success_rate = np.mean([
                result["actual_results"].get("episode_success_rate", 0)
                for result in execution_results.values()
            ])

            success = total_similarity > 0.3 and total_success_rate > 0.2

            final_result = {
                "success": success,
                "message": "Execution completed successfully!" if success else "Execution had issues, but completed.",
                "execution_results": execution_results,
                "overall_metrics": {
                    "total_similarity": total_similarity,
                    "total_success_rate": total_success_rate,
                    "steps_completed": len(execution_results),
                    "total_duration": sum(r["planned_duration"] for r in execution_results.values())
                }
            }

            # Update robot memory
            if success:
                self.robot_memory["total_tasks_completed"] += 1
                if self.current_conversation:
                    task = self.current_conversation.user_input
                    if task not in self.robot_memory["favorite_tasks"]:
                        self.robot_memory["favorite_tasks"].append(task)

            # Add to execution history
            self.execution_history.append({
                "task": self.current_conversation.user_input if self.current_conversation else "Unknown",
                "plan": self.current_plan.task_description,
                "results": final_result,
                "timestamp": time.time()
            })

            return final_result

        except Exception as e:
            print(f"Execution error: {e}")
            return {
                "success": False,
                "message": f"Execution failed: {str(e)}",
                "execution_results": execution_results
            }
        finally:
            self.is_executing = False

    def _execute_single_instruction(self, instruction: str, duration: int, show_live: bool = True) -> Dict:
        """Execute a single instruction using your existing motion system"""

        try:
            # Use your existing evaluation system
            results = self.motion_agent.evaluate_instruction(
                instruction=instruction,
                model_path=self.current_model_path,
                num_episodes=1,  # Single episode for real-time execution
                language_reward_weight=0.7,
                deterministic=True,
                render=show_live,
                record_video=False  # Can enable if needed
            )

            return results

        except Exception as e:
            print(f"Error executing instruction '{instruction}': {e}")
            return {
                "mean_similarity": 0.0,
                "episode_success_rate": 0.0,
                "mean_total_reward": 0.0,
                "error": str(e)
            }

    def process_feedback(self, feedback: str) -> Dict:
        """Process user feedback on the last execution"""

        if not self.current_conversation or not self.execution_history:
            return {"message": "No recent execution to provide feedback on"}

        print(f"\n{'=' * 60}")
        print(f"FEEDBACK: {feedback}")
        print(f"{'=' * 60}")

        # Get last execution info
        last_execution = self.execution_history[-1]
        motion_sequence = [step.instruction for step in self.current_plan.motion_steps] if self.current_plan else []

        # Process feedback
        feedback_entry = self.feedback_system.process_feedback(
            task_description=last_execution["task"],
            motion_sequence=motion_sequence,
            user_feedback=feedback,
            execution_metrics=last_execution["results"].get("overall_metrics", {})
        )

        # Update LLM with feedback
        self.llm_interface.add_feedback(feedback, last_execution["results"])

        # Generate response
        if feedback_entry.sentiment == "positive":
            robot_response = "Thank you! I'm glad you liked it. I'll remember your preferences for next time."
        elif feedback_entry.sentiment == "negative":
            suggestions = ", ".join(feedback_entry.improvement_suggestions)
            robot_response = f"I apologize that wasn't what you wanted. I'll work on: {suggestions}"
        else:
            robot_response = "Thank you for the feedback. I'll keep learning to improve."

        return {
            "robot_response": robot_response,
            "feedback_processed": True,
            "sentiment": feedback_entry.sentiment,
            "improvements_identified": feedback_entry.improvement_suggestions,
            "updated_preferences": self.feedback_system.user_preferences.__dict__
        }

    def _build_conversation_context(self) -> Dict:
        """Build context for conversation from robot memory and feedback"""
        context = {
            "robot_memory": self.robot_memory,
            "feedback_summary": self.feedback_system.get_feedback_summary(),
            "recent_executions": self.execution_history[-3:] if self.execution_history else [],
            "user_preferences": self.feedback_system.user_preferences.__dict__
        }

        return context

    def _generate_robot_response(self, conversation_turn: ConversationTurn,
                                 plan: ExecutionPlan, recommendations: Dict) -> str:
        """Generate natural robot response"""

        # Parse LLM response for thought process
        try:
            import json
            llm_data = json.loads(conversation_turn.llm_response)
            thought = llm_data.get("thought_process", "I understand what you want me to do.")
        except:
            thought = "I understand what you want me to do."

        # Build response
        response_parts = []

        # Acknowledge understanding
        response_parts.append(f"I understand! {thought}")

        # Describe plan
        if len(plan.motion_steps) == 1:
            response_parts.append(f"I'll {plan.motion_steps[0].instruction}.")
        else:
            actions = [step.instruction for step in plan.motion_steps[:3]]  # First 3 actions
            actions_str = ", ".join(actions)
            if len(plan.motion_steps) > 3:
                actions_str += f", and {len(plan.motion_steps) - 3} more steps"
            response_parts.append(f"My plan is to: {actions_str}.")

        # Add personalization
        if recommendations["warnings"]:
            response_parts.append("I've adjusted the plan based on your preferences.")

        # Add duration estimate
        estimated_time = plan.total_duration / 100  # Convert steps to approximate seconds
        response_parts.append(f"This should take about {estimated_time:.1f} seconds.")

        # Add safety note if needed
        if plan.safety_level in ["high", "very_high"]:
            response_parts.append("I'll be extra careful with these movements.")

        return " ".join(response_parts)

    def _update_robot_memory(self, user_input: str, response: Dict):
        """Update robot memory with recent interaction"""
        self.robot_memory["recent_context"].append({
            "user_input": user_input,
            "robot_response": response.get("robot_response", ""),
            "timestamp": time.time(),
            "had_plan": response.get("ready_to_execute", False)
        })

        # Keep only last 10 interactions
        if len(self.robot_memory["recent_context"]) > 10:
            self.robot_memory["recent_context"] = self.robot_memory["recent_context"][-10:]

    def get_status(self) -> Dict:
        """Get current robot status"""
        return {
            "is_executing": self.is_executing,
            "has_current_plan": self.current_plan is not None,
            "model_loaded": self.current_model_path is not None,
            "total_conversations": len(self.llm_interface.conversation_history),
            "total_executions": len(self.execution_history),
            "feedback_received": len(self.feedback_system.feedback_history),
            "user_satisfaction": f"{self.feedback_system.user_preferences.success_rate:.1%}",
            "robot_memory": self.robot_memory
        }

    def save_session(self, filename: str = None):
        """Save current session data"""
        if filename is None:
            filename = f"robot_session_{int(time.time())}.json"

        session_data = {
            "robot_memory": self.robot_memory,
            "execution_history": self.execution_history,
            "model_path": self.current_model_path,
            "session_end": time.time()
        }

        import json
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        print(f"Session saved to {filename}")


def test_conversational_robot():
    """Test the conversational robot system"""
    print("Testing Conversational Robot System")
    print("=" * 50)

    # Initialize robot (without model for testing)
    robot = ConversationalRobot(
        model_path=None,  # No model for testing
        env_name="Humanoid-v4"
    )

    # Test conversation flow
    test_messages = [
        "Hey robot, can you walk forward?",
        "That was good, but can you turn left now?",
        "Clean the table please",
        "Dance for me!"
    ]

    for message in test_messages:
        print(f"\n{'=' * 70}")

        # Process message
        response = robot.process_user_message(message)
        print(f"ROBOT: {response['robot_response']}")

        if response["ready_to_execute"]:
            print(f"Plan: {len(response['execution_plan'].motion_steps)} steps")
            print(f"Duration: {response['estimated_duration']}")

            # Simulate execution (without actual model)
            print("(Simulating execution since no model loaded)")

            # Simulate feedback
            if "walk forward" in message:
                feedback_response = robot.process_feedback("Great! Very smooth movement.")
            elif "turn" in message:
                feedback_response = robot.process_feedback("Too fast, please be more gentle.")
            else:
                feedback_response = robot.process_feedback("I liked that!")

            print(f"FEEDBACK RESPONSE: {feedback_response['robot_response']}")

    # Show final status
    print(f"\n{'=' * 70}")
    print("Final Status:")
    status = robot.get_status()
    for key, value in status.items():
        if key != "robot_memory":
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_conversational_robot()