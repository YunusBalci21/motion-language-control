#!/usr/bin/env python3
"""
Conversational Robot with Real DeepSeek Integration
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

# Import conversation components
from conversation.deepseek_llm import DeepSeekLLM, ConversationTurn
from conversation.task_planner import TaskPlanner, ExecutionPlan
from conversation.feedback_system import FeedbackSystem

# Import your existing motion system
from agents.hierarchical_agent import EnhancedMotionLanguageAgent


class ConversationalRobotWithDeepSeek:
    """Conversational robot with real DeepSeek LLM"""
    
    def __init__(self, model_path: Optional[str] = None, env_name: str = "Humanoid-v4", device: str = "cuda"):
        print("Initializing Conversational Robot with DeepSeek")
        print("=" * 50)
        
        # Initialize components
        self.llm_interface = DeepSeekLLM()
        self.task_planner = TaskPlanner()
        self.feedback_system = FeedbackSystem()
        self.motion_agent = EnhancedMotionLanguageAgent(env_name=env_name, device=device)
        self.current_model_path = model_path
        
        # State tracking
        self.current_conversation: Optional[ConversationTurn] = None
        self.current_plan: Optional[ExecutionPlan] = None
        self.execution_history: List[Dict] = []
        
        print("✓ DeepSeek LLM ready")
        print("✓ Task Planner ready")
        print("✓ Feedback System ready") 
        print("✓ Motion System ready")
        print("\nConversational Robot with DeepSeek initialized!")
    
    def process_user_message(self, user_input: str) -> Dict:
        """Process user message with DeepSeek"""
        print(f"\n{'=' * 60}")
        print(f"USER: {user_input}")
        print(f"{'=' * 60}")
        
        # Process with DeepSeek
        conversation_turn = self.llm_interface.process_user_request(user_input)
        self.current_conversation = conversation_turn
        
        # Create execution plan
        if conversation_turn.extracted_actions:
            plan = self.task_planner.create_execution_plan(
                conversation_turn.extracted_actions,
                task_description=user_input
            )
            self.current_plan = plan
            
            response = {
                "robot_response": self._generate_robot_response(conversation_turn, plan),
                "execution_plan": plan,
                "ready_to_execute": True,
                "estimated_duration": f"{plan.total_duration} steps (~{plan.total_duration / 100:.1f} seconds)"
            }
        else:
            response = {
                "robot_response": "I understand, but I'm not sure what actions to take. Could you be more specific?",
                "execution_plan": None,
                "ready_to_execute": False,
                "estimated_duration": "N/A"
            }
        
        return response
    
    def _generate_robot_response(self, conversation_turn: ConversationTurn, plan: ExecutionPlan) -> str:
        """Generate robot response from DeepSeek output"""
        try:
            import json
            llm_data = json.loads(conversation_turn.llm_response)
            thought = llm_data.get("thought_process", "I understand what you want me to do.")
        except:
            thought = "I understand what you want me to do."
        
        response_parts = [f"I understand! {thought}"]
        
        if len(plan.motion_steps) == 1:
            response_parts.append(f"I'll {plan.motion_steps[0].instruction}.")
        else:
            actions = [step.instruction for step in plan.motion_steps[:3]]
            actions_str = ", ".join(actions)
            response_parts.append(f"My plan: {actions_str}.")
        
        estimated_time = plan.total_duration / 100
        response_parts.append(f"This should take about {estimated_time:.1f} seconds.")
        
        return " ".join(response_parts)
    
    def process_feedback(self, feedback: str) -> Dict:
        """Process feedback with proper response"""
        if any(word in feedback.lower() for word in ['good', 'great', 'perfect', 'smooth']):
            sentiment = "positive"
            response = "Thank you! I'm glad you liked it. I'll remember your preferences."
        elif any(word in feedback.lower() for word in ['bad', 'terrible', 'fast', 'jerky']):
            sentiment = "negative"
            response = "I apologize. I'll work on being more gentle and smooth."
        else:
            sentiment = "neutral"
            response = "Thank you for the feedback. I'll keep learning."
        
        return {
            "robot_response": response,
            "feedback_processed": True,
            "sentiment": sentiment,
            "improvements_identified": ["smoother_movements"] if sentiment == "negative" else []
        }


def test_deepseek_robot():
    """Test the DeepSeek robot"""
    print("Testing DeepSeek Conversational Robot")
    print("=" * 50)
    
    robot = ConversationalRobotWithDeepSeek()
    
    test_messages = [
        "Walk forward please",
        "Can you clean the table?", 
        "Turn left and dance"
    ]
    
    for message in test_messages:
        print(f"\n{'='*60}")
        response = robot.process_user_message(message)
        print(f"ROBOT: {response['robot_response']}")
        
        if response["ready_to_execute"]:
            print(f"Plan: {len(response['execution_plan'].motion_steps)} steps")
            
            # Simulate feedback
            feedback_response = robot.process_feedback("That was great!")
            print(f"FEEDBACK: {feedback_response['robot_response']}")


if __name__ == "__main__":
    test_deepseek_robot()
