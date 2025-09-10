#!/usr/bin/env python3
"""
LLM Interface for Conversational Robotics
Handles communication with DeepSeek, Claude, or other LLMs
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConversationTurn:
    """Single conversation turn"""
    user_input: str
    llm_response: str
    extracted_actions: List[str]
    timestamp: float
    context: Dict


class LLMInterface:
    """Interface for LLM communication with chain-of-thought reasoning"""
    
    def __init__(self, model_name: str = "deepseek", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.conversation_history: List[ConversationTurn] = []
        
        print(f"LLM Interface initialized with model: {model_name}")
    
    def process_user_request(self, user_input: str, context: Dict = None) -> ConversationTurn:
        """Process user request and extract motion commands"""
        if context is None:
            context = {}
            
        print(f"\nUser: {user_input}")
        
        # Simulate LLM processing (replace with actual API call)
        llm_response = self._call_llm(user_input, context)
        
        # Extract motion commands from LLM response
        extracted_actions = self._extract_motion_commands(llm_response)
        
        # Create conversation turn
        turn = ConversationTurn(
            user_input=user_input,
            llm_response=llm_response,
            extracted_actions=extracted_actions,
            timestamp=time.time(),
            context=context
        )
        
        self.conversation_history.append(turn)
        
        print(f"LLM: {llm_response}")
        print(f"Extracted Actions: {extracted_actions}")
        
        return turn
    
    def _call_llm(self, user_input: str, context: Dict) -> str:
        """Call LLM API (simulated for now)"""
        
        # Simulate different responses based on input
        if "clean" in user_input.lower() and "table" in user_input.lower():
            return json.dumps({
                "thought_process": "To clean the table, I need to: 1) Walk to the table, 2) Reach forward to access the surface, 3) Pick up any objects, 4) Wipe the surface clean",
                "motion_commands": ["walk forward", "reach forward", "grasp object", "turn left", "release object", "reach forward", "wipe surface"],
                "estimated_duration": "30 seconds",
                "safety_notes": "Ensure objects are securely grasped before moving"
            })
        
        elif "walk" in user_input.lower():
            direction = "forward"
            if "backward" in user_input.lower() or "back" in user_input.lower():
                direction = "backward"
            elif "left" in user_input.lower():
                direction = "left"
            elif "right" in user_input.lower():
                direction = "right"
                
            return json.dumps({
                "thought_process": f"User wants me to walk {direction}. This is a simple locomotion task.",
                "motion_commands": [f"walk {direction}"],
                "estimated_duration": "10 seconds",
                "safety_notes": "Maintain balance while walking"
            })
        
        elif "dance" in user_input.lower():
            return json.dumps({
                "thought_process": "User wants me to dance. I'll create a sequence of movements that are rhythmic and expressive.",
                "motion_commands": ["walk forward", "turn left", "turn right", "walk backward"],
                "estimated_duration": "20 seconds",
                "safety_notes": "Ensure smooth transitions between movements"
            })
        
        else:
            # Generic response
            return json.dumps({
                "thought_process": "I'll interpret this as a general movement request and try to be helpful.",
                "motion_commands": ["walk forward"],
                "estimated_duration": "5 seconds",
                "safety_notes": "Proceeding with basic movement"
            })
    
    def _extract_motion_commands(self, llm_response: str) -> List[str]:
        """Extract motion commands from LLM JSON response"""
        try:
            response_data = json.loads(llm_response)
            return response_data.get("motion_commands", [])
        except json.JSONDecodeError:
            print("Error parsing LLM response, extracting commands manually")
            return ["walk forward"]
    
    def add_feedback(self, feedback: str, execution_result: Dict):
        """Add feedback to the last conversation turn"""
        if self.conversation_history:
            last_turn = self.conversation_history[-1]
            last_turn.context["feedback"] = feedback
            last_turn.context["execution_result"] = execution_result
            print(f"Feedback added: {feedback}")
