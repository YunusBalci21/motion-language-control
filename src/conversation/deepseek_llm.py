#!/usr/bin/env python3
"""
Real DeepSeek LLM Integration using Transformers
"""

import json
import time
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available, using simulated responses")


@dataclass
class ConversationTurn:
    user_input: str
    llm_response: str
    extracted_actions: List[str]
    timestamp: float
    context: Dict


class DeepSeekLLM:
    """Real DeepSeek LLM using Transformers"""
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
        self.model_name = model_name
        self.conversation_history: List[ConversationTurn] = []
        
        # System prompt for robotics
        self.system_prompt = """You are a helpful robot assistant. When users ask you to perform physical tasks, respond in JSON format with:
{
    "thought_process": "step-by-step reasoning",
    "motion_commands": ["walk forward", "turn left", etc.],
    "estimated_duration": "time in seconds", 
    "safety_notes": "safety considerations"
}

Available motion commands: walk forward, walk backward, turn left, turn right, stop moving, crouch down, stand up, reach forward, grasp object, release object."""

        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading DeepSeek model: {model_name}")
                self.pipe = pipeline(
                    "text-generation", 
                    model=model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                print("DeepSeek model loaded successfully!")
                self.use_real_model = True
            except Exception as e:
                print(f"Failed to load DeepSeek model: {e}")
                print("Falling back to simulated responses")
                self.use_real_model = False
        else:
            self.use_real_model = False
    
    def process_user_request(self, user_input: str, context: Dict = None) -> ConversationTurn:
        """Process request with real DeepSeek or fallback"""
        if context is None:
            context = {}
        
        print(f"\nUser: {user_input}")
        
        if self.use_real_model:
            llm_response = self._call_real_deepseek(user_input, context)
        else:
            llm_response = self._fallback_response(user_input, context)
        
        extracted_actions = self._extract_motion_commands(llm_response)
        
        turn = ConversationTurn(
            user_input=user_input,
            llm_response=llm_response,
            extracted_actions=extracted_actions,
            timestamp=time.time(),
            context=context
        )
        
        self.conversation_history.append(turn)
        
        print(f"DeepSeek: {llm_response}")
        print(f"Extracted Actions: {extracted_actions}")
        
        return turn
    
    def _call_real_deepseek(self, user_input: str, context: Dict) -> str:
        """Call real DeepSeek model"""
        try:
            # Build conversation with system prompt
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Generate response
            response = self.pipe(
                messages,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.pipe.tokenizer.eos_token_id
            )
            
            # Extract response text
            response_text = response[0]['generated_text'][-1]['content']
            
            return response_text
            
        except Exception as e:
            print(f"DeepSeek generation failed: {e}")
            return self._fallback_response(user_input, context)
    
    def _fallback_response(self, user_input: str, context: Dict) -> str:
        """Fallback simulated responses"""
        if "clean" in user_input.lower() and "table" in user_input.lower():
            return json.dumps({
                "thought_process": "To clean the table, I need to: 1) Walk to the table, 2) Pick up objects, 3) Wipe the surface",
                "motion_commands": ["walk forward", "reach forward", "grasp object", "turn left", "release object", "wipe surface"],
                "estimated_duration": "30 seconds",
                "safety_notes": "Be careful with objects"
            })
        elif "walk" in user_input.lower():
            direction = "forward"
            if "left" in user_input.lower():
                direction = "left"
            elif "right" in user_input.lower():
                direction = "right"
            elif "backward" in user_input.lower():
                direction = "backward"
            
            return json.dumps({
                "thought_process": f"User wants me to walk {direction}. This is a basic locomotion task.",
                "motion_commands": [f"walk {direction}"],
                "estimated_duration": "10 seconds",
                "safety_notes": "Maintain balance"
            })
        elif "dance" in user_input.lower():
            return json.dumps({
                "thought_process": "User wants me to dance. I'll create a fun movement sequence.",
                "motion_commands": ["walk forward", "turn left", "turn right", "walk backward"],
                "estimated_duration": "20 seconds",
                "safety_notes": "Smooth movements"
            })
        else:
            return json.dumps({
                "thought_process": "I'll help with a basic movement.",
                "motion_commands": ["walk forward"],
                "estimated_duration": "5 seconds",
                "safety_notes": "Basic movement"
            })
    
    def _extract_motion_commands(self, llm_response: str) -> List[str]:
        """Extract motion commands from response"""
        try:
            if llm_response.startswith('{'):
                response_data = json.loads(llm_response)
                return response_data.get("motion_commands", ["walk forward"])
            else:
                # Try to extract JSON from text
                start = llm_response.find('{')
                end = llm_response.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = llm_response[start:end]
                    response_data = json.loads(json_str)
                    return response_data.get("motion_commands", ["walk forward"])
                else:
                    return ["walk forward"]
        except:
            return ["walk forward"]
    
    def add_feedback(self, feedback: str, execution_result: Dict):
        """Add feedback (for compatibility)"""
        print(f"Feedback received: {feedback}")


# Test function
def test_deepseek():
    print("Testing DeepSeek Integration")
    print("=" * 40)
    
    llm = DeepSeekLLM()
    
    test_queries = [
        "Walk forward please",
        "Can you clean the table?",
        "Turn left and dance",
        "Pick up that object"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        turn = llm.process_user_request(query)
        print(f"Actions: {turn.extracted_actions}")


if __name__ == "__main__":
    test_deepseek()
