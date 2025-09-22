#!/usr/bin/env python3
"""
Real DeepSeek LLM Integration using Transformers - FIXED with timeout
"""

import json
import time
import torch
import signal
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


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("LLM generation timed out")


class DeepSeekLLM:
    """Real DeepSeek LLM using Transformers with timeout fallback"""

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", timeout_seconds: int = 10):
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
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
                print("⚠️  Note: Large model may be slow. Using 10-second timeout.")

                # Try to load with timeout
                self.pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_new_tokens=100,  # Limit tokens for speed
                    do_sample=True,
                    temperature=0.3  # Lower temperature for more consistent responses
                )
                print("DeepSeek model loaded successfully!")
                self.use_real_model = True
            except Exception as e:
                print(f"Failed to load DeepSeek model: {e}")
                print("🎭 Using fast simulated responses for better user experience")
                self.use_real_model = False
        else:
            self.use_real_model = False

    def process_user_request(self, user_input: str, context: Dict = None) -> ConversationTurn:
        """Process request with DeepSeek or fast fallback"""
        if context is None:
            context = {}

        print(f"\nUser: {user_input}")

        if self.use_real_model:
            llm_response = self._call_real_deepseek_with_timeout(user_input, context)
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

    def _call_real_deepseek_with_timeout(self, user_input: str, context: Dict) -> str:
        """Call real DeepSeek model with timeout fallback"""
        try:
            print("🧠 Calling DeepSeek model (with timeout)...")

            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)

            try:
                # Build conversation with system prompt
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ]

                # Generate response
                response = self.pipe(
                    messages,
                    max_new_tokens=50,  # Keep it short for speed
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.pipe.tokenizer.eos_token_id
                )

                # Extract response text
                response_text = response[0]['generated_text'][-1]['content']

                # Cancel timeout
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

                print("✅ DeepSeek responded successfully!")
                return response_text

            except TimeoutError:
                print(f"⏰ DeepSeek timed out after {self.timeout_seconds}s, using fast fallback")
                return self._fallback_response(user_input, context)

        except Exception as e:
            print(f"❌ DeepSeek generation failed: {e}")
            print("🎭 Using fast fallback response")
            return self._fallback_response(user_input, context)
        finally:
            # Always restore signal handler
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            except:
                pass

    def _fallback_response(self, user_input: str, context: Dict) -> str:
        """Fast fallback simulated responses"""
        user_lower = user_input.lower()

        if "clean" in user_lower and "table" in user_lower:
            return json.dumps({
                "thought_process": "To clean the table, I need to: 1) Walk to the table, 2) Pick up objects, 3) Wipe the surface clean",
                "motion_commands": ["walk forward", "reach forward", "grasp object", "turn left", "release object",
                                    "wipe surface"],
                "estimated_duration": "30 seconds",
                "safety_notes": "Be careful with objects"
            })
        elif "walk" in user_lower:
            direction = "forward"
            if "left" in user_lower:
                direction = "left"
            elif "right" in user_lower:
                direction = "right"
            elif "backward" in user_lower:
                direction = "backward"

            return json.dumps({
                "thought_process": f"User wants me to walk {direction}. This is a basic locomotion task.",
                "motion_commands": [f"walk {direction}"],
                "estimated_duration": "10 seconds",
                "safety_notes": "Maintain balance"
            })
        elif "turn" in user_lower:
            direction = "left" if "left" in user_lower else "right" if "right" in user_lower else "left"
            return json.dumps({
                "thought_process": f"User wants me to turn {direction}. I need to rotate my body.",
                "motion_commands": [f"turn {direction}"],
                "estimated_duration": "8 seconds",
                "safety_notes": "Turn smoothly"
            })
        elif "dance" in user_lower:
            return json.dumps({
                "thought_process": "User wants me to dance. I'll create a fun movement sequence.",
                "motion_commands": ["walk forward", "turn left", "turn right", "walk backward"],
                "estimated_duration": "20 seconds",
                "safety_notes": "Smooth movements"
            })
        elif "pick up" in user_lower or "grasp" in user_lower:
            return json.dumps({
                "thought_process": "User wants me to pick up an object. I need to approach and grasp it carefully.",
                "motion_commands": ["walk forward", "reach forward", "grasp object"],
                "estimated_duration": "15 seconds",
                "safety_notes": "Gentle grip on objects"
            })
        elif "organize" in user_lower or "help" in user_lower:
            return json.dumps({
                "thought_process": "User wants help organizing. I'll walk around and pick up objects systematically.",
                "motion_commands": ["walk forward", "turn left", "reach forward", "grasp object", "turn right",
                                    "release object"],
                "estimated_duration": "45 seconds",
                "safety_notes": "Careful with fragile items"
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
    print("Testing Fixed DeepSeek Integration")
    print("=" * 40)

    llm = DeepSeekLLM()

    test_queries = [
        "Walk forward please",
        "Can you clean the table?",
        "Turn left and dance",
        "Pick up that object"
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        start_time = time.time()
        turn = llm.process_user_request(query)
        end_time = time.time()
        print(f"Response time: {end_time - start_time:.2f}s")
        print(f"Actions: {turn.extracted_actions}")


if __name__ == "__main__":
    test_deepseek()