#!/usr/bin/env python3
"""
Real DeepSeek LLM Integration using HuggingFace Transformers
"""

import json
import time
import re
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

try:
    from transformers import pipeline, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠ Transformers not available, install with: pip install transformers")


@dataclass
class ConversationTurn:
    user_input: str
    llm_response: str
    chain_of_thought: str
    extracted_actions: List[str]
    timestamp: float
    context: Dict

    def to_dict(self):
        return asdict(self)


class DeepSeekLLM:
    """
    DeepSeek LLM Interface using HuggingFace Transformers
    """

    def __init__(self,
                 model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_real_model: bool = True):

        self.model_name = model_name
        self.device = device
        self.use_real_model = use_real_model and TRANSFORMERS_AVAILABLE
        self.conversation_history: List[ConversationTurn] = []
        self.pipeline = None
        self.tokenizer = None

        # System prompt for robot control
        self.system_prompt = """You are a helpful robot assistant that controls physical robots. 

When users ask you to perform tasks:
1. Show your chain of thought reasoning using <think></think> tags
2. Break down the task into simple motion commands
3. Use ONLY these available commands:
   - walk forward [slowly/quickly]
   - walk backward
   - turn left
   - turn right  
   - stop moving
   - stand still
   - crouch down
   - reach forward
   - pick up object
   - place object
   - wave hand

Respond in this format:
<think>
Your step-by-step reasoning here
</think>

Commands: command1, command2, command3

Be safe and clear. If unsure, ask for clarification."""

        # Initialize model
        if self.use_real_model:
            self._initialize_model()
        else:
            print("⚠ Using simulated responses")

    def _initialize_model(self):
        """Initialize DeepSeek model from HuggingFace"""
        try:
            print(f"\n🧠 Loading DeepSeek model: {self.model_name}")
            print(f"   Device: {self.device}")
            print("   This may take a minute...")

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Initialize pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                tokenizer=self.tokenizer,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            print("✅ DeepSeek model loaded successfully!")
            self.use_real_model = True

        except Exception as e:
            print(f"❌ Failed to load DeepSeek model: {e}")
            print("   Falling back to simulated responses...")
            self.use_real_model = False
            self.pipeline = None

    def process_user_request(self, user_input: str, context: Dict = None) -> ConversationTurn:
        """
        Process user request with DeepSeek
        """
        if context is None:
            context = {}

        print(f"\n{'=' * 60}")
        print(f"User: {user_input}")
        print(f"{'=' * 60}")

        # Get LLM response
        if self.use_real_model and self.pipeline:
            llm_response = self._call_deepseek(user_input, context)
        else:
            llm_response = self._simulated_response(user_input, context)

        # Extract chain of thought
        chain_of_thought = self._extract_chain_of_thought(llm_response)

        # Extract motion commands
        extracted_actions = self._extract_motion_commands(llm_response)

        # Create conversation turn
        turn = ConversationTurn(
            user_input=user_input,
            llm_response=llm_response,
            chain_of_thought=chain_of_thought,
            extracted_actions=extracted_actions,
            timestamp=time.time(),
            context=context
        )

        self.conversation_history.append(turn)

        # Display results
        print(f"\n🤖 DeepSeek Response:")
        print(f"{chain_of_thought}")
        print(f"\n📝 Extracted Commands: {extracted_actions}")

        return turn

    def _call_deepseek(self, user_input: str, context: Dict) -> str:
        """Call real DeepSeek model with simpler prompting"""
        try:
            # Much simpler prompt - DeepSeek-R1 works better with direct instructions
            full_prompt = f"""You are controlling a robot. The user says: "{user_input}"

    Think step-by-step, then list the robot commands.

    Your response:
    <think>
    """

            print("\n⏳ Generating response from DeepSeek...")

            start_time = time.time()
            response = self.pipeline(
                full_prompt,
                max_new_tokens=250,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                top_p=0.9
            )

            elapsed = time.time() - start_time
            print(f"✓ Response generated in {elapsed:.2f}s")

            generated_text = response[0]['generated_text']

            # Extract after our prompt
            if "Your response:" in generated_text:
                llm_response = generated_text.split("Your response:")[-1].strip()
            else:
                llm_response = generated_text

            # Ensure it has the think structure
            if "<think>" not in llm_response:
                llm_response = f"<think>\nAnalyzing: {user_input}\n</think>\n\n{llm_response}"

            # If response doesn't have commands, add them
            if "Commands:" not in llm_response.lower():
                # Extract actions from user input
                user_lower = user_input.lower()
                if "walk forward" in user_lower:
                    cmd = "walk forward slowly" if "slow" in user_lower else "walk forward"
                    llm_response += f"\n\nCommands: {cmd}"
                elif "turn left" in user_lower:
                    llm_response += "\n\nCommands: turn left"
                elif "turn right" in user_lower:
                    llm_response += "\n\nCommands: turn right"
                elif "stop" in user_lower:
                    llm_response += "\n\nCommands: stop moving"
                else:
                    llm_response += f"\n\nCommands: {user_input.lower()}"

            return llm_response

        except Exception as e:
            print(f"❌ DeepSeek call failed: {e}")
            import traceback
            traceback.print_exc()
            return self._simulated_response(user_input, context)

    def _simulated_response(self, user_input: str, context: Dict) -> str:
        """
        Simulated LLM response with chain of thought
        """
        user_lower = user_input.lower()

        # Pattern matching for common requests
        if "clean" in user_lower or "wipe" in user_lower:
            return """<think>
The user wants me to clean something. Let me break this down:
1. First, I need to reach forward to the surface
2. Then make wiping motions across the surface
3. I should move slowly to be thorough
4. Finally, return to neutral position
This requires: reach, repetitive forward motions, and controlled movement
</think>

Commands: reach forward, walk forward slowly, turn left, walk forward slowly, turn right, stop moving"""

        elif "walk forward" in user_lower or "move forward" in user_lower:
            speed = "slowly" if "slow" in user_lower else "quickly" if "fast" in user_lower or "quick" in user_lower else ""
            return f"""<think>
User wants me to walk forward. I need to:
1. Check that the path is clear (assumed)
2. Maintain good balance while walking
3. Walk at {'slow' if speed == 'slowly' else 'fast' if speed == 'quickly' else 'normal'} speed
4. Be ready to stop if needed
</think>

Commands: walk forward {speed}"""

        elif "turn" in user_lower:
            direction = "left" if "left" in user_lower else "right"
            return f"""<think>
User wants me to turn {direction}. Steps:
1. Stop current motion first
2. Turn {direction} while maintaining balance
3. Complete the turn
4. Be ready for next instruction
</think>

Commands: stop moving, turn {direction}, stand still"""

        elif "dance" in user_lower:
            return """<think>
User wants me to dance! This is fun but requires:
1. Maintaining balance throughout
2. Alternating motions for rhythm
3. Multiple movement types
4. Safe execution
Let me create a simple dance sequence
</think>

Commands: walk forward slowly, turn left, walk forward slowly, turn right, wave hand, stop moving"""

        elif "stop" in user_lower or "still" in user_lower:
            return """<think>
User wants me to stop all motion. Safety first:
1. Immediately cease all movement
2. Maintain stable standing position
3. Wait for next instruction
</think>

Commands: stop moving, stand still"""

        elif "pick" in user_lower or "grab" in user_lower or "grasp" in user_lower:
            return """<think>
User wants me to pick something up. Sequence:
1. Reach forward to the object
2. Position hand correctly
3. Grasp the object securely
4. Lift carefully
</think>

Commands: reach forward, pick up object, stand still"""

        else:
            return f"""<think>
I received the request: "{user_input}"
I'm not sure exactly what motion this requires. Let me:
1. Ask for clarification to be safe
2. Suggest some alternatives if helpful
I should not execute unclear commands for safety
</think>

I'm not sure I understand. Could you clarify? Try commands like: walk forward, turn left, stop moving, clean table, pick up object."""

    def _extract_chain_of_thought(self, response: str) -> str:
        """Extract chain of thought from <think> tags"""
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "No explicit reasoning provided"

    def _extract_motion_commands(self, response: str) -> List[str]:
        """Extract motion commands from response"""
        # Look for "Commands:" line
        commands_match = re.search(r'Commands?:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if commands_match:
            commands_str = commands_match.group(1).strip()
            # Split by comma
            commands = [cmd.strip() for cmd in commands_str.split(',')]
            return [cmd for cmd in commands if cmd]

        # Fallback: extract known commands
        known_commands = [
            "walk forward slowly", "walk forward quickly", "walk forward",
            "walk backward", "turn left", "turn right",
            "stop moving", "stand still", "crouch down",
            "reach forward", "pick up object", "place object",
            "wave hand", "clean table"
        ]

        found_commands = []
        response_lower = response.lower()
        for cmd in known_commands:
            if cmd in response_lower:
                found_commands.append(cmd)

        return found_commands if found_commands else ["stand still"]

    def get_conversation_summary(self) -> Dict:
        """Get summary of conversation"""
        return {
            "total_turns": len(self.conversation_history),
            "total_commands_issued": sum(len(turn.extracted_actions) for turn in self.conversation_history),
            "recent_turns": [turn.to_dict() for turn in self.conversation_history[-5:]],
            "using_real_model": self.use_real_model
        }


def test_deepseek_llm():
    """Test the DeepSeek LLM interface"""
    print("Testing DeepSeek LLM Interface")
    print("=" * 60)

    # Try to use real model
    llm = DeepSeekLLM(use_real_model=True)

    test_requests = [
        "Hey, can you clean the table?",
        "Walk forward please",
        "Turn left and then walk forward",
        "Stop moving",
        "Dance for me!",
        "Pick up that object over there"
    ]

    for request in test_requests:
        turn = llm.process_user_request(request)
        print(f"\n✓ Processed: '{request}'")
        print(f"  Actions: {turn.extracted_actions}")
        time.sleep(1)

    print("\n" + "=" * 60)
    print("Summary:")
    summary = llm.get_conversation_summary()
    print(f"Total turns: {summary['total_turns']}")
    print(f"Total commands: {summary['total_commands_issued']}")
    print(f"Using real DeepSeek: {summary['using_real_model']}")


if __name__ == "__main__":
    test_deepseek_llm()