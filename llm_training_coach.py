import sys
import re
import json

# Ensure we can import from src
sys.path.append("src")
from conversation.deepseek_llm import DeepSeekLLM


class TrainingCoach:
    """
    Uses the LLM to act as a Physics Training Coach.
    Translates natural language feedback ("too jittery") into
    physics hyperparameters ("energy_penalty": 0.05).
    """

    def __init__(self):
        self.llm = DeepSeekLLM(use_real_model=True)

    def analyze_feedback(self,
                         instruction: str,
                         feedback: str,
                         current_metrics: dict,
                         current_params: dict) -> dict:

        prompt = f"""
You are an Expert Reinforcement Learning Coach tuning a physics-based character.
Your goal is to adjust training hyperparameters based on the User's feedback and current metrics.

CONTEXT:
- Task: "{instruction}"
- User Feedback: "{feedback}"
- Current Metrics: 
  * Velocity: {current_metrics.get('vx', 0):.2f} m/s
  * Success Rate: {current_metrics.get('success_rate', 0):.2f}
  * Falls: {current_metrics.get('falls', 0)}
- Current Parameters: {json.dumps(current_params, indent=2)}

INSTRUCTIONS:
1. Analyze why the agent is failing or acting undesirably using <think> tags.
   - If "not moving": Increase 'forward_reward_weight', decrease 'stability_weight'.
   - If "too aggressive/jittery": Increase 'energy_penalty', decrease 'action_magnitude'.
   - If "falling": Increase 'stability_weight', decrease 'target_speed'.
2. Output a JSON object with the UPDATED parameters.

RESPONSE FORMAT:
<think>
[Reasoning about the physics and reward function]
</think>

```json
{{
  "forward_reward_weight": <float>,
  "stability_weight": <float>,
  "energy_penalty": <float>,
  "target_speed": <float>,
  "action_magnitude": <float>
}}
```
"""
        response = self.llm.process_user_request(prompt).llm_response
        return self._extract_json(response, current_params)

    def _extract_json(self, response: str, fallback: dict) -> dict:
        try:
            # Find json block
            match = re.search(r"```json\s*({.*?})\s*```", response, re.DOTALL)
            if not match:
                match = re.search(r"({.*})", response, re.DOTALL)

            if match:
                new_params = json.loads(match.group(1))
                # Merge with fallback to ensure no keys are lost
                updated = fallback.copy()
                updated.update(new_params)
                return updated
        except Exception as e:
            print(f"⚠ Failed to parse LLM JSON: {e}")

        return fallback