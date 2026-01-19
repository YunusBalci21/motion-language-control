#!/usr/bin/env python3
"""
Feedback Loop Demo - Continuous Control from Open-Vocabulary Feedback

This is the CORE THESIS CONTRIBUTION:
1. User gives natural language command
2. LLM interprets with chain-of-thought reasoning
3. Agent executes motion with MotionGPT scoring
4. User provides natural language FEEDBACK
5. LLM interprets feedback → reward adjustments
6. System retrains with adjusted rewards
7. Agent executes improved motion

Author: Yunus Emre Balci
Thesis: Continuous Control from Open-Vocabulary Feedback
"""

import sys
import os
import re
import json
import time
import copy
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import requests

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


# ============================================================================
# LLM INTERFACE - Chain of Thought Reasoning
# ============================================================================

class ChainOfThoughtLLM:
    """
    LLM for interpreting commands AND feedback with chain-of-thought reasoning.
    """

    # HuggingFace token - REPLACE WITH YOUR OWN
    HF_TOKEN = "******REMOVED******xKtdAamhcRUkFeugbAjPTtvCPjeMXhxUCE"  # <-- PLACE YOUR TOKEN HERE

    COMMAND_PROMPT = """You are a robot motion controller. Interpret the user's request and output motion commands.

AVAILABLE COMMANDS: walk forward, walk backward, run forward, turn left, turn right, stand still, hop forward

Respond with:
<think>
Your reasoning about what the user wants
</think>

Commands: command1, command2

Response: Brief acknowledgment

User: {input}"""

    FEEDBACK_PROMPT = """You are analyzing feedback about a robot's motion. The robot was told to "{instruction}".

Interpret the feedback and determine what reward adjustments are needed.

POSSIBLE ADJUSTMENTS:
- speed: "increase" or "decrease" (if too slow/fast)
- stability: "increase" (if falling/wobbly) or "decrease" (if too stiff)
- language_alignment: "increase" (if not following instruction)
- energy: "increase" (if too aggressive) or "decrease" (if too passive)

Respond with:
<think>
Your analysis of what's wrong and what to adjust
</think>

Adjustments: adjustment1, adjustment2
(format: "speed:decrease" or "stability:increase" etc.)

Response: Brief acknowledgment

Feedback: {feedback}"""

    def __init__(self):
        self.use_api = self._check_api()
        if self.use_api:
            print("✓ LLM: HuggingFace API connected")
        else:
            print("✓ LLM: Using local pattern matching (API unavailable)")

    def _check_api(self) -> bool:
        """Check if HuggingFace API is available"""
        if not self.HF_TOKEN or self.HF_TOKEN == "PLACE YOUR TOKEN HERE":
            return False
        try:
            # Quick check
            return True  # Assume it works, will fallback if not
        except:
            return False

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call HuggingFace Inference API"""
        headers = {
            "Authorization": f"Bearer {self.HF_TOKEN}",
            "Content-Type": "application/json"
        }

        # Try Mistral (free tier, reliable)
        url = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"

        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  ⚠ API error: {e}")

        return None

    def interpret_command(self, user_input: str) -> Dict:
        """Interpret a motion command with CoT reasoning."""

        # Try API first
        if self.use_api:
            prompt = self.COMMAND_PROMPT.format(input=user_input)
            response = self._call_api(prompt)

            if response:
                cot = self._extract_think(response)
                commands = self._extract_commands(response)
                print(f"   <think>{cot}</think>")
                print(f"   Commands: {commands}")
                return {"chain_of_thought": cot, "commands": commands, "source": "api"}

        # Fallback to pattern matching
        return self._local_interpret_command(user_input)

    def interpret_feedback(self, feedback: str, instruction: str = None) -> Dict:
        """Interpret feedback with CoT reasoning."""

        # Try API first
        if self.use_api:
            prompt = self.FEEDBACK_PROMPT.format(feedback=feedback, instruction=instruction or "walk forward")
            response = self._call_api(prompt)

            if response:
                cot = self._extract_think(response)
                adjustments = self._extract_adjustments(response)
                print(f"   <think>{cot}</think>")
                print(f"   Adjustments: {adjustments}")
                return {"chain_of_thought": cot, "adjustments": adjustments, "source": "api", "is_positive": False}

        # Fallback to pattern matching
        return self._local_interpret_feedback(feedback)

    def _extract_think(self, response: str) -> str:
        """Extract <think> content"""
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Return first paragraph as reasoning
        lines = response.strip().split('\n')
        return lines[0] if lines else "Analyzing request..."

    def _extract_commands(self, response: str) -> List[str]:
        """Extract commands from response"""
        match = re.search(r'Commands?:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if match:
            cmds = match.group(1).strip()
            return [c.strip() for c in cmds.split(',') if c.strip()]
        return ["stand still"]

    def _extract_adjustments(self, response: str) -> Dict:
        """Extract adjustments from response"""
        match = re.search(r'Adjustments?:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        adjustments = {}

        if match:
            adj_str = match.group(1).strip()
            for part in adj_str.split(','):
                part = part.strip()
                if ':' in part:
                    key, val = part.split(':', 1)
                    adjustments[key.strip().lower()] = val.strip().lower()

        return adjustments

    def _local_interpret_command(self, user_input: str) -> Dict:
        """Local pattern matching for commands"""
        user_lower = user_input.lower()

        # Pattern matching - check backward BEFORE forward (since "forward" could match partial words)
        if any(w in user_lower for w in ["backward", "backwards", "back up", "reverse", "retreat"]):
            cot = "User wants backward movement. I'll walk backward."
            cmds = ["walk backward"]
        elif any(w in user_lower for w in ["forward", "walk", "come", "approach", "ahead"]):
            if "slow" in user_lower:
                cot = "User wants slow forward movement."
                cmds = ["walk forward slowly"]
            elif "fast" in user_lower or "run" in user_lower or "quick" in user_lower:
                cot = "User wants fast movement. I'll run forward."
                cmds = ["run forward"]
            else:
                cot = "User wants forward movement. I'll walk forward."
                cmds = ["walk forward"]
        elif any(w in user_lower for w in ["run", "fast", "quick", "hurry", "sprint"]):
            cot = "User wants fast movement. I'll run forward."
            cmds = ["run forward"]
        elif any(w in user_lower for w in ["left"]):
            cot = "User wants me to turn left."
            cmds = ["turn left"]
        elif any(w in user_lower for w in ["right"]):
            cot = "User wants me to turn right."
            cmds = ["turn right"]
        elif any(w in user_lower for w in ["stop", "halt", "freeze", "still", "stay"]):
            cot = "User wants me to stop moving."
            cmds = ["stand still"]
        elif any(w in user_lower for w in ["hop", "jump", "bounce"]):
            cot = "User wants hopping motion."
            cmds = ["hop forward"]
        elif any(w in user_lower for w in ["fall", "down", "drop"]):
            cot = "User wants me to fall/go down. I'll try to lower myself."
            cmds = ["stand still"]  # No explicit fall command, use stand still
        else:
            cot = "I'll interpret this as a general forward movement request."
            cmds = ["walk forward"]

        print(f"   <think>{cot}</think>")
        print(f"   Commands: {cmds}")
        return {"chain_of_thought": cot, "commands": cmds, "source": "local"}

    def _local_interpret_feedback(self, feedback: str) -> Dict:
        """Local pattern matching for feedback"""
        feedback_lower = feedback.lower()
        adjustments = {}
        cot_parts = []
        is_positive = False

        # Speed feedback
        if any(w in feedback_lower for w in ["too fast", "too quick", "slow down", "slower"]):
            adjustments["speed"] = "decrease"
            cot_parts.append("Motion is too fast, need to slow down")
        elif any(w in feedback_lower for w in ["too slow", "faster", "speed up", "quicker"]):
            adjustments["speed"] = "increase"
            cot_parts.append("Motion is too slow, need to speed up")

        # Stability feedback
        if any(w in feedback_lower for w in ["fall", "unstable", "wobbly", "balance", "tipping"]):
            adjustments["stability"] = "increase"
            cot_parts.append("Agent is unstable, need more stability focus")
        elif any(w in feedback_lower for w in ["stiff", "rigid", "robotic"]):
            adjustments["stability"] = "decrease"
            cot_parts.append("Motion is too stiff, allow more freedom")

        # Energy feedback
        if any(w in feedback_lower for w in ["aggressive", "violent", "jerky", "harsh"]):
            adjustments["energy"] = "decrease"  # decrease means increase penalty
            cot_parts.append("Motion is too aggressive, penalize high energy more")
        elif any(w in feedback_lower for w in ["lazy", "weak", "passive", "sluggish"]):
            adjustments["energy"] = "increase"
            cot_parts.append("Motion lacks energy, allow more energy")

        # Direction/instruction feedback
        if any(w in feedback_lower for w in ["wrong", "not what", "different", "other way"]):
            adjustments["language_alignment"] = "increase"
            cot_parts.append("Not following instruction well, increase language weight")

        # Positive feedback
        if any(w in feedback_lower for w in ["good", "great", "perfect", "nice", "yes", "correct"]):
            is_positive = True
            cot_parts.append("Positive feedback - no changes needed")

        cot = ". ".join(cot_parts) if cot_parts else "Analyzing feedback for adjustments"

        print(f"   <think>{cot}</think>")
        print(f"   Adjustments: {adjustments}")
        return {"chain_of_thought": cot, "adjustments": adjustments, "source": "local", "is_positive": is_positive}


# ============================================================================
# ADAPTIVE REWARD WRAPPER - Adjustable reward weights
# ============================================================================

class AdaptiveMotionLanguageWrapper(gym.Wrapper):
    """
    Wrapper with ADJUSTABLE reward weights that can be modified based on feedback.
    """

    def __init__(
            self,
            env: gym.Env,
            motion_tokenizer,
            instruction: str = "walk forward",
            initial_weights: Dict = None,
    ):
        super().__init__(env)
        self.motion_tokenizer = motion_tokenizer
        self.instruction = instruction

        # Default reward weights (can be adjusted via feedback)
        self.weights = {
            "stability": 0.25,
            "task": 0.15,
            "language": 0.50,  # MotionGPT similarity - main contribution
            "consistency": 0.10,
            "energy_penalty": 0.001,
            "speed_target": 0.8,  # Target forward speed
        }

        if initial_weights:
            self.weights.update(initial_weights)

        # Motion history for MotionGPT
        from collections import deque
        self.motion_history = deque(maxlen=32)
        self.speed_history = deque(maxlen=32)

        # MuJoCo handles
        self._init_mujoco()

        # Tracking
        self.episode_similarities = []
        self.episode_rewards = []

        # Environment info
        self.env_name = getattr(getattr(env, 'spec', None), 'id', None) or "Humanoid-v4"

    def _init_mujoco(self):
        try:
            self.mj_data = getattr(self.unwrapped, "data", None)
            self.mj_model = getattr(self.unwrapped, "model", None)
        except:
            self.mj_data = None
            self.mj_model = None

    def adjust_weights(self, adjustments: Dict):
        """Apply feedback-based adjustments to reward weights."""
        weight_map = {
            "stability_weight": "stability",
            "task_weight": "task",
            "language_weight": "language",
            "consistency_weight": "consistency",
            "energy_penalty": "energy_penalty",
            "speed_target": "speed_target",
        }

        for adj_key, delta in adjustments.items():
            if adj_key in weight_map:
                weight_key = weight_map[adj_key]
                old_val = self.weights[weight_key]
                new_val = max(0.0, min(1.0, old_val + delta))  # Clamp to [0, 1]
                self.weights[weight_key] = new_val
                print(f"    {weight_key}: {old_val:.2f} → {new_val:.2f}")

    def get_weights_summary(self) -> str:
        return (f"stability={self.weights['stability']:.2f}, "
                f"task={self.weights['task']:.2f}, "
                f"language={self.weights['language']:.2f}, "
                f"speed_target={self.weights['speed_target']:.1f}")

    def reset(self, **kwargs):
        self.motion_history.clear()
        self.speed_history.clear()
        self.episode_similarities = []
        self.episode_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Get state
        state = self._get_state()

        # Extract motion features
        motion_features = self.motion_tokenizer.extract_motion_from_obs(obs, self.env_name)
        self.motion_history.append(motion_features)
        self.speed_history.append(state['vx'])

        # Compute reward components
        stability_reward = self._compute_stability_reward(state)
        task_reward = self._compute_task_reward(state)
        language_reward, similarity = self._compute_language_reward()
        consistency_reward = self._compute_consistency_reward()
        energy_penalty = self.weights["energy_penalty"] * np.sum(np.square(action))

        # Combine with adjustable weights
        total_reward = (
                self.weights["stability"] * stability_reward +
                self.weights["task"] * task_reward +
                self.weights["language"] * language_reward +
                self.weights["consistency"] * consistency_reward -
                energy_penalty
        )

        # Scale for PPO
        total_reward *= 2.0

        # Track
        self.episode_similarities.append(similarity)
        self.episode_rewards.append(total_reward)

        info.update({
            "motion_similarity": similarity,
            "stability_reward": stability_reward,
            "task_reward": task_reward,
            "language_reward": language_reward,
            "weights": self.weights.copy(),
            "vx": state["vx"],
            "height": state["height"],
        })

        return obs, total_reward, terminated, truncated, info

    def _get_state(self) -> Dict:
        # Default height depends on environment
        default_height = 0.75 if "Ant" in self.env_name else 1.25
        state = {"height": default_height, "vx": 0.0, "vy": 0.0}
        if self.mj_data is not None:
            try:
                state["height"] = float(self.mj_data.qpos[2])
                state["vx"] = float(self.mj_data.qvel[0])
                state["vy"] = float(self.mj_data.qvel[1])
            except:
                pass
        return state

    def _compute_stability_reward(self, state: Dict) -> float:
        height = state["height"]
        # Adaptive target height based on environment
        if "Humanoid" in self.env_name:
            target_height = 1.25
        elif "Ant" in self.env_name:
            target_height = 0.75
        elif "HalfCheetah" in self.env_name:
            target_height = 0.7
        elif "Walker2d" in self.env_name:
            target_height = 1.2
        else:
            target_height = height  # No penalty if unknown

        height_error = abs(height - target_height)
        return np.exp(-2.0 * height_error ** 2)

    def _compute_task_reward(self, state: Dict) -> float:
        vx = state["vx"]
        target_speed = self.weights["speed_target"]

        if "forward" in self.instruction.lower():
            if vx > 0.1:
                speed_error = abs(vx - target_speed)
                return np.exp(-2.0 * speed_error ** 2)
            return 0.3 * max(0, vx / 0.1)
        elif "backward" in self.instruction.lower():
            if vx < -0.1:
                return min(-vx / target_speed, 1.0)
            return 0.3 * max(0, -vx / 0.1)
        elif "stand" in self.instruction.lower() or "still" in self.instruction.lower():
            speed = np.sqrt(vx ** 2 + state["vy"] ** 2)
            return np.exp(-5.0 * speed ** 2)
        return 0.5

    def _compute_language_reward(self) -> Tuple[float, float]:
        if len(self.motion_history) < 16:
            return 0.3, 0.0

        try:
            motion_seq = np.array(list(self.motion_history))
            similarity = self.motion_tokenizer.compute_motion_language_similarity(
                motion_seq, self.instruction, temporal_aggregation="mean"
            )
            return float(similarity), float(similarity)
        except:
            return 0.3, 0.0

    def _compute_consistency_reward(self) -> float:
        if len(self.speed_history) < 10:
            return 0.0
        recent = list(self.speed_history)[-10:]
        std = np.std(recent)
        mean = np.mean(recent)
        if std < 0.3 and abs(mean) > 0.2:
            return 0.2
        return 0.0


# ============================================================================
# FEEDBACK LEARNING SYSTEM
# ============================================================================

class FeedbackLearningSystem:
    """
    The core thesis contribution: Learning from natural language feedback.

    Pipeline:
    1. LLM interprets command with chain-of-thought
    2. Execute motion with current policy
    3. LLM interprets feedback with chain-of-thought
    4. Adjust reward weights based on feedback
    5. Fine-tune policy with adjusted rewards
    6. Execute again to show improvement
    """

    MOTION_CHECKPOINT = "./external/MotionGPT/prepare/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar"

    def __init__(
            self,
            env_name: str = "Humanoid-v4",
            model_path: str = "./checkpoints_v2/humanoid_walk_forward/final_model.zip",
            device: str = "cuda",
    ):
        print("\n" + "=" * 70)
        print("🎯 FEEDBACK LEARNING SYSTEM")
        print("   Continuous Control from Open-Vocabulary Feedback")
        print("=" * 70)

        self.env_name = env_name
        self.device = device
        self.model_path = model_path

        # Initialize LLM
        print("\n📦 Loading LLM...")
        self.llm = ChainOfThoughtLLM()

        # Load MotionGPT tokenizer
        print("\n📦 Loading MotionGPT...")
        from models.motion_tokenizer import MotionTokenizer
        self.motion_tokenizer = MotionTokenizer(
            device=device,
            checkpoint_path=self.MOTION_CHECKPOINT
        )

        # Load base model
        print("\n📦 Loading trained model...")
        self.base_model = PPO.load(model_path, device=device)
        self.current_model = PPO.load(model_path, device=device)

        # Current reward weights
        self.current_weights = {
            "stability": 0.25,
            "task": 0.15,
            "language": 0.50,
            "consistency": 0.10,
            "energy_penalty": 0.001,
            "speed_target": 0.8,
        }

        # Current instruction
        self.current_instruction = None

        # History
        self.feedback_history = []
        self.execution_history = []

        print("\n✅ System ready!")
        print(f"   Environment: {env_name}")
        print(f"   Model: {model_path}")
        print("=" * 70)

    def _create_env(self, instruction: str, render: bool = False, live: bool = False, max_steps: int = None) -> gym.Env:
        """Create environment with adaptive reward wrapper."""
        if live:
            render_mode = "human"  # Live window
        elif render:
            render_mode = "rgb_array"  # For video recording
        else:
            render_mode = None

        # Override max_episode_steps if specified (default is usually 1000)
        if max_steps and max_steps > 1000:
            base_env = gym.make(self.env_name, render_mode=render_mode, max_episode_steps=max_steps)
        else:
            base_env = gym.make(self.env_name, render_mode=render_mode)

        wrapped_env = AdaptiveMotionLanguageWrapper(
            base_env,
            motion_tokenizer=self.motion_tokenizer,
            instruction=instruction,
            initial_weights=self.current_weights.copy(),
        )
        return wrapped_env

    def execute(
            self,
            instruction: str,
            steps: int = 200,
            record: bool = False,
            live: bool = False,
            video_dir: str = "./feedback_videos"
    ) -> Dict:
        """Execute current policy and return results."""

        mode_str = "LIVE" if live else ("recording" if record else "")
        print(f"\n🏃 Executing: '{instruction}' for {steps} steps {mode_str}")
        print(f"   Current weights: {self._weights_str()}")

        # Pass steps to create env with higher max_episode_steps if needed
        env = self._create_env(instruction, render=record, live=live, max_steps=steps)
        obs, info = env.reset()

        frames = [] if record else None
        rewards = []
        similarities = []

        for step in range(steps):
            action, _ = self.current_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            rewards.append(reward)
            if "motion_similarity" in info:
                similarities.append(info["motion_similarity"])

            if record:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if (step + 1) % 50 == 0:
                avg_sim = np.mean(similarities[-50:]) if similarities else 0
                print(f"   Step {step + 1}/{steps}: reward={np.mean(rewards[-50:]):.2f}, similarity={avg_sim:.3f}")

            if terminated or truncated:
                print(f"   Episode ended at step {step + 1}")
                break

        env.close()

        # Save video if recording
        video_path = None
        if record and frames and IMAGEIO_AVAILABLE:
            os.makedirs(video_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"{video_dir}/{instruction.replace(' ', '_')}_{timestamp}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            print(f"   📹 Video saved: {video_path}")

        result = {
            "instruction": instruction,
            "steps": len(rewards),
            "mean_reward": float(np.mean(rewards)),
            "mean_similarity": float(np.mean(similarities)) if similarities else 0,
            "max_similarity": float(np.max(similarities)) if similarities else 0,
            "weights": self.current_weights.copy(),
            "video_path": video_path,
        }

        self.execution_history.append(result)

        print(f"\n   ✅ Result: reward={result['mean_reward']:.2f}, "
              f"similarity={result['mean_similarity']:.3f} (max: {result['max_similarity']:.3f})")

        return result

    def process_command(self, user_input: str) -> Dict:
        """
        Process user command through LLM with chain-of-thought and execute.
        """
        # LLM interprets command with CoT
        interpretation = self.llm.interpret_command(user_input)

        # Get the primary command
        commands = interpretation["commands"]
        primary_command = commands[0] if commands else "stand still"

        # Store current instruction
        self.current_instruction = primary_command

        return interpretation

    def process_feedback(self, feedback: str) -> Dict:
        """Process natural language feedback through LLM and adjust reward weights."""

        print(f"\n💬 Processing feedback: \"{feedback}\"")

        if self.current_instruction is None:
            print("   ⚠ No instruction set. Give a command first.")
            return {"action": "none"}

        # LLM interprets feedback with CoT
        interpretation = self.llm.interpret_feedback(feedback, self.current_instruction)
        adjustments = interpretation["adjustments"]

        if adjustments.get("positive"):
            print("   ✅ Positive feedback - keeping current settings")
            return {"action": "none", "interpretation": interpretation}

        if not adjustments:
            print("   ⚠ Could not interpret feedback into adjustments")
            return {"action": "none", "interpretation": interpretation}

        # Apply LLM-interpreted adjustments
        print(f"\n   📊 Adjusting reward weights based on feedback:")
        self._apply_adjustments(adjustments)

        self.feedback_history.append({
            "feedback": feedback,
            "interpretation": interpretation,
            "new_weights": self.current_weights.copy(),
            "timestamp": datetime.now().isoformat()
        })

        return {"action": "adjusted", "interpretation": interpretation, "new_weights": self.current_weights.copy()}

    def _apply_adjustments(self, adjustments: Dict):
        """Apply LLM-interpreted adjustments to weights."""

        for key, direction in adjustments.items():
            if key == "positive":
                continue

            delta = 0.15 if direction == "increase" else -0.15

            if key == "speed":
                old_val = self.current_weights["speed_target"]
                new_val = max(0.3, min(1.5, old_val + (delta * 2)))
                self.current_weights["speed_target"] = new_val
                print(f"      speed_target: {old_val:.2f} → {new_val:.2f}")

            elif key == "stability":
                old_val = self.current_weights["stability"]
                new_val = max(0.1, min(0.6, old_val + delta))
                self.current_weights["stability"] = new_val
                print(f"      stability: {old_val:.2f} → {new_val:.2f}")

            elif key == "language_alignment":
                old_val = self.current_weights["language"]
                new_val = max(0.2, min(0.8, old_val + delta))
                self.current_weights["language"] = new_val
                print(f"      language: {old_val:.2f} → {new_val:.2f}")

            elif key == "energy":
                old_val = self.current_weights["energy_penalty"]
                new_val = max(0.0005, min(0.005, old_val + (delta * 0.02)))
                self.current_weights["energy_penalty"] = new_val
                print(f"      energy_penalty: {old_val:.4f} → {new_val:.4f}")

    def _apply_llm_adjustments(self, adjustments: Dict):
        """Apply LLM-interpreted adjustments to weights (alias for interactive demo)."""
        for key, direction in adjustments.items():
            if key in ["positive", "none"]:
                continue

            delta = 0.15 if direction == "increase" else -0.15

            if key == "speed":
                old_val = self.current_weights["speed_target"]
                new_val = max(0.3, min(1.5, old_val + (delta * 2)))
                self.current_weights["speed_target"] = new_val
                print(f"      speed_target: {old_val:.2f} → {new_val:.2f}")

            elif key == "stability":
                old_val = self.current_weights["stability"]
                new_val = max(0.1, min(0.6, old_val + delta))
                self.current_weights["stability"] = new_val
                print(f"      stability: {old_val:.2f} → {new_val:.2f}")

            elif key == "language_alignment":
                old_val = self.current_weights["language"]
                new_val = max(0.2, min(0.8, old_val + delta))
                self.current_weights["language"] = new_val
                print(f"      language: {old_val:.2f} → {new_val:.2f}")

            elif key == "energy":
                old_val = self.current_weights["energy_penalty"]
                # Decrease energy penalty = allow more aggressive motion
                # Increase energy penalty = penalize aggressive motion more
                mult = -1 if direction == "decrease" else 1  # flip for intuitive feedback
                new_val = max(0.0005, min(0.005, old_val + (0.001 * mult)))
                self.current_weights["energy_penalty"] = new_val
                print(f"      energy_penalty: {old_val:.4f} → {new_val:.4f}")

    def fine_tune(
            self,
            instruction: str,
            training_steps: int = 20000,
            show_progress: bool = True
    ):
        """Fine-tune the policy with current reward weights."""

        print(f"\n🔧 Fine-tuning for {training_steps:,} steps with adjusted rewards...")
        print(f"   Weights: {self._weights_str()}")

        # Create vectorized environment with current weights (4 parallel envs to match training)
        def make_env():
            env = gym.make(self.env_name)
            return AdaptiveMotionLanguageWrapper(
                env,
                motion_tokenizer=self.motion_tokenizer,
                instruction=instruction,
                initial_weights=self.current_weights.copy()
            )

        # Use 4 envs to match the original training setup
        vec_env = DummyVecEnv([make_env for _ in range(4)])

        # Reload model with new environment (required when n_envs differs or env changes)
        self.current_model = PPO.load(
            self.model_path,
            env=vec_env,
            device=self.device
        )

        # Fine-tune
        self.current_model.learn(
            total_timesteps=training_steps,
            progress_bar=show_progress,
            reset_num_timesteps=False
        )

        vec_env.close()

        print(f"   ✅ Fine-tuning complete!")

    def reset_to_base(self):
        """Reset model to original trained weights."""
        print("\n🔄 Resetting to base model...")
        self.current_model = PPO.load(self.model_path, device=self.device)
        self.current_weights = {
            "stability": 0.25,
            "task": 0.15,
            "language": 0.50,
            "consistency": 0.10,
            "energy_penalty": 0.001,
            "speed_target": 0.8,
        }
        print("   ✅ Reset complete")

    def _weights_str(self) -> str:
        return (f"stab={self.current_weights['stability']:.2f}, "
                f"task={self.current_weights['task']:.2f}, "
                f"lang={self.current_weights['language']:.2f}, "
                f"speed={self.current_weights['speed_target']:.1f}")

    def save_session(self, path: str = "./feedback_session.json"):
        """Save feedback session for analysis."""
        session = {
            "feedback_history": self.feedback_history,
            "execution_history": self.execution_history,
            "final_weights": self.current_weights,
        }
        with open(path, "w") as f:
            json.dump(session, f, indent=2)
        print(f"✅ Session saved to {path}")


# ============================================================================
# INTERACTIVE DEMO
# ============================================================================

def run_interactive_demo():
    """Run interactive feedback loop demo."""

    print("\n" + "=" * 70)
    print("🤖 INTERACTIVE FEEDBACK LEARNING DEMO")
    print("=" * 70)
    print("""
This demo shows the CORE THESIS CONTRIBUTION:
Learning continuous control from open-vocabulary feedback.

WORKFLOW:
1. Give a command (e.g., 'walk forward')
2. Watch the agent execute
3. Provide feedback (e.g., 'too fast', 'falling over')
4. System adjusts and optionally retrains
5. See improvement!

COMMANDS:
  <any text>      - Interpreted as motion command first time, feedback after
  !walk forward   - Force as motion command (start with !)
  !train          - Fine-tune with current feedback (default 20k steps)
  !train 50000    - Fine-tune with specific steps
  !reset          - Reset to original model
  !weights        - Show current reward weights
  !history        - Show feedback history
  !record on/off  - Toggle video recording
  !live on/off    - Toggle live rendering window (ON by default)
  !envs           - List available environments
  !env ant        - Switch environment (ant, humanoid, halfcheetah, walker2d, hopper)
  !quit           - Exit

EXAMPLE SESSION:
  You: walk forward
  [agent walks, shows similarity score]
  You: too fast and aggressive
  [LLM interprets → adjusts weights → AUTO-TRAINS 5k steps → executes again]
  You: still too wobbly
  [adjusts stability → trains → shows improvement]
  You: perfect!
  [positive feedback, no changes]
""")
    print("=" * 70)

    # Initialize system - Use ANT (more stable for demo)
    system = FeedbackLearningSystem(
        env_name="Ant-v4",
        model_path="./checkpoints_v2/ant_walk_forward/final_model.zip",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    current_instruction = None
    recording = False
    live_mode = True  # Default to live rendering!

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["!quit", "!exit", "quit", "exit"]:
                system.save_session()
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == "!weights":
                print(f"\n📊 Current weights: {system._weights_str()}")
                continue

            if user_input.lower() == "!history":
                print("\n📜 Feedback history:")
                for i, fb in enumerate(system.feedback_history):
                    interp = fb.get('interpretation', fb.get('parsed', {}))
                    adj = interp.get('adjustments', interp.get('matched_patterns', []))
                    print(f"  {i + 1}. \"{fb['feedback']}\" → {adj}")
                continue

            if user_input.lower() == "!reset":
                system.reset_to_base()
                current_instruction = None
                continue

            if user_input.lower() == "!record on":
                recording = True
                print("📹 Recording: ON")
                continue

            if user_input.lower() == "!record off":
                recording = False
                print("📹 Recording: OFF")
                continue

            if user_input.lower() == "!live on":
                live_mode = True
                print("🖥️ Live rendering: ON")
                continue

            if user_input.lower() == "!live off":
                live_mode = False
                print("🖥️ Live rendering: OFF")
                continue

            if user_input.lower() == "!envs" or user_input.lower() == "!environments":
                print("\n🌍 Available environments:")
                print("   ant        - Quadruped (most stable, recommended)")
                print("   humanoid   - Bipedal humanoid (challenging)")
                print("   halfcheetah - 2D runner (fast)")
                print("   walker2d   - 2D bipedal walker")
                print("   hopper     - 2D single-leg hopper")
                print(f"\n   Current: {system.env_name}")
                print("\n   Usage: !env ant")
                continue

            if user_input.lower().startswith("!env "):
                env_key = user_input[5:].strip().lower()

                # Map short names to full config
                env_configs = {
                    "ant": ("Ant-v4", "./checkpoints_v2/ant_walk_forward/final_model.zip"),
                    "humanoid": ("Humanoid-v4", "./checkpoints_v2/humanoid_walk_forward/final_model.zip"),
                    "halfcheetah": ("HalfCheetah-v4", "./checkpoints_v2/halfcheetah_run_forward/final_model.zip"),
                    "walker2d": ("Walker2d-v4", "./checkpoints_v2/walker2d_walk_forward/final_model.zip"),
                    "hopper": ("Hopper-v4", "./checkpoints_v2/hopper_hop_forward/final_model.zip"),
                }

                if env_key not in env_configs:
                    print(f"⚠ Unknown environment: {env_key}")
                    print("   Available: ant, humanoid, halfcheetah, walker2d, hopper")
                    continue

                env_name, model_path = env_configs[env_key]

                print(f"\n🔄 Switching to {env_key}...")
                system.env_name = env_name
                system.model_path = model_path

                # Reload model
                try:
                    system.current_model = PPO.load(model_path, device=system.device)
                    system.base_model = PPO.load(model_path, device=system.device)
                    # Reset weights
                    system.current_weights = {
                        "stability": 0.25,
                        "task": 0.15,
                        "language": 0.50,
                        "consistency": 0.10,
                        "energy_penalty": 0.001,
                        "speed_target": 0.8,
                    }
                    current_instruction = None
                    print(f"✅ Now using: {env_name}")
                    print(f"   Model: {model_path}")
                except Exception as e:
                    print(f"⚠ Failed to load model: {e}")
                continue

            if user_input.lower().startswith("!train"):
                if current_instruction is None:
                    print("⚠ No instruction set. Give a command first (e.g., 'walk forward')")
                    continue

                # Parse optional steps
                parts = user_input.split()
                steps = 20000
                if len(parts) > 1:
                    try:
                        steps = int(parts[1])
                    except:
                        pass

                system.fine_tune(current_instruction, training_steps=steps)

                # Execute again to show improvement
                print("\n🔄 Executing again to show improvement...")
                system.execute(current_instruction, steps=200, record=recording, live=live_mode)
                continue

            # Force as instruction (starts with !)
            if user_input.startswith("!"):
                raw_instruction = user_input[1:].strip()
                parts = raw_instruction.split()

                # Extract step count from anywhere
                steps = 300
                for p in parts:
                    if p.isdigit():
                        steps = int(p)
                        break

                # Clean input for LLM
                clean_input = " ".join([p for p in parts if not p.isdigit() and p.lower() != "steps"])

                # Use LLM to interpret
                print(f"\n🧠 LLM interpreting: \"{clean_input}\"")
                interpretation = system.llm.interpret_command(clean_input)

                current_instruction = interpretation['commands'][0] if interpretation['commands'] else clean_input
                system.current_instruction = current_instruction
                system.execute(current_instruction, steps=steps, record=recording, live=live_mode)
                continue

            # Check if user wants to switch environment via natural language
            env_names_in_text = {
                "ant": "ant",
                "humanoid": "humanoid",
                "half cheetah": "halfcheetah",
                "halfcheetah": "halfcheetah",
                "cheetah": "halfcheetah",
                "walker": "walker2d",
                "walker2d": "walker2d",
                "hopper": "hopper",
            }

            user_lower_check = user_input.lower()
            detected_env = None

            # Check if any environment name is mentioned
            for env_text, env_key in env_names_in_text.items():
                if env_text in user_lower_check:
                    # Make sure it's not the current environment
                    current_env_short = system.env_name.split("-")[0].lower()
                    if env_key != current_env_short:
                        detected_env = env_key
                        print(f"   [Detected environment: {detected_env}]")
                        break

            if detected_env:
                # Extract step count if present
                requested_steps = 300
                parts = user_input.split()
                for i, p in enumerate(parts):
                    if p.isdigit():
                        requested_steps = int(p)
                        break

                # Switch environment
                env_configs = {
                    "ant": ("Ant-v4", "./checkpoints_v2/ant_walk_forward/final_model.zip"),
                    "humanoid": ("Humanoid-v4", "./checkpoints_v2/humanoid_walk_forward/final_model.zip"),
                    "halfcheetah": ("HalfCheetah-v4", "./checkpoints_v2/halfcheetah_run_forward/final_model.zip"),
                    "walker2d": ("Walker2d-v4", "./checkpoints_v2/walker2d_walk_forward/final_model.zip"),
                    "hopper": ("Hopper-v4", "./checkpoints_v2/hopper_hop_forward/final_model.zip"),
                }

                env_name, model_path = env_configs[detected_env]

                print(f"\n🔄 Switching to {detected_env}...")
                system.env_name = env_name
                system.model_path = model_path

                try:
                    system.current_model = PPO.load(model_path, device=system.device)
                    system.base_model = PPO.load(model_path, device=system.device)
                    system.current_weights = {
                        "stability": 0.25, "task": 0.15, "language": 0.50,
                        "consistency": 0.10, "energy_penalty": 0.001, "speed_target": 0.8,
                    }
                    print(f"✅ Now using: {env_name}")

                    # Check if they also want to run a command
                    motion_words = ["walk", "run", "turn", "stop", "hop", "forward", "backward"]
                    has_motion = any(w in user_lower_check for w in motion_words)

                    if has_motion:
                        # Extract and run the command
                        print(f"\n🧠 LLM interpreting command...")
                        interpretation = system.llm.interpret_command(user_input)
                        current_instruction = interpretation['commands'][0] if interpretation[
                            'commands'] else "walk forward"
                        system.current_instruction = current_instruction
                        system.execute(current_instruction, steps=requested_steps, record=recording, live=live_mode)
                    else:
                        current_instruction = None
                        print("   Ready! Give a motion command.")

                except Exception as e:
                    print(f"⚠ Failed to load model: {e}")
                continue

            # Check if it's a train command (without !)
            if user_input.lower().startswith("train"):
                if current_instruction is None:
                    print("⚠ No instruction set. Give a command first (e.g., 'walk forward')")
                    continue

                parts = user_input.split()
                steps = 20000
                if len(parts) > 1:
                    try:
                        steps = int(parts[1])
                    except:
                        pass

                system.fine_tune(current_instruction, training_steps=steps)

                print("\n🔄 Executing again to show improvement...")
                system.execute(current_instruction, steps=200, record=recording, live=live_mode)
                continue

            # Interpret as instruction or feedback
            if current_instruction is None:
                # Extract step count from anywhere in input
                parts = user_input.split()
                steps = 300  # Default 300 steps for better demo
                for p in parts:
                    if p.isdigit():
                        steps = int(p)
                        break

                # Remove step count and "steps" word from input for LLM
                clean_input = " ".join([p for p in parts if not p.isdigit() and p.lower() != "steps"])

                # First input - use LLM to interpret as command
                print(f"\n🧠 LLM interpreting: \"{clean_input}\"")
                interpretation = system.llm.interpret_command(clean_input)

                current_instruction = interpretation['commands'][0] if interpretation['commands'] else clean_input
                system.current_instruction = current_instruction
                system.execute(current_instruction, steps=steps, record=recording, live=live_mode)
            else:
                # Subsequent input - check if it's feedback or new instruction
                user_lower = user_input.lower()

                # Command indicators - these suggest a new command
                command_words = ["can you", "do ", "please", "now ", "try", "walk", "run",
                                 "turn", "stop", "go ", "move", "backward", "forward"]
                has_command_intent = any(w in user_lower for w in command_words)

                # Feedback words
                feedback_words = ["too", "not ", "wrong", "more", "less", "good", "bad",
                                  "fast", "slow", "fall", "wobbl", "perfect", "great",
                                  "aggressive", "gentle", "harsh", "nice", "correct", "like"]
                has_feedback_words = any(w in user_lower for w in feedback_words)

                # If it has command intent AND a motion word, it's a new command
                motion_words = ["walk", "run", "turn", "stop", "backward", "forward", "left", "right", "hop"]
                has_motion = any(w in user_lower for w in motion_words)

                # Decide: command if (has command words AND motion words) OR (has motion but no pure feedback)
                is_new_command = (has_command_intent and has_motion) or (has_motion and not has_feedback_words)
                is_feedback = has_feedback_words and not is_new_command

                if is_feedback:
                    # Use LLM to interpret feedback
                    print(f"\n🧠 LLM interpreting feedback: \"{user_input}\"")
                    interpretation = system.llm.interpret_feedback(user_input)

                    if interpretation.get('is_positive') or not interpretation.get('adjustments'):
                        print("   ✅ Positive feedback - keeping current settings")
                    else:
                        # Apply adjustments
                        print("\n   📊 Applying adjustments:")
                        system._apply_llm_adjustments(interpretation['adjustments'])

                        # Save to history
                        system.feedback_history.append({
                            "feedback": user_input,
                            "interpretation": interpretation,
                            "new_weights": system.current_weights.copy(),
                            "timestamp": datetime.now().isoformat()
                        })

                        # AUTO-TRAIN with new weights!
                        print("\n🔧 Auto-training with adjusted rewards (5000 steps)...")
                        system.fine_tune(current_instruction, training_steps=5000, show_progress=True)

                        print("\n🔄 Executing again to show improvement...")
                        system.execute(current_instruction, steps=300, record=recording, live=live_mode)
                else:
                    # New instruction - extract step count from anywhere
                    parts = user_input.split()
                    steps = 300  # Default 300 steps
                    for p in parts:
                        if p.isdigit():
                            steps = int(p)
                            break

                    # Clean input for LLM
                    clean_input = " ".join([p for p in parts if not p.isdigit() and p.lower() != "steps"])

                    # Use LLM to interpret
                    print(f"\n🧠 LLM interpreting: \"{clean_input}\"")
                    interpretation = system.llm.interpret_command(clean_input)

                    current_instruction = interpretation['commands'][0] if interpretation['commands'] else clean_input
                    system.current_instruction = current_instruction
                    system.execute(current_instruction, steps=steps, record=recording, live=live_mode)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Saving session...")
            system.save_session()
            break


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feedback Learning Demo")
    parser.add_argument("--env", default="Ant-v4", help="Environment (Ant-v4, Humanoid-v4, HalfCheetah-v4)")
    parser.add_argument("--model", default="./checkpoints_v2/ant_walk_forward/final_model.zip")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    run_interactive_demo()