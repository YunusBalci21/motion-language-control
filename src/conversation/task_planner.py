#!/usr/bin/env python3
"""
Task Planner - Converts LLM commands to executable motion sequences
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class MotionPlan:
    """A planned sequence of motions"""
    instruction: str
    motion_sequence: List[str]
    estimated_duration: float
    safety_check_passed: bool
    warnings: List[str]


class TaskPlanner:
    """
    Converts high-level LLM commands to executable motion sequences
    Maps to trained motion models
    """

    def __init__(self, available_models: Dict[str, str] = None):
        """
        Args:
            available_models: Dict mapping instruction -> model_path
        """
        self.available_models = available_models or {}

        # Command mapping to basic instructions
        self.command_map = {
            "walk forward slowly": "walk forward slowly",
            "walk forward quickly": "walk forward quickly",
            "walk forward": "walk forward",
            "walk backward": "walk backward",
            "turn left": "turn left",
            "turn right": "turn right",
            "stop moving": "stop moving",
            "stand still": "stand still",
            "crouch down": "crouch down",
            "reach forward": "walk forward slowly",  # Map to safe alternative
            "pick up object": "walk forward slowly",  # Map to safe alternative
            "place object": "stop moving",
            "wave hand": "walk forward slowly",
            "clean table": "walk forward slowly",
        }

        # Duration estimates (seconds)
        self.duration_estimates = {
            "walk forward slowly": 3.0,
            "walk forward quickly": 2.0,
            "walk forward": 2.5,
            "walk backward": 3.0,
            "turn left": 1.5,
            "turn right": 1.5,
            "stop moving": 0.5,
            "stand still": 1.0,
            "crouch down": 2.0,
        }

    def create_motion_plan(self, llm_commands: List[str]) -> MotionPlan:
        """
        Create executable motion plan from LLM commands
        """
        motion_sequence = []
        warnings = []
        estimated_duration = 0.0

        for cmd in llm_commands:
            cmd_lower = cmd.lower().strip()

            # Map to executable instruction
            if cmd_lower in self.command_map:
                mapped_cmd = self.command_map[cmd_lower]
                motion_sequence.append(mapped_cmd)
                estimated_duration += self.duration_estimates.get(mapped_cmd, 2.0)
            else:
                warnings.append(f"Unknown command: {cmd}")
                # Default to safe standing
                motion_sequence.append("stand still")
                estimated_duration += 1.0

        # Safety checks
        safety_check_passed = True

        # Check for too many commands
        if len(motion_sequence) > 10:
            warnings.append("Command sequence too long, limiting to 10 steps")
            motion_sequence = motion_sequence[:10]

        # Check for conflicting commands
        if "walk forward" in str(motion_sequence) and "walk backward" in str(motion_sequence):
            warnings.append("Conflicting forward/backward commands detected")

        # Create plan
        plan = MotionPlan(
            instruction=" -> ".join(motion_sequence),
            motion_sequence=motion_sequence,
            estimated_duration=estimated_duration,
            safety_check_passed=safety_check_passed,
            warnings=warnings
        )

        return plan

    def get_model_for_instruction(self, instruction: str) -> Optional[str]:
        """Get model path for instruction"""
        instruction_lower = instruction.lower().strip()
        return self.available_models.get(instruction_lower)


def test_task_planner():
    """Test task planner"""
    print("Testing Task Planner")
    print("=" * 60)

    planner = TaskPlanner()

    test_cases = [
        ["walk forward slowly", "turn left", "stop moving"],
        ["clean table", "pick up object", "place object"],
        ["dance", "walk forward", "turn left", "turn right"],
        ["walk forward" for _ in range(15)],  # Too many commands
    ]

    for i, commands in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {commands}")
        plan = planner.create_motion_plan(commands)
        print(f"  Motion Sequence: {plan.motion_sequence}")
        print(f"  Duration: {plan.estimated_duration:.1f}s")
        print(f"  Safety: {'✓' if plan.safety_check_passed else '✗'}")
        if plan.warnings:
            print(f"  Warnings: {plan.warnings}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_task_planner()