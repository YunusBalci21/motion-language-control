#!/usr/bin/env python3
"""
Task Planner for Conversational Robotics
Converts LLM commands into executable motion sequences
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MotionStep:
    """Single motion step in a plan"""
    instruction: str
    duration: float
    priority: float
    expected_outcome: str


@dataclass
class ExecutionPlan:
    """Complete execution plan for a task"""
    task_description: str
    motion_steps: List[MotionStep]
    total_duration: float
    safety_level: str
    fallback_plan: Optional[List[str]] = None


class TaskPlanner:
    """Plans and sequences motion commands from LLM responses"""

    def __init__(self):
        # ENHANCED: Map LLM commands to motion system instructions
        self.command_mapping = {
            # Basic locomotion
            "walk forward": "walk forward",
            "walk backward": "walk backward",
            "walk back": "walk backward",
            "turn left": "turn left",
            "turn right": "turn right",
            "stop": "stop moving",
            "stop moving": "stop moving",

            # Advanced movements
            "walk left": "turn left",
            "walk right": "turn right",
            "move forward": "walk forward",
            "go forward": "walk forward",
            "go back": "walk backward",
            "go backward": "walk backward",

            # ENHANCED: Complex cleaning/manipulation actions
            "reach forward": ["walk forward", "walk forward"],  # Approach then reach
            "reach up": ["walk forward", "stop moving"],  # Approach then stop
            "crouch down": ["stop moving", "stop moving"],  # Double stop for stability
            "stand up": ["walk forward", "stop moving"],  # Movement then stabilize
            "wave hand": ["turn left", "turn right", "turn left"],  # Wave motion
            "wipe surface": ["turn left", "turn right", "turn left", "turn right"],  # Wiping motion
            "grasp object": ["walk forward", "stop moving"],  # Approach then stabilize
            "release object": ["turn left", "turn right"],  # Release motion
            "pick up object": ["walk forward", "stop moving", "turn left"],  # Pick up sequence
            "place object": ["walk forward", "turn right", "stop moving"],  # Place sequence

            # ENHANCED: Cleaning-specific commands
            "clean table": ["walk forward", "turn left", "turn right", "turn left", "turn right"],
            "clean surface": ["turn left", "turn right", "turn left", "turn right"],
            "organize items": ["walk forward", "turn left", "walk forward", "turn right"],
            "move object": ["walk forward", "turn left", "walk backward"],
            "clean the table": ["walk forward", "turn left", "turn right", "turn left", "turn right"],
            "wipe the surface": ["turn left", "turn right", "turn left", "turn right"],
            "pick up that object": ["walk forward", "stop moving", "turn left"],
            "place it over there": ["walk forward", "turn right", "stop moving"],
            "organize the items": ["walk forward", "turn left", "walk forward", "turn right"],
            "reach for the object": ["walk forward", "walk forward"],
            "grasp it carefully": ["walk forward", "stop moving"],
            "move it slowly": ["walk forward", "turn left", "walk backward"],
            "clean thoroughly": ["turn left", "turn right", "turn left", "turn right", "turn left"],
            "tidy up the area": ["walk forward", "turn left", "walk forward", "turn right", "walk backward"],

            # Dance moves (existing)
            "dance": ["walk forward", "turn left", "turn right", "walk backward"],
            "spin": ["turn left", "turn left", "turn left", "turn left"],
        }

        # ENHANCED: Default durations for each instruction type (in steps)
        self.default_durations = {
            "walk forward": 200,
            "walk backward": 200,
            "turn left": 150,
            "turn right": 150,
            "stop moving": 50,
        }

        # ENHANCED: Safety classifications
        self.safety_levels = {
            "low": ["stop moving"],
            "medium": ["walk forward", "walk backward"],
            "high": ["turn left", "turn right"],
            "very_high": ["dance", "spin"]
        }

        print("Enhanced Task Planner initialized with manipulation support")

    def create_execution_plan(self, llm_commands: List[str], task_description: str = "") -> ExecutionPlan:
        """Create detailed execution plan from LLM commands"""
        motion_steps = []
        total_duration = 0.0

        print(f"\nPlanning task: '{task_description}'")
        print(f"LLM Commands: {llm_commands}")

        for i, command in enumerate(llm_commands):
            steps = self._process_single_command(command, i)
            motion_steps.extend(steps)
            total_duration += sum(step.duration for step in steps)

        # Determine safety level
        safety_level = self._assess_safety_level(motion_steps)

        # Create fallback plan
        fallback_plan = self._create_fallback_plan(llm_commands)

        plan = ExecutionPlan(
            task_description=task_description,
            motion_steps=motion_steps,
            total_duration=total_duration,
            safety_level=safety_level,
            fallback_plan=fallback_plan
        )

        print(f"Created plan with {len(motion_steps)} steps, duration: {total_duration} steps")
        self._print_plan(plan)

        return plan

    def _process_single_command(self, command: str, sequence_index: int) -> List[MotionStep]:
        """Process a single LLM command into motion steps"""
        command = command.lower().strip()

        # Check if command maps to multiple actions
        if command in self.command_mapping:
            mapped = self.command_mapping[command]
            if isinstance(mapped, list):
                # Command maps to sequence of actions
                steps = []
                for j, sub_command in enumerate(mapped):
                    duration = self.default_durations.get(sub_command, 100)
                    steps.append(MotionStep(
                        instruction=sub_command,
                        duration=duration,
                        priority=1.0 - (sequence_index * 0.1) - (j * 0.05),
                        # Later sub-commands have slightly lower priority
                        expected_outcome=f"Execute {sub_command} (part of {command})"
                    ))
                return steps
            else:
                # Command maps to single action
                duration = self.default_durations.get(mapped, 100)
                return [MotionStep(
                    instruction=mapped,
                    duration=duration,
                    priority=1.0 - (sequence_index * 0.1),
                    expected_outcome=f"Execute {mapped}"
                )]

        # Try to find best match for unmapped commands
        best_match = self._find_best_match(command)
        duration = self.default_durations.get(best_match, 100)

        return [MotionStep(
            instruction=best_match,
            duration=duration,
            priority=0.5,  # Lower priority for uncertain matches
            expected_outcome=f"Attempt {best_match} (from '{command}')"
        )]

    def _find_best_match(self, command: str) -> str:
        """ENHANCED: Find best matching instruction for unknown command"""
        command_words = command.split()

        # Enhanced keyword matching for manipulation tasks
        if any(word in command for word in ["clean", "wipe", "surface", "table"]):
            return "turn left"  # Use turning for cleaning motions
        elif any(word in command for word in ["pick", "grasp", "grab", "take"]):
            return "walk forward"  # Approach for picking up
        elif any(word in command for word in ["place", "put", "drop", "release"]):
            return "turn right"  # Turn for placing
        elif any(word in command for word in ["reach", "extend", "stretch"]):
            return "walk forward"  # Approach for reaching
        elif any(word in command for word in ["organize", "tidy", "arrange"]):
            return "walk forward"  # Move around for organizing
        elif any(word in command for word in ["forward", "ahead", "front"]):
            return "walk forward"
        elif any(word in command for word in ["backward", "back", "behind"]):
            return "walk backward"
        elif any(word in command for word in ["left"]):
            return "turn left"
        elif any(word in command for word in ["right"]):
            return "turn right"
        elif any(word in command for word in ["stop", "halt", "pause", "wait"]):
            return "stop moving"
        else:
            # Default fallback
            return "walk forward"

    def _assess_safety_level(self, motion_steps: List[MotionStep]) -> str:
        """Assess safety level of the execution plan"""
        risk_score = 0

        for step in motion_steps:
            if step.instruction in self.safety_levels["very_high"]:
                risk_score += 3
            elif step.instruction in self.safety_levels["high"]:
                risk_score += 2
            elif step.instruction in self.safety_levels["medium"]:
                risk_score += 1

        if risk_score == 0:
            return "low"
        elif risk_score <= 2:
            return "medium"
        elif risk_score <= 5:
            return "high"
        else:
            return "very_high"

    def _create_fallback_plan(self, original_commands: List[str]) -> List[str]:
        """Create simple fallback plan in case main plan fails"""
        if any("stop" in cmd.lower() for cmd in original_commands):
            return ["stop moving"]
        else:
            return ["walk forward", "stop moving"]

    def _print_plan(self, plan: ExecutionPlan):
        """Print execution plan for debugging"""
        print("\n--- Enhanced Execution Plan ---")
        for i, step in enumerate(plan.motion_steps):
            print(f"  {i + 1}. {step.instruction} ({step.duration} steps, priority: {step.priority:.2f})")
        print(f"Total duration: {plan.total_duration} steps")
        print(f"Safety level: {plan.safety_level}")
        print("--------------------------------")

    def optimize_plan(self, plan: ExecutionPlan, constraints: Dict = None) -> ExecutionPlan:
        """Optimize execution plan based on constraints"""
        if constraints is None:
            constraints = {}

        max_duration = constraints.get("max_duration", float('inf'))
        max_complexity = constraints.get("max_complexity", 10)

        optimized_steps = plan.motion_steps.copy()

        # Remove low-priority steps if over duration limit
        if plan.total_duration > max_duration:
            optimized_steps.sort(key=lambda x: x.priority, reverse=True)
            optimized_steps = optimized_steps[:max_complexity]

        # Merge consecutive identical instructions
        merged_steps = []
        i = 0
        while i < len(optimized_steps):
            current_step = optimized_steps[i]
            merged_duration = current_step.duration

            # Look for consecutive identical instructions
            j = i + 1
            while j < len(optimized_steps) and optimized_steps[j].instruction == current_step.instruction:
                merged_duration += optimized_steps[j].duration
                j += 1

            # Create merged step
            merged_step = MotionStep(
                instruction=current_step.instruction,
                duration=merged_duration,
                priority=current_step.priority,
                expected_outcome=current_step.expected_outcome
            )
            merged_steps.append(merged_step)
            i = j

        # Create optimized plan
        optimized_plan = ExecutionPlan(
            task_description=plan.task_description + " (optimized)",
            motion_steps=merged_steps,
            total_duration=sum(step.duration for step in merged_steps),
            safety_level=plan.safety_level,
            fallback_plan=plan.fallback_plan
        )

        print(f"Optimized plan: {len(plan.motion_steps)} → {len(merged_steps)} steps")
        return optimized_plan

    def get_next_instruction(self, plan: ExecutionPlan, current_step: int) -> Optional[Tuple[str, float]]:
        """Get next instruction from execution plan"""
        if current_step >= len(plan.motion_steps):
            return None

        step = plan.motion_steps[current_step]
        return step.instruction, step.duration


def test_task_planner():
    """Test the enhanced task planner"""
    print("Testing Enhanced Task Planner")
    print("=" * 40)

    planner = TaskPlanner()

    test_scenarios = [
        {
            "task": "Clean the table",
            "commands": ["walk forward", "reach forward", "grasp object", "turn left", "release object"]
        },
        {
            "task": "Enhanced cleaning",
            "commands": ["clean the table", "wipe the surface", "organize items"]
        },
        {
            "task": "Manipulation sequence",
            "commands": ["pick up that object", "place it over there", "tidy up the area"]
        },
        {
            "task": "Simple walk",
            "commands": ["walk forward"]
        },
        {
            "task": "Dance routine",
            "commands": ["dance", "turn left", "turn right", "spin"]
        }
    ]

    for scenario in test_scenarios:
        print(f"\n{'=' * 50}")
        plan = planner.create_execution_plan(scenario["commands"], scenario["task"])

        # Test optimization
        optimized = planner.optimize_plan(plan, {"max_duration": 300, "max_complexity": 5})
        print(f"Original: {len(plan.motion_steps)} steps, Optimized: {len(optimized.motion_steps)} steps")


if __name__ == "__main__":
    test_task_planner()