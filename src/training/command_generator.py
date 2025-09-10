#!/usr/bin/env python3
"""
Self-Generating Command System for Motion-Language Learning
Generates diverse training instructions based on success feedback
"""

import random
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CommandTemplate:
    """Template for generating commands"""
    base_action: str
    modifiers: List[str]
    success_weight: float = 1.0


class InstructionGenerator:
    """Generates training instructions dynamically based on success rates"""

    def __init__(self):
        # Base movement commands
        self.base_actions = [
            "walk forward", "walk backward", "turn left", "turn right",
            "stop moving", "jump", "crouch", "stand up"
        ]

        # Stability modifiers (as suggested by your teacher)
        self.stability_modifiers = [
            "stably", "without falling", "carefully", "smoothly",
            "with good balance", "steadily", "with good posture",
            "without stumbling", "maintaining balance", "gracefully"
        ]

        # Speed modifiers
        self.speed_modifiers = [
            "slowly", "quickly", "fast", "at normal speed",
            "at a steady pace", "gradually"
        ]

        # Duration modifiers
        self.duration_modifiers = [
            "for 5 steps", "for 10 steps", "briefly", "continuously",
            "for a few seconds", "until I say stop"
        ]

        # Success tracking for each command type
        self.command_success_rates: Dict[str, List[float]] = {}
        self.generated_commands: List[str] = []

        print("Instruction Generator initialized with dynamic command generation!")

    def generate_diverse_commands(self, num_commands: int = 20,
                                  focus_on_stability: bool = True) -> List[str]:
        """Generate diverse training commands with focus on stability"""

        commands = []

        # Generate combinations
        for _ in range(num_commands):
            base_action = random.choice(self.base_actions)

            # Bias towards stability modifiers as suggested
            if focus_on_stability and random.random() < 0.6:
                modifier = random.choice(self.stability_modifiers)
                command = f"{base_action} {modifier}"
            else:
                # Mix other modifiers
                modifier_type = random.choice(['speed', 'duration', 'stability'])

                if modifier_type == 'speed':
                    modifier = random.choice(self.speed_modifiers)
                elif modifier_type == 'duration':
                    modifier = random.choice(self.duration_modifiers)
                else:
                    modifier = random.choice(self.stability_modifiers)

                command = f"{base_action} {modifier}"

            if command not in commands:
                commands.append(command)

        # Add some base commands without modifiers
        for base_action in self.base_actions[:5]:
            if base_action not in commands:
                commands.append(base_action)

        self.generated_commands.extend(commands)
        print(f"Generated {len(commands)} diverse training commands")
        return commands

    def update_success_rate(self, command: str, success_rate: float):
        """Update success rate for a command"""
        if command not in self.command_success_rates:
            self.command_success_rates[command] = []

        self.command_success_rates[command].append(success_rate)

        # Keep only last 10 results
        if len(self.command_success_rates[command]) > 10:
            self.command_success_rates[command] = self.command_success_rates[command][-10:]

    def get_underperforming_commands(self, threshold: float = 0.5) -> List[str]:
        """Get commands that need more training"""
        underperforming = []

        for command, success_rates in self.command_success_rates.items():
            if len(success_rates) >= 3:  # Need at least 3 attempts
                avg_success = np.mean(success_rates)
                if avg_success < threshold:
                    underperforming.append(command)

        return underperforming

    def generate_variations_of_command(self, base_command: str, num_variations: int = 5) -> List[str]:
        """Generate variations of a specific command that's underperforming"""

        variations = [base_command]  # Include original

        # Extract base action
        base_action = base_command.split()[0] + " " + base_command.split()[1]  # "walk forward" etc

        # Generate variations with different modifiers
        for _ in range(num_variations - 1):
            modifier = random.choice(self.stability_modifiers + self.speed_modifiers)
            variation = f"{base_action} {modifier}"

            if variation not in variations:
                variations.append(variation)

        return variations

    def get_adaptive_training_curriculum(self, current_success_rates: Dict[str, float]) -> List[str]:
        """Generate adaptive curriculum based on current performance"""

        # Update our success tracking
        for command, success_rate in current_success_rates.items():
            self.update_success_rate(command, success_rate)

        # Get underperforming commands
        underperforming = self.get_underperforming_commands(threshold=0.4)

        training_commands = []

        # Focus more on underperforming commands
        for command in underperforming[:5]:  # Top 5 underperforming
            variations = self.generate_variations_of_command(command, 3)
            training_commands.extend(variations)

        # Add some new diverse commands
        new_commands = self.generate_diverse_commands(10, focus_on_stability=True)
        training_commands.extend(new_commands[:5])

        # Add some high-performing commands for stability
        high_performing = []
        for command, success_rates in self.command_success_rates.items():
            if len(success_rates) >= 3 and np.mean(success_rates) > 0.7:
                high_performing.append(command)

        if high_performing:
            training_commands.extend(random.sample(high_performing, min(3, len(high_performing))))

        return list(set(training_commands))  # Remove duplicates


def test_instruction_generator():
    """Test the instruction generator"""
    print("Testing Instruction Generator")
    print("=" * 40)

    generator = InstructionGenerator()

    # Generate diverse commands
    commands = generator.generate_diverse_commands(15, focus_on_stability=True)

    print("\nGenerated Commands:")
    for i, command in enumerate(commands, 1):
        print(f"{i:2d}. {command}")

    # Simulate some success rates
    fake_success_rates = {
        "walk forward": 0.8,
        "walk forward stably": 0.6,
        "turn left": 0.3,  # Underperforming
        "walk backward": 0.2,  # Very underperforming
        "stop moving": 0.9
    }

    # Get adaptive curriculum
    adaptive_commands = generator.get_adaptive_training_curriculum(fake_success_rates)

    print(f"\nAdaptive Training Curriculum ({len(adaptive_commands)} commands):")
    for i, command in enumerate(adaptive_commands, 1):
        print(f"{i:2d}. {command}")

    print("\nInstruction Generator test completed!")


if __name__ == "__main__":
    test_instruction_generator()