#!/usr/bin/env python3
"""
Feedback System - Learns from user preferences (like/dislike)
"""

import json
import time
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeedbackEntry:
    """Single feedback entry"""
    timestamp: float
    user_input: str
    commands_executed: List[str]
    feedback_type: str  # "like", "dislike", "neutral"
    feedback_text: str
    performance_score: float


class FeedbackSystem:
    """
    Learns from user feedback to improve motion execution
    """

    def __init__(self, save_path: str = "./feedback_history.json"):
        self.save_path = Path(save_path)
        self.feedback_history: List[FeedbackEntry] = []
        self.preference_scores: Dict[str, List[float]] = {}

        # Load existing history
        self._load_history()

    def record_feedback(self,
                        user_input: str,
                        commands_executed: List[str],
                        feedback_text: str) -> FeedbackEntry:
        """
        Record user feedback
        """
        # Analyze sentiment
        feedback_type, score = self._analyze_sentiment(feedback_text)

        entry = FeedbackEntry(
            timestamp=time.time(),
            user_input=user_input,
            commands_executed=commands_executed,
            feedback_type=feedback_type,
            feedback_text=feedback_text,
            performance_score=score
        )

        self.feedback_history.append(entry)

        # Update preference scores
        for cmd in commands_executed:
            if cmd not in self.preference_scores:
                self.preference_scores[cmd] = []
            self.preference_scores[cmd].append(score)

        # Save to disk
        self._save_history()

        return entry

    def _analyze_sentiment(self, feedback_text: str) -> tuple:
        """
        Analyze feedback sentiment
        Returns: (feedback_type, score)
        """
        feedback_lower = feedback_text.lower()

        # Positive indicators
        positive_words = [
            "good", "great", "excellent", "perfect", "nice", "love",
            "amazing", "awesome", "yes", "correct", "right", "better"
        ]

        # Negative indicators
        negative_words = [
            "bad", "wrong", "no", "terrible", "awful", "hate",
            "stop", "incorrect", "worse", "rough", "aggressive", "too fast"
        ]

        positive_count = sum(1 for word in positive_words if word in feedback_lower)
        negative_count = sum(1 for word in negative_words if word in feedback_lower)

        if positive_count > negative_count:
            return "like", 1.0
        elif negative_count > positive_count:
            return "dislike", -1.0
        else:
            return "neutral", 0.0

    def get_command_preference(self, command: str) -> float:
        """
        Get average preference score for a command
        Returns score in range [-1, 1]
        """
        if command not in self.preference_scores:
            return 0.0

        scores = self.preference_scores[command]
        return sum(scores) / len(scores) if scores else 0.0

    def get_recommendations(self) -> List[str]:
        """
        Get recommendations based on feedback
        """
        recommendations = []

        # Find commands with negative scores
        for cmd, scores in self.preference_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < -0.5:
                recommendations.append(f"Consider improving: {cmd} (avg score: {avg_score:.2f})")

        # Find highly rated commands
        for cmd, scores in self.preference_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > 0.7:
                recommendations.append(f"Keep doing: {cmd} (avg score: {avg_score:.2f})")

        if not recommendations:
            recommendations.append("Not enough feedback data yet")

        return recommendations

    def _save_history(self):
        """Save feedback history to JSON"""
        try:
            data = {
                "feedback_history": [
                    {
                        "timestamp": e.timestamp,
                        "user_input": e.user_input,
                        "commands_executed": e.commands_executed,
                        "feedback_type": e.feedback_type,
                        "feedback_text": e.feedback_text,
                        "performance_score": e.performance_score
                    }
                    for e in self.feedback_history
                ],
                "preference_scores": self.preference_scores
            }

            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Failed to save feedback history: {e}")

    def _load_history(self):
        """Load feedback history from JSON"""
        try:
            if self.save_path.exists():
                with open(self.save_path, 'r') as f:
                    data = json.load(f)

                self.feedback_history = [
                    FeedbackEntry(**entry) for entry in data.get("feedback_history", [])
                ]
                self.preference_scores = data.get("preference_scores", {})

        except Exception as e:
            print(f"Failed to load feedback history: {e}")


def test_feedback_system():
    """Test feedback system"""
    print("Testing Feedback System")
    print("=" * 60)

    feedback = FeedbackSystem(save_path="./test_feedback.json")

    # Test cases
    test_feedbacks = [
        ("walk forward", ["walk forward"], "That was great!"),
        ("turn left", ["turn left"], "Perfect turn"),
        ("walk forward", ["walk forward quickly"], "Too fast, slow down"),
        ("clean table", ["walk forward slowly"], "Good cleaning motion"),
        ("dance", ["walk forward", "turn left", "turn right"], "Not smooth enough"),
    ]

    for user_input, commands, feedback_text in test_feedbacks:
        entry = feedback.record_feedback(user_input, commands, feedback_text)
        print(f"\nFeedback: '{feedback_text}'")
        print(f"  Type: {entry.feedback_type}")
        print(f"  Score: {entry.performance_score}")

    print("\n" + "=" * 60)
    print("Recommendations:")
    for rec in feedback.get_recommendations():
        print(f"  • {rec}")

    print("\nCommand Preferences:")
    for cmd in ["walk forward", "turn left", "walk forward quickly"]:
        score = feedback.get_command_preference(cmd)
        print(f"  {cmd}: {score:.2f}")


if __name__ == "__main__":
    test_feedback_system()