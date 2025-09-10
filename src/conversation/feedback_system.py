#!/usr/bin/env python3
"""
Feedback System for Conversational Robotics
Handles user feedback and learns preferences
"""

import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class FeedbackEntry:
    """Single feedback entry"""
    task_description: str
    motion_sequence: List[str]
    user_feedback: str
    sentiment: str  # "positive", "negative", "neutral"
    execution_metrics: Dict
    timestamp: float
    improvement_suggestions: List[str]


@dataclass
class PreferenceProfile:
    """User preference profile"""
    preferred_speed: str  # "slow", "normal", "fast"
    preferred_style: str  # "gentle", "normal", "energetic"
    disliked_actions: List[str]
    favorite_actions: List[str]
    safety_sensitivity: float  # 0.0 to 1.0
    feedback_count: int
    success_rate: float


class FeedbackSystem:
    """System for processing and learning from user feedback"""

    def __init__(self, feedback_file: str = "user_feedback.json"):
        self.feedback_file = Path(feedback_file)
        self.feedback_history: List[FeedbackEntry] = []
        self.user_preferences = PreferenceProfile(
            preferred_speed="normal",
            preferred_style="normal",
            disliked_actions=[],
            favorite_actions=[],
            safety_sensitivity=0.5,
            feedback_count=0,
            success_rate=0.0
        )

        # Load existing feedback
        self._load_feedback_history()

        # Sentiment keywords
        self.positive_keywords = [
            "good", "great", "excellent", "perfect", "nice", "smooth", "like", "love",
            "awesome", "fantastic", "wonderful", "amazing", "brilliant", "well done"
        ]

        self.negative_keywords = [
            "bad", "terrible", "awful", "hate", "dislike", "wrong", "rough", "aggressive",
            "too fast", "too slow", "jerky", "unnatural", "scary", "dangerous"
        ]

        self.speed_keywords = {
            "slow": ["slow", "slower", "gentle", "careful", "gradual"],
            "fast": ["fast", "faster", "quick", "rapid", "speed up", "hurry"]
        }

        self.style_keywords = {
            "gentle": ["gentle", "soft", "smooth", "careful", "delicate"],
            "energetic": ["energetic", "dynamic", "vigorous", "lively", "active"]
        }

        print("Feedback System initialized")
        print(f"Loaded {len(self.feedback_history)} previous feedback entries")

    def process_feedback(self,
                         task_description: str,
                         motion_sequence: List[str],
                         user_feedback: str,
                         execution_metrics: Dict = None) -> FeedbackEntry:
        """Process user feedback and update preferences"""

        if execution_metrics is None:
            execution_metrics = {}

        print(f"\nProcessing feedback: '{user_feedback}'")

        # Analyze sentiment
        sentiment = self._analyze_sentiment(user_feedback)

        # Extract improvement suggestions
        suggestions = self._extract_improvement_suggestions(user_feedback, motion_sequence)

        # Create feedback entry
        entry = FeedbackEntry(
            task_description=task_description,
            motion_sequence=motion_sequence,
            user_feedback=user_feedback,
            sentiment=sentiment,
            execution_metrics=execution_metrics,
            timestamp=time.time(),
            improvement_suggestions=suggestions
        )

        # Add to history
        self.feedback_history.append(entry)

        # Update user preferences
        self._update_preferences(entry)

        # Save feedback
        self._save_feedback_history()

        print(f"Feedback processed - Sentiment: {sentiment}")
        print(f"Suggestions: {suggestions}")

        return entry

    def _analyze_sentiment(self, feedback_text: str) -> str:
        """Analyze sentiment of feedback text"""
        feedback_lower = feedback_text.lower()

        positive_score = sum(1 for word in self.positive_keywords if word in feedback_lower)
        negative_score = sum(1 for word in self.negative_keywords if word in feedback_lower)

        if positive_score > negative_score:
            return "positive"
        elif negative_score > positive_score:
            return "negative"
        else:
            return "neutral"

    def _extract_improvement_suggestions(self, feedback_text: str, motion_sequence: List[str]) -> List[str]:
        """Extract actionable improvement suggestions from feedback"""
        suggestions = []
        feedback_lower = feedback_text.lower()

        # Speed adjustments
        for speed, keywords in self.speed_keywords.items():
            if any(keyword in feedback_lower for keyword in keywords):
                if speed == "slow":
                    suggestions.append("reduce_movement_speed")
                elif speed == "fast":
                    suggestions.append("increase_movement_speed")

        # Style adjustments
        for style, keywords in self.style_keywords.items():
            if any(keyword in feedback_lower for keyword in keywords):
                suggestions.append(f"adjust_style_to_{style}")

        # Specific action feedback
        if "turning" in feedback_lower and any("too" in feedback_lower for word in ["fast", "sharp", "aggressive"]):
            suggestions.append("smoother_turning")

        if "walking" in feedback_lower and "jerky" in feedback_lower:
            suggestions.append("smoother_walking")

        # Safety concerns
        if any(word in feedback_lower for word in ["scary", "dangerous", "unsafe"]):
            suggestions.append("increase_safety_margin")

        # If no specific suggestions, provide general ones based on sentiment
        if not suggestions:
            if "negative" in self._analyze_sentiment(feedback_text):
                suggestions.append("review_motion_parameters")
            else:
                suggestions.append("maintain_current_approach")

        return suggestions

    def _update_preferences(self, entry: FeedbackEntry):
        """Update user preference profile based on feedback"""
        feedback_lower = entry.user_feedback.lower()

        # Update feedback count
        self.user_preferences.feedback_count += 1

        # Update success rate (simple moving average)
        if entry.sentiment == "positive":
            success = 1.0
        elif entry.sentiment == "negative":
            success = 0.0
        else:
            success = 0.5

        alpha = 0.1  # Learning rate
        self.user_preferences.success_rate = (
                (1 - alpha) * self.user_preferences.success_rate + alpha * success
        )

        # Update speed preferences
        for speed, keywords in self.speed_keywords.items():
            if any(keyword in feedback_lower for keyword in keywords):
                if entry.sentiment == "positive":
                    self.user_preferences.preferred_speed = speed

        # Update style preferences
        for style, keywords in self.style_keywords.items():
            if any(keyword in feedback_lower for keyword in keywords):
                if entry.sentiment == "positive":
                    self.user_preferences.preferred_style = style

        # Update action preferences
        if entry.sentiment == "positive":
            for action in entry.motion_sequence:
                if action not in self.user_preferences.favorite_actions:
                    self.user_preferences.favorite_actions.append(action)
        elif entry.sentiment == "negative":
            for action in entry.motion_sequence:
                if action not in self.user_preferences.disliked_actions:
                    self.user_preferences.disliked_actions.append(action)

        # Update safety sensitivity
        if any(word in feedback_lower for word in ["scary", "dangerous", "unsafe"]):
            self.user_preferences.safety_sensitivity = min(1.0, self.user_preferences.safety_sensitivity + 0.1)
        elif any(word in feedback_lower for word in ["boring", "too careful", "too slow"]):
            self.user_preferences.safety_sensitivity = max(0.0, self.user_preferences.safety_sensitivity - 0.1)

    def get_personalized_recommendations(self, proposed_actions: List[str]) -> Dict:
        """Get personalized recommendations for proposed actions"""
        recommendations = {
            "modified_actions": proposed_actions.copy(),
            "warnings": [],
            "suggested_improvements": [],
            "confidence": 0.5
        }

        # Check against disliked actions
        for i, action in enumerate(proposed_actions):
            if action in self.user_preferences.disliked_actions:
                recommendations["warnings"].append(f"User previously disliked '{action}'")
                # Try to substitute with preferred action
                if self.user_preferences.favorite_actions:
                    alternative = np.random.choice(self.user_preferences.favorite_actions)
                    recommendations["modified_actions"][i] = alternative
                    recommendations["suggested_improvements"].append(f"Replaced '{action}' with '{alternative}'")

        # Apply speed preferences
        if self.user_preferences.preferred_speed == "slow":
            recommendations["suggested_improvements"].append("Consider reducing movement speed")
        elif self.user_preferences.preferred_speed == "fast":
            recommendations["suggested_improvements"].append("Consider increasing movement speed")

        # Apply style preferences
        if self.user_preferences.preferred_style == "gentle":
            recommendations["suggested_improvements"].append("Use gentler, smoother movements")
        elif self.user_preferences.preferred_style == "energetic":
            recommendations["suggested_improvements"].append("Use more dynamic, energetic movements")

        # Safety considerations
        if self.user_preferences.safety_sensitivity > 0.7:
            recommendations["suggested_improvements"].append("Prioritize safety and smooth movements")

        # Calculate confidence based on feedback history
        if self.user_preferences.feedback_count > 5:
            recommendations["confidence"] = min(0.9, 0.5 + self.user_preferences.feedback_count * 0.05)

        return recommendations

    def get_feedback_summary(self) -> Dict:
        """Get summary of all feedback received"""
        if not self.feedback_history:
            return {"message": "No feedback received yet"}

        total_feedback = len(self.feedback_history)
        positive_feedback = sum(1 for entry in self.feedback_history if entry.sentiment == "positive")
        negative_feedback = sum(1 for entry in self.feedback_history if entry.sentiment == "negative")

        # Most common suggestions
        all_suggestions = []
        for entry in self.feedback_history:
            all_suggestions.extend(entry.improvement_suggestions)

        suggestion_counts = {}
        for suggestion in all_suggestions:
            suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1

        top_suggestions = sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "success_rate": f"{positive_feedback / total_feedback * 100:.1f}%",
            "user_preferences": asdict(self.user_preferences),
            "top_improvement_suggestions": top_suggestions,
            "recent_feedback": [entry.user_feedback for entry in self.feedback_history[-5:]]
        }

    def _load_feedback_history(self):
        """Load feedback history from file"""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)

                # Load feedback entries
                for entry_data in data.get("feedback_history", []):
                    entry = FeedbackEntry(**entry_data)
                    self.feedback_history.append(entry)

                # Load user preferences
                if "user_preferences" in data:
                    pref_data = data["user_preferences"]
                    self.user_preferences = PreferenceProfile(**pref_data)

            except Exception as e:
                print(f"Error loading feedback history: {e}")

    def _save_feedback_history(self):
        """Save feedback history to file"""
        try:
            data = {
                "feedback_history": [asdict(entry) for entry in self.feedback_history],
                "user_preferences": asdict(self.user_preferences),
                "last_updated": time.time()
            }

            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            print(f"Error saving feedback history: {e}")


def test_feedback_system():
    """Test the feedback system"""
    print("Testing Feedback System")
    print("=" * 40)

    feedback_sys = FeedbackSystem("test_feedback.json")

    # Test scenarios
    test_cases = [
        {
            "task": "Walk forward",
            "actions": ["walk forward"],
            "feedback": "Great job! Very smooth movement.",
            "metrics": {"similarity": 0.8, "duration": 10}
        },
        {
            "task": "Turn around",
            "actions": ["turn left", "turn left"],
            "feedback": "Too fast and jerky, please be more gentle.",
            "metrics": {"similarity": 0.6, "duration": 8}
        },
        {
            "task": "Dance",
            "actions": ["walk forward", "turn left", "turn right"],
            "feedback": "I love the dancing! Maybe a bit faster next time.",
            "metrics": {"similarity": 0.7, "duration": 15}
        },
        {
            "task": "Clean table",
            "actions": ["walk forward", "reach forward", "wipe surface"],
            "feedback": "Good cleaning, but the movement was scary fast.",
            "metrics": {"similarity": 0.5, "duration": 20}
        }
    ]

    # Process test feedback
    for test_case in test_cases:
        print(f"\n{'=' * 50}")
        entry = feedback_sys.process_feedback(
            test_case["task"],
            test_case["actions"],
            test_case["feedback"],
            test_case["metrics"]
        )

    # Test recommendations
    print(f"\n{'=' * 50}")
    print("Testing Recommendations:")

    proposed_actions = ["walk forward", "turn left", "jump up"]
    recommendations = feedback_sys.get_personalized_recommendations(proposed_actions)

    print(f"Original actions: {proposed_actions}")
    print(f"Recommended actions: {recommendations['modified_actions']}")
    print(f"Warnings: {recommendations['warnings']}")
    print(f"Improvements: {recommendations['suggested_improvements']}")
    print(f"Confidence: {recommendations['confidence']:.2f}")

    # Print summary
    print(f"\n{'=' * 50}")
    print("Feedback Summary:")
    summary = feedback_sys.get_feedback_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_feedback_system()