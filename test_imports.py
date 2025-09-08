#!/usr/bin/env python3
"""
Simple test to verify imports work correctly
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))


def test_motion_tokenizer():
    """Test MotionTokenizer import"""
    try:
        from models.motion_tokenizer import MotionTokenizer
        print("✓ MotionTokenizer import successful")

        # Test creation
        tokenizer = MotionTokenizer()
        print("✓ MotionTokenizer creation successful")
        return True
    except Exception as e:
        print(f"✗ MotionTokenizer failed: {e}")
        return False


def test_agent():
    """Test MotionLanguageAgent import"""
    try:
        from agents.hierarchical_agent import MotionLanguageAgent
        print("✓ MotionLanguageAgent import successful")
        return True
    except Exception as e:
        print(f"✗ MotionLanguageAgent failed: {e}")
        return False


def test_wrapper():
    """Test DirectMotionLanguageWrapper import"""
    try:
        from agents.hierarchical_agent import DirectMotionLanguageWrapper
        print("✓ DirectMotionLanguageWrapper import successful")
        return True
    except Exception as e:
        print(f"✗ DirectMotionLanguageWrapper failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing project imports...")
    print("=" * 40)

    results = []
    results.append(test_motion_tokenizer())
    results.append(test_agent())
    results.append(test_wrapper())

    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All imports working!")
        print("\nYou can now run:")
        print("  python train_motion_language_agent.py --quick-test --curriculum basic")
    else:
        print("❌ Some imports failed. Check errors above.")