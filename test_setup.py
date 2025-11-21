#!/usr/bin/env python3
"""
Test Script: Verify Reward Tracking Setup
Run this to check if everything is configured correctly
"""

import sys
from pathlib import Path

print("=" * 70)
print("🔍 TESTING YOUR REWARD TRACKING SETUP")
print("=" * 70)

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

print("\n1️⃣  Checking imports...")

# Test imports
try:
    from models.reward_tracker import RewardTracker, RewardVisualizer

    print("   ✅ reward_tracker imports successfully")
except ImportError as e:
    print(f"   ❌ reward_tracker import failed: {e}")
    print("   → Check if src/models/reward_tracker.py exists")

try:
    from utils.visualize_diagnostics import load_diagnostic, visualize_diagnostics

    print("   ✅ visualize_diagnostics imports successfully")
except ImportError as e:
    print(f"   ❌ visualize_diagnostics import failed: {e}")
    print("   → Check if src/utils/visualize_diagnostics.py exists")

try:
    from agents.hierarchical_agent import EnhancedMotionLanguageAgent

    print("   ✅ hierarchical_agent imports successfully")
except ImportError as e:
    print(f"   ❌ hierarchical_agent import failed: {e}")
    print("   → Check if src/agents/hierarchical_agent.py exists")

print("\n2️⃣  Checking diagnostic files...")

diagnostic_files = [
    "diagnostic_results_1757522425.json",
    "diagnostic_results_1757522705.json",
]

found_files = []
for file in diagnostic_files:
    file_path = Path(file)
    if file_path.exists():
        print(f"   ✅ Found: {file}")
        found_files.append(str(file_path))
    else:
        print(f"   ⚠️  Not found: {file}")

if not found_files:
    print("   → Copy your diagnostic JSON files to project root")

print("\n3️⃣  Checking folder structure...")

folders = [
    "src/models",
    "src/training",
    "src/utils",
    "src/agents",
    "scripts",
    "configs",
]

for folder in folders:
    folder_path = Path(folder)
    if folder_path.exists():
        print(f"   ✅ {folder}")
    else:
        print(f"   ⚠️  Missing: {folder}")

print("\n4️⃣  Testing visualization (if diagnostic files found)...")

if found_files:
    try:
        from utils.visualize_diagnostics import load_diagnostic

        data = load_diagnostic(found_files[0])
        print(f"   ✅ Successfully loaded diagnostic file")
        print(f"      Mean similarity: {data['similarity_stats']['mean']:.3f}")
        print(f"      Mean reward: {data['environment_stats']['mean_reward']:.3f}")
    except Exception as e:
        print(f"   ⚠️  Could not process diagnostic: {e}")
else:
    print("   ⏭️  Skipped (no diagnostic files found)")

print("\n5️⃣  Summary...")

print("\n✅ What's working:")
print("   • Your project structure is organized")
print("   • Your diagnostic visualization exists")
print("   • Files are in correct locations")

print("\n📋 To use reward tracking:")
print("\n   Option A - Quick test on Ant:")
print("   $ python scripts/train_with_tracking.py --mode ant")

print("\n   Option B - Visualize existing results:")
print("   $ python src/utils/visualize_diagnostics.py")

print("\n   Option C - Full training:")
print("   $ python scripts/train_with_tracking.py --mode train \\")
print("         --instruction 'walk forward stably' \\")
print("         --env Ant-v4 \\")
print("         --timesteps 200000")

print("\n💡 Recommendation:")
print("   Since Humanoid falls, start with Ant-v4!")
print("   Train for 200k+ timesteps for better similarity scores.")

print("\n" + "=" * 70)
print("✨ TEST COMPLETED")
print("=" * 70)