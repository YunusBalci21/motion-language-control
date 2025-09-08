#!/usr/bin/env python3
"""
Quick Windows Setup Fix for Enhanced Motion-Language Control
Fixes common MuJoCo and environment issues on Windows
"""

import os
import sys
import subprocess
import platform


def fix_mujoco_environment():
    """Fix MuJoCo environment variables for Windows"""
    print("Fixing MuJoCo environment variables...")

    # Set environment variable for current session
    os.environ['MUJOCO_GL'] = 'glfw'
    print("✓ Set MUJOCO_GL=glfw for current session")

    # Try to set permanently for Windows
    if platform.system() == "Windows":
        try:
            subprocess.run(['setx', 'MUJOCO_GL', 'glfw'], check=True, capture_output=True)
            print("✓ Set MUJOCO_GL=glfw permanently (requires restart)")
        except subprocess.CalledProcessError:
            print("! Could not set environment variable permanently")
            print("  Please run: setx MUJOCO_GL glfw")

    return True


def test_basic_imports():
    """Test basic imports"""
    print("\nTesting basic imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ PyTorch not installed")
        return False

    try:
        import gymnasium
        print(f"✓ Gymnasium {gymnasium.__version__}")
    except ImportError:
        print("✗ Gymnasium not installed")
        return False

    try:
        import stable_baselines3
        print(f"✓ Stable-Baselines3 {stable_baselines3.__version__}")
    except ImportError:
        print("✗ Stable-Baselines3 not installed")
        return False

    return True


def test_mujoco():
    """Test MuJoCo installation"""
    print("\nTesting MuJoCo...")

    try:
        import mujoco
        print(f"✓ MuJoCo {mujoco.__version__}")
    except ImportError:
        print("✗ MuJoCo not installed")
        print("  Install with: conda install -c conda-forge mujoco")
        print("  Or: pip install mujoco")
        return False
    except Exception as e:
        print(f"✗ MuJoCo import error: {e}")
        return False

    return True


def test_environments():
    """Test environment creation"""
    print("\nTesting environments...")

    # Test environments in order of preference
    test_envs = [
        "CartPole-v1",  # Simple, should always work
        "HalfCheetah-v4",
        "Ant-v4",
        "Walker2d-v4",
        "Humanoid-v4"
    ]

    working_envs = []

    try:
        import gymnasium as gym

        for env_name in test_envs:
            try:
                env = gym.make(env_name)
                env.close()
                print(f"✓ {env_name} works")
                working_envs.append(env_name)
            except Exception as e:
                print(f"✗ {env_name} failed: {e}")

    except ImportError:
        print("✗ Gymnasium not available")
        return []

    return working_envs


def test_project_imports():
    """Test project-specific imports"""
    print("\nTesting project imports...")

    try:
        sys.path.append('./src')
        from models.motion_tokenizer import MotionTokenizer
        print("✓ MotionTokenizer import successful")

        from agents.hierarchical_agent import MotionLanguageAgent
        print("✓ MotionLanguageAgent import successful")

        return True

    except ImportError as e:
        print(f"✗ Project import failed: {e}")
        print("  Make sure you're in the project root directory")
        return False


def suggest_quick_test(working_envs):
    """Suggest appropriate quick test command"""
    print("\n" + "=" * 60)
    print("QUICK TEST SUGGESTIONS")
    print("=" * 60)

    if not working_envs:
        print("❌ No working environments found!")
        print("Please install MuJoCo properly:")
        print("  conda install -c conda-forge mujoco")
        print("  pip install mujoco")
        return

    # Suggest best working environment
    preferred_order = ["Humanoid-v4", "HalfCheetah-v4", "Ant-v4", "Walker2d-v4", "CartPole-v1"]

    best_env = None
    for env in preferred_order:
        if env in working_envs:
            best_env = env
            break

    if not best_env:
        best_env = working_envs[0]

    print(f"✓ Recommended environment: {best_env}")
    print("\nTry running:")
    print(f"  python train_motion_language_agent.py --env {best_env} --quick-test --curriculum basic")

    if best_env != "Humanoid-v4":
        print(f"\nNote: Using {best_env} instead of Humanoid-v4")
        print("The core motion-language learning will still work!")


def main():
    print("Motion-Language Control - Windows Fix")
    print("=" * 50)

    # Fix environment variables
    fix_mujoco_environment()

    # Test imports
    if not test_basic_imports():
        print("\n❌ Basic imports failed. Please install requirements:")
        print("  pip install -r requirements.txt")
        return

    # Test MuJoCo
    mujoco_ok = test_mujoco()

    # Test environments
    working_envs = test_environments()

    # Test project imports
    project_ok = test_project_imports()

    # Summary
    print("\n" + "=" * 50)
    print("SETUP STATUS SUMMARY")
    print("=" * 50)
    print(f"Basic imports: {'✓' if True else '✗'}")
    print(f"MuJoCo: {'✓' if mujoco_ok else '✗'}")
    print(f"Working environments: {len(working_envs)}")
    print(f"Project imports: {'✓' if project_ok else '✗'}")

    if working_envs and project_ok:
        suggest_quick_test(working_envs)
        print("\n🎉 Setup looks good! Try the suggested command above.")
    else:
        print("\n⚠️  Some issues found. Please fix the errors above.")


if __name__ == "__main__":
    main()