#!/usr/bin/env python3
"""
Quick Component Test
Test individual components before running the full conversational flow
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

def test_deepseek_llm():
    """Test DeepSeek LLM component"""
    print("🧠 Testing DeepSeek LLM...")
    try:
        from conversation.deepseek_llm import DeepSeekLLM
        
        llm = DeepSeekLLM()
        response = llm.process_user_request("Hey, can you clean the table?")
        
        print(f"✅ DeepSeek LLM working!")
        print(f"   Response: {response.llm_response[:50]}...")
        print(f"   Actions: {response.extracted_actions}")
        return True
    except Exception as e:
        print(f"❌ DeepSeek LLM failed: {e}")
        return False

def test_motion_tokenizer():
    """Test MotionGPT tokenizer"""
    print("\n🎯 Testing MotionGPT Tokenizer...")
    try:
        from models.motion_tokenizer import MotionTokenizer
        import numpy as np
        
        tokenizer = MotionTokenizer()
        
        # Test motion extraction
        mock_obs = np.random.randn(376)  # Humanoid obs
        mock_obs[0] = 1.0  # height
        
        motion_features = tokenizer.extract_motion_from_obs(mock_obs, "Humanoid-v4")
        
        # Test similarity
        mock_motion = np.tile(motion_features, (10, 1))
        similarity = tokenizer.compute_motion_language_similarity(mock_motion, "walk forward")
        
        print(f"✅ MotionGPT Tokenizer working!")
        print(f"   Motion features shape: {motion_features.shape}")
        print(f"   Similarity score: {similarity:.3f}")
        return True
    except Exception as e:
        print(f"❌ MotionGPT Tokenizer failed: {e}")
        return False

def test_task_planner():
    """Test task planner"""
    print("\n📋 Testing Task Planner...")
    try:
        from conversation.task_planner import TaskPlanner
        
        planner = TaskPlanner()
        plan = planner.create_execution_plan(
            ["walk forward", "reach forward", "wipe surface"],
            "Clean the table"
        )
        
        print(f"✅ Task Planner working!")
        print(f"   Motion steps: {len(plan.motion_steps)}")
        print(f"   Duration: {plan.total_duration} steps")
        return True
    except Exception as e:
        print(f"❌ Task Planner failed: {e}")
        return False

def test_feedback_system():
    """Test feedback system"""
    print("\n💬 Testing Feedback System...")
    try:
        from conversation.feedback_system import FeedbackSystem
        
        feedback_sys = FeedbackSystem()
        feedback_entry = feedback_sys.process_feedback(
            "Clean table",
            ["walk forward", "wipe surface"],
            "Great job, very smooth!",
            {"similarity": 0.8}
        )
        
        print(f"✅ Feedback System working!")
        print(f"   Sentiment: {feedback_entry.sentiment}")
        print(f"   Improvements: {feedback_entry.improvement_suggestions}")
        return True
    except Exception as e:
        print(f"❌ Feedback System failed: {e}")
        return False

def test_motion_agent():
    """Test motion execution agent"""
    print("\n🦾 Testing Motion Agent...")
    try:
        from agents.hierarchical_agent import EnhancedMotionLanguageAgent
        
        agent = EnhancedMotionLanguageAgent("Humanoid-v4")
        
        print(f"✅ Motion Agent working!")
        print(f"   Environment: {agent.env_name}")
        print(f"   Device: {agent.device}")
        return True
    except Exception as e:
        print(f"❌ Motion Agent failed: {e}")
        return False

def main():
    """Run quick component tests"""
    print("🚀 QUICK COMPONENT TEST")
    print("=" * 50)
    print("Testing all components before full conversational demo...")
    print("=" * 50)

    tests = [
        test_deepseek_llm,
        test_motion_tokenizer, 
        test_task_planner,
        test_feedback_system,
        test_motion_agent
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All components working! Ready for conversational demo!")
        print("\n🚀 Next steps:")
        print("1. Run performance test: python test_current_performance.py")
        print("2. Run conversational demo: python enhanced_conversational_demo.py")
    else:
        print("❌ Some components need attention.")
        print("Check error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
