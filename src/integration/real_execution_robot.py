#!/usr/bin/env python3
"""
Real Execution Robot - Actually runs the trained models and records results
"""

import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from conversation.deepseek_llm import DeepSeekLLM
from conversation.task_planner import TaskPlanner
from conversation.feedback_system import FeedbackSystem

try:
    from agents.hierarchical_agent import EnhancedMotionLanguageAgent

    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


class RealExecutionRobot:
    """
    Robot that ACTUALLY executes commands and records results for the paper
    """

    def __init__(self,
                 model_path: str,
                 env_name: str = "Humanoid-v4",
                 device: str = "cuda",
                 output_dir: str = "./execution_results"):

        self.model_path = model_path
        self.env_name = env_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        print("🤖 Initializing Real Execution Robot")
        print("=" * 60)

        self.llm = DeepSeekLLM(use_real_model=True, device=device)
        self.planner = TaskPlanner()
        self.feedback = FeedbackSystem()

        # Initialize agent with trained model
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.agent = EnhancedMotionLanguageAgent(
            env_name=env_name,
            device=device,
            use_stability_focus=True
        )

        print(f"✓ Loaded model: {model_path}")
        print("=" * 60)

        # Results tracking
        self.execution_results = []
        self.session_start = datetime.now()

    def execute_command(self, user_input: str, record_video: bool = True) -> Dict:
        """
        Execute a command and return detailed results
        """
        print(f"\n{'=' * 80}")
        print(f"EXECUTING: {user_input}")
        print(f"{'=' * 80}")

        result = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "llm_response": None,
            "motion_plan": None,
            "execution_metrics": None,
            "success": False,
            "video_path": None
        }

        # Step 1: LLM Understanding
        print("\n1️⃣ LLM Processing...")
        conversation_turn = self.llm.process_user_request(user_input)
        result["llm_response"] = conversation_turn.llm_response
        result["chain_of_thought"] = conversation_turn.chain_of_thought

        if not conversation_turn.extracted_actions:
            print("⚠ No commands extracted")
            return result

        # Step 2: Task Planning
        print("\n2️⃣ Motion Planning...")
        motion_plan = self.planner.create_motion_plan(conversation_turn.extracted_actions)
        result["motion_plan"] = {
            "sequence": motion_plan.motion_sequence,
            "duration": motion_plan.estimated_duration,
            "warnings": motion_plan.warnings
        }

        print(f"   Sequence: {' → '.join(motion_plan.motion_sequence)}")
        print(f"   Estimated: {motion_plan.estimated_duration:.1f}s")

        # Step 3: Execute each motion with the trained model
        print("\n3️⃣ Executing Motions...")
        execution_results = []

        for i, instruction in enumerate(motion_plan.motion_sequence):
            print(f"\n   Motion {i + 1}/{len(motion_plan.motion_sequence)}: {instruction}")

            # Set up video recording
            video_path = None
            if record_video:
                video_path = str(
                    self.output_dir / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{instruction.replace(' ', '_')}")

            # Execute with the trained model
            try:
                eval_result = self.agent.evaluate_instruction(
                    instruction=instruction,
                    model_path=self.model_path,
                    num_episodes=1,
                    language_reward_weight=0.7,
                    deterministic=True,
                    record_video=record_video,
                    video_path=video_path,
                    render=False  # Don't render to screen, save video instead
                )

                execution_results.append({
                    "instruction": instruction,
                    "similarity": eval_result.get("mean_similarity", 0.0),
                    "success_rate": eval_result.get("episode_success_rate", 0.0),
                    "motion_quality": eval_result.get("mean_motion_overall_quality", 0.0),
                    "reward": eval_result.get("mean_total_reward", 0.0),
                    "video_path": video_path
                })

                print(f"      Similarity: {eval_result.get('mean_similarity', 0.0):.3f}")
                print(f"      Success: {eval_result.get('episode_success_rate', 0.0):.1%}")
                print(f"      Quality: {eval_result.get('mean_motion_overall_quality', 0.0):.3f}")

            except Exception as e:
                print(f"      ✗ Execution failed: {e}")
                execution_results.append({
                    "instruction": instruction,
                    "error": str(e)
                })

        # Step 4: Aggregate metrics
        result["execution_metrics"] = {
            "individual_motions": execution_results,
            "overall_similarity": sum(r.get("similarity", 0) for r in execution_results) / len(execution_results),
            "overall_success_rate": sum(r.get("success_rate", 0) for r in execution_results) / len(execution_results),
            "overall_quality": sum(r.get("motion_quality", 0) for r in execution_results) / len(execution_results),
        }

        result["success"] = result["execution_metrics"]["overall_success_rate"] > 0.5

        # Save result
        self.execution_results.append(result)
        self._save_results()

        # Print summary
        print(f"\n{'=' * 80}")
        print(f"EXECUTION COMPLETE")
        print(f"{'=' * 80}")
        print(f"Overall Similarity: {result['execution_metrics']['overall_similarity']:.3f}")
        print(f"Overall Success: {result['execution_metrics']['overall_success_rate']:.1%}")
        print(f"Overall Quality: {result['execution_metrics']['overall_quality']:.3f}")
        print(f"{'=' * 80}\n")

        return result

    def run_experiment(self, test_commands: List[str]) -> Dict:
        """
        Run a full experiment with multiple commands
        """
        print(f"\n{'#' * 80}")
        print(f"STARTING EXPERIMENT: {len(test_commands)} commands")
        print(f"Model: {self.model_path}")
        print(f"Environment: {self.env_name}")
        print(f"{'#' * 80}\n")

        experiment_results = {
            "session_id": self.session_start.isoformat(),
            "model_path": str(self.model_path),
            "environment": self.env_name,
            "commands": []
        }

        for i, command in enumerate(test_commands, 1):
            print(f"\n{'=' * 80}")
            print(f"Command {i}/{len(test_commands)}")
            print(f"{'=' * 80}")

            result = self.execute_command(command, record_video=True)
            experiment_results["commands"].append(result)

            time.sleep(2)  # Brief pause between commands

        # Generate report
        self._generate_report(experiment_results)

        return experiment_results

    def _save_results(self):
        """Save results to JSON"""
        output_file = self.output_dir / f"results_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, 'w') as f:
            json.dump({
                "session_start": self.session_start.isoformat(),
                "model_path": str(self.model_path),
                "environment": self.env_name,
                "results": self.execution_results
            }, f, indent=2)

        print(f"💾 Results saved to: {output_file}")

    def _generate_report(self, experiment_results: Dict):
        """Generate a report for the paper"""
        report_file = self.output_dir / f"report_{self.session_start.strftime('%Y%m%d_%H%M%S')}.md"

        # Calculate aggregate statistics
        all_similarities = []
        all_success_rates = []
        all_qualities = []

        for cmd_result in experiment_results["commands"]:
            if cmd_result.get("execution_metrics"):
                metrics = cmd_result["execution_metrics"]
                all_similarities.append(metrics["overall_similarity"])
                all_success_rates.append(metrics["overall_success_rate"])
                all_qualities.append(metrics["overall_quality"])

        if not all_similarities:
            print("⚠ No valid metrics to generate report")
            return

        # Generate markdown report (use -> instead of →)
        report = f"""# Conversational Robot Execution Report

    **Session:** {experiment_results['session_id']}  
    **Model:** {experiment_results['model_path']}  
    **Environment:** {experiment_results['environment']}  
    **Total Commands:** {len(experiment_results['commands'])}

    ## Summary Statistics

    | Metric | Mean | Std | Min | Max |
    |--------|------|-----|-----|-----|
    | Similarity | {sum(all_similarities) / len(all_similarities):.3f} | {(sum((x - sum(all_similarities) / len(all_similarities)) ** 2 for x in all_similarities) / len(all_similarities)) ** 0.5:.3f} | {min(all_similarities):.3f} | {max(all_similarities):.3f} |
    | Success Rate | {sum(all_success_rates) / len(all_success_rates):.1%} | - | {min(all_success_rates):.1%} | {max(all_success_rates):.1%} |
    | Motion Quality | {sum(all_qualities) / len(all_qualities):.3f} | {(sum((x - sum(all_qualities) / len(all_qualities)) ** 2 for x in all_qualities) / len(all_qualities)) ** 0.5:.3f} | {min(all_qualities):.3f} | {max(all_qualities):.3f} |

    ## Individual Command Results

    """

        for i, cmd_result in enumerate(experiment_results["commands"], 1):
            report += f"\n### {i}. {cmd_result['user_input']}\n\n"

            if cmd_result.get("execution_metrics"):
                metrics = cmd_result["execution_metrics"]
                motion_plan = cmd_result.get('motion_plan', {})
                sequence = motion_plan.get('sequence', [])

                report += f"""
    **LLM Reasoning:**
    ```
    {cmd_result.get('chain_of_thought', 'N/A')}
    ```

    **Motion Plan:** {' -> '.join(sequence)}

    **Results:**
    - Similarity: {metrics['overall_similarity']:.3f}
    - Success Rate: {metrics['overall_success_rate']:.1%}
    - Motion Quality: {metrics['overall_quality']:.3f}

    """

        report += f"\n---\n*Generated: {datetime.now().isoformat()}*\n"

        # CRITICAL FIX: Use UTF-8 encoding
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📊 Report generated: {report_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--env', type=str, default='Humanoid-v4')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='./execution_results')
    args = parser.parse_args()

    robot = RealExecutionRobot(
        model_path=args.model,
        env_name=args.env,
        device=args.device,
        output_dir=args.output
    )

    # Test commands for the paper
    test_commands = [
        "Walk forward",
        "Turn left",
        "Walk forward slowly",
        "Stop moving",
        "Turn right and walk forward",
    ]

    robot.run_experiment(test_commands)


if __name__ == "__main__":
    main()