"""
Complete Paper Package Generator
Automates paper generation + multi-environment experiments
"""

import sys
import subprocess
from pathlib import Path
import time


def run_step(name, command, skip_on_error=False):
    """Run a step and track progress"""
    print("\n" + "="*60)
    print(f"STEP: {name}")
    print("="*60)
    
    try:
        if callable(command):
            command()
        else:
            subprocess.run(command, shell=True, check=True)
        print(f"✓ {name} complete")
        return True
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        if not skip_on_error:
            return False
        return True


def generate_paper_package(include_training=False):
    """Generate complete paper package"""
    
    start_time = time.time()
    
    print("="*60)
    print("GENERATING COMPLETE PAPER PACKAGE")
    print("="*60)
    print(f"Include new training: {include_training}")
    print(f"Estimated time: {'4-6 hours' if include_training else '30 minutes'}")
    print()
    
    steps_completed = 0
    total_steps = 5 if include_training else 3
    
    # Step 1: Generate paper draft
    if run_step("Generate Paper Draft", "python generate_paper.py"):
        steps_completed += 1
    
    # Step 2: Generate teacher presentation
    if run_step("Generate Teacher Presentation", "python create_teacher_report.py", skip_on_error=True):
        steps_completed += 1
    
    # Step 3: Create current reports
    if run_step("Generate Current Reports", "python simple_report.py", skip_on_error=True):
        steps_completed += 1
    
    if include_training:
        # Step 4: Train on multiple environments
        if run_step("Train Multiple Environments", "python train_multiple_envs.py --env all"):
            steps_completed += 1
        
        # Step 5: Evaluate all environments
        if run_step("Evaluate All Environments", "python evaluate_all_environments.py"):
            steps_completed += 1
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("PACKAGE GENERATION COMPLETE")
    print("="*60)
    print(f"Completed: {steps_completed}/{total_steps} steps")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print()
    
    # List generated files
    print("Generated files:")
    
    files_to_check = [
        ("research_paper_draft.md", "Paper draft (Markdown)"),
        ("research_paper.tex", "Paper template (LaTeX)"),
        ("teacher_presentation.png", "Teacher presentation"),
        ("teacher_summary.txt", "Teacher summary"),
        ("reports/phases_comparison.png", "Phases comparison"),
        ("reports/multi_env_comparison.png", "Multi-env comparison"),
        ("reports/multi_env_results.md", "Multi-env results (Markdown)"),
        ("reports/multi_env_results.tex", "Multi-env results (LaTeX)"),
    ]
    
    existing_files = []
    missing_files = []
    
    for filepath, description in files_to_check:
        path = Path(filepath)
        if path.exists():
            existing_files.append((filepath, description))
            print(f"  ✓ {filepath} - {description}")
        else:
            missing_files.append((filepath, description))
    
    if missing_files:
        print("\nMissing files (need training):")
        for filepath, description in missing_files:
            print(f"  ✗ {filepath} - {description}")
    
    # Next steps
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if not include_training:
        print("1. Review research_paper_draft.md")
        print("2. (Optional) Train on more environments:")
        print("   python complete_paper_package.py --train")
        print("3. Convert to PDF:")
        print("   pandoc research_paper_draft.md -o paper.pdf")
        print("4. Discuss with your teacher")
    else:
        print("1. Review all generated files")
        print("2. Update research_paper_draft.md with multi-env results")
        print("3. Convert to PDF for submission")
        print("4. Present to your teacher")
    
    print("\nFiles for teacher:")
    print("  📄 research_paper_draft.md (or PDF)")
    print("  📊 teacher_presentation.png")
    print("  📊 reports/phases_comparison.png")
    if include_training:
        print("  📊 reports/multi_env_comparison.png")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate complete paper package')
    parser.add_argument('--train', action='store_true',
                       help='Include training on multiple environments (4-6 hours)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: only generate from existing data')
    args = parser.parse_args()
    
    if args.quick:
        print("Quick mode: Using existing data only")
        include_training = False
    else:
        include_training = args.train
    
    generate_paper_package(include_training=include_training)
