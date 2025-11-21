import sys
from pathlib import Path
import subprocess

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def generate_all_reports():
    """Generate all reward reports and visualizations"""
    
    checkpoints_dir = Path("./checkpoints")
    
    if not checkpoints_dir.exists():
        print("Error: No checkpoints directory found")
        return
    
    # Find all phase directories
    phase_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()])
    
    if not phase_dirs:
        print("Error: No checkpoint directories found")
        return
    
    print(f"Found {len(phase_dirs)} checkpoint directories")
    print("="*60)
    
    reports_dir = Path("src/reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Generate individual reports for each phase
    for phase_dir in phase_dirs:
        print(f"\nGenerating report for {phase_dir.name}...")
        
        from generate_reward_report import generate_complete_report
        try:
            generate_complete_report(str(phase_dir), "src/reports")
        except Exception as e:
            print(f"Error generating report for {phase_dir.name}: {e}")
    
    # Generate comparison if multiple phases
    if len(phase_dirs) > 1:
        print("\nGenerating comparison plot...")
        
        from plot_training_rewards import compare_phases, create_reward_summary
        
        try:
            fig = compare_phases(
                [str(d) for d in phase_dirs],
                labels=[d.name for d in phase_dirs],
                save_path="./reports/all_phases_comparison.png"
            )
            
            # Print summary
            summary = create_reward_summary(
                [str(d) for d in phase_dirs],
                labels=[d.name for d in phase_dirs]
            )
            
            print("\n" + "="*80)
            print("TRAINING SUMMARY - ALL PHASES")
            print("="*80)
            print(f"{'Phase':<25} {'Final Reward':<15} {'Mean Reward':<15} {'Max Length':<15}")
            print("-"*80)
            for stat in summary:
                print(f"{stat['Phase']:<25} {stat['Final Reward']:<15.2f} "
                      f"{stat['Mean Reward']:<15.2f} {stat['Max Episode Length']:<15.0f}")
            
        except Exception as e:
            print(f"Error generating comparison: {e}")
    
    print("\n" + "="*60)
    print("✓ All reports generated!")
    print(f"✓ Reports saved to: {reports_dir.absolute()}")
    print("="*60)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(reports_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    generate_all_reports()
