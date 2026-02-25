"""
Batch experiment runner for testing different conditions.
Run multiple experiments systematically based on research parameters.
"""

from simulation import run_experiment
from config import (
    get_density_experiment_configs,
    get_mutation_experiment_configs, 
    get_radius_experiment_configs,
    SimulationConfig
)
import time
from datetime import datetime


def run_density_experiments():
    """
    Run density experiments: 10%, 20%, 40%, 60%
    Based on Chou (1997): 10-50% optimal, >60% overcrowding
    """
    print("\n" + "="*70)
    print("DENSITY EXPERIMENTS")
    print("Testing: 10%, 20%, 40%, 60% initial density")
    print("="*70)
    
    configs = get_density_experiment_configs()
    
    for name, config in configs:
        print(f"\n>>> Starting: {name}")
        start = time.time()
        
        results = run_experiment(config, name=f"density_exp/{name}")
        
        elapsed = time.time() - start
        print(f">>> Completed in {elapsed/60:.1f} minutes")
        print(f">>> Replicators: {results['final_replicator_count']}")
        print(f">>> Complexity: {results['complexity_score']:.2f}")
        print(f">>> Takeover: {results['takeover_detected']}")


def run_mutation_experiments():
    """
    Run mutation rate experiments: 0%, 0.01%, 0.1%, 1%
    Based on Agüera y Arcas: 0.024% optimal, >1% destructive
    """
    print("\n" + "="*70)
    print("MUTATION RATE EXPERIMENTS")
    print("Testing: 0%, 0.01%, 0.1%, 1% mutation rates")
    print("="*70)
    
    configs = get_mutation_experiment_configs()
    
    for name, config in configs:
        print(f"\n>>> Starting: {name}")
        start = time.time()
        
        results = run_experiment(config, name=f"mutation_exp/{name}")
        
        elapsed = time.time() - start
        print(f">>> Completed in {elapsed/60:.1f} minutes")
        print(f">>> Replicators: {results['final_replicator_count']}")
        print(f">>> Complexity: {results['complexity_score']:.2f}")


def run_radius_experiments():
    """
    Run interaction radius experiments: 1, 2, 3, 4
    Based on research: ≤2 limits propagation, 3+ may increase complexity
    """
    print("\n" + "="*70)
    print("INTERACTION RADIUS EXPERIMENTS")
    print("Testing: radius 1, 2, 3, 4")
    print("="*70)
    
    configs = get_radius_experiment_configs()
    
    for name, config in configs:
        print(f"\n>>> Starting: {name}")
        start = time.time()
        
        results = run_experiment(config, name=f"radius_exp/{name}")
        
        elapsed = time.time() - start
        print(f">>> Completed in {elapsed/60:.1f} minutes")
        print(f">>> Replicators: {results['final_replicator_count']}")
        print(f">>> Complexity: {results['complexity_score']:.2f}")


def run_long_term_experiment():
    """
    Run a long-term experiment with optimal parameters.
    This tests maximum achievable complexity with modern compute.
    """
    print("\n" + "="*70)
    print("LONG-TERM COMPLEXITY EXPERIMENT")
    print("100K epochs to test maximum achievable complexity")
    print("="*70)
    
    config = SimulationConfig(
        grid_height=128,
        grid_width=128,
        initial_density=0.3,
        max_epochs=100000,
        executions_per_epoch=200,
        mutation_rate=0.00024,  # Optimal from research
        interaction_radius=2,
        record_interval=100,
        checkpoint_interval=10000
    )
    
    print("\n⚠️  This will take several hours! Progress saved every 10K epochs.")
    print("Configuration:")
    print(f"  Grid: {config.grid_height}×{config.grid_width}")
    print(f"  Total epochs: {config.max_epochs:,}")
    print(f"  Executions per epoch: {config.executions_per_epoch}")
    print(f"  Total executions: {config.max_epochs * config.executions_per_epoch:,}")
    
    start = time.time()
    results = run_experiment(config, name="long_term_complexity")
    elapsed = time.time() - start
    
    print(f"\n✓ Completed in {elapsed/3600:.1f} hours")
    print(f"Final Results:")
    print(f"  Replicators: {results['final_replicator_count']}")
    print(f"  Complexity: {results['complexity_score']:.2f}")
    print(f"  Max code length: {max(results['metrics_history']['code_length']):.1f}")
    print(f"  Takeover: {results['takeover_detected']} at epoch {results['takeover_epoch']}")


def run_quick_batch():
    """
    Run a quick batch of experiments for testing.
    Smaller scale but covers all parameter types.
    """
    print("\n" + "="*70)
    print("QUICK BATCH TEST")
    print("Small-scale experiments across all parameter types")
    print("="*70)
    
    experiments = [
        ("quick_baseline", SimulationConfig(
            grid_height=32, grid_width=32, initial_density=0.2,
            max_epochs=1000, executions_per_epoch=50
        )),
        ("quick_high_density", SimulationConfig(
            grid_height=32, grid_width=32, initial_density=0.5,
            max_epochs=1000, executions_per_epoch=50
        )),
        ("quick_with_mutation", SimulationConfig(
            grid_height=32, grid_width=32, initial_density=0.2,
            max_epochs=1000, executions_per_epoch=50, mutation_rate=0.001
        )),
        ("quick_radius_2", SimulationConfig(
            grid_height=32, grid_width=32, initial_density=0.2,
            max_epochs=1000, executions_per_epoch=50, interaction_radius=2
        ))
    ]
    
    results_summary = []
    
    for name, config in experiments:
        print(f"\n>>> Running: {name}")
        start = time.time()
        results = run_experiment(config, name=f"quick_batch/{name}")
        elapsed = time.time() - start
        
        results_summary.append({
            'name': name,
            'time': elapsed,
            'replicators': results['final_replicator_count'],
            'complexity': results['complexity_score'],
            'takeover': results['takeover_detected']
        })
    
    print("\n" + "="*70)
    print("QUICK BATCH SUMMARY")
    print("="*70)
    for r in results_summary:
        print(f"{r['name']:20s} | {r['time']:5.1f}s | "
              f"Replicators: {r['replicators']:3d} | "
              f"Complexity: {r['complexity']:6.2f} | "
              f"Takeover: {r['takeover']}")


def main():
    """Main batch runner."""
    print("\n" + "#"*70)
    print("# PRIMORDIAL SOUP - BATCH EXPERIMENT RUNNER")
    print("# PWS Research - Job van der Veen")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*70)
    
    print("\nAvailable experiment batches:")
    print("  1. Quick batch (4 experiments, ~5-10 minutes)")
    print("  2. Density experiments (4 configs, ~40 minutes)")
    print("  3. Mutation experiments (4 configs, ~40 minutes)")
    print("  4. Radius experiments (4 configs, ~40 minutes)")
    print("  5. Long-term complexity (1 config, several hours)")
    print("  6. All standard experiments (density + mutation + radius)")
    
    choice = input("\nSelect experiment batch (1-6) or 'q' to quit: ").strip()
    
    if choice == '1':
        run_quick_batch()
    elif choice == '2':
        run_density_experiments()
    elif choice == '3':
        run_mutation_experiments()
    elif choice == '4':
        run_radius_experiments()
    elif choice == '5':
        run_long_term_experiment()
    elif choice == '6':
        run_density_experiments()
        run_mutation_experiments()
        run_radius_experiments()
    elif choice.lower() == 'q':
        print("Exiting...")
        return
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "#"*70)
    print(f"# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("# All results saved to: experiments/")
    print("#"*70)


if __name__ == "__main__":
    # Can run directly or import functions
    main()
