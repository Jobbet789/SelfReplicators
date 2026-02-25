"""
Example script demonstrating how to run the primordial soup simulation.
This shows the basic usage without GUI.
"""

from simulation import PrimordialSoup, run_experiment
from config import SimulationConfig, get_preset_config
import json


def simple_demo():
    """Run a simple demonstration of the simulation."""
    print("="*70)
    print("PRIMORDIAL SOUP SIMULATION - Simple Demo")
    print("="*70)
    print("\nThis demonstrates emergent self-replication from random code.")
    print("Based on research by Agüera y Arcas (2024) and Chou & Reggia (1997)")
    print()
    
    # Create a minimal configuration for quick demo
    config = SimulationConfig(
        grid_height=32,
        grid_width=32,
        initial_density=0.2,
        max_epochs=500,
        executions_per_epoch=50,
        mutation_rate=0.0001,
        record_interval=10
    )
    
    print("Configuration:")
    print(f"  Grid size: {config.grid_height}x{config.grid_width}")
    print(f"  Initial density: {config.initial_density*100}%")
    print(f"  Mutation rate: {config.mutation_rate*100}%")
    print(f"  Running for {config.max_epochs} epochs")
    print()
    
    # Create simulation
    soup = PrimordialSoup(config)
    
    print("Initial grid (first 10 rows):")
    print(soup.visualize_grid(max_rows=10))
    print()
    
    # Run simulation with progress updates
    def progress_callback(epoch, soup_instance):
        if epoch % 100 == 0 and epoch > 0:
            print(f"\nProgress at epoch {epoch}:")
            replicators = soup_instance.replicator_detector.detect_replicators()
            if replicators > 0:
                top = soup_instance.replicator_detector.get_most_successful(3)
                print(f"  Found {replicators} unique replicating patterns")
                print(f"  Top 3 replicators:")
                for i, (pattern, count) in enumerate(top, 1):
                    print(f"    {i}. '{pattern}' (appears {count} times, length {len(pattern)})")
    
    print("Starting simulation...")
    print("-" * 70)
    
    results = soup.run(callback=progress_callback)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"\nFinal Statistics:")
    print(f"  Total epochs: {results['final_epoch']}")
    print(f"  Total executions: {results['total_executions']}")
    print(f"  Success rate: {results['success_rate']*100:.2f}%")
    print(f"  Replicators found: {results['final_replicator_count']}")
    print(f"  Complexity score: {results['complexity_score']:.2f}")
    print(f"  Takeover detected: {results['takeover_detected']}")
    if results['takeover_detected']:
        print(f"  Takeover at epoch: {results['takeover_epoch']}")
    
    print(f"\nFinal Metrics:")
    for key, value in results['final_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    if results['top_replicators']:
        print(f"\nTop 5 Replicators:")
        for i, (pattern, count) in enumerate(results['top_replicators'][:5], 1):
            print(f"  {i}. '{pattern}'")
            print(f"     Length: {len(pattern)} | Copies: {count}")
    
    print("\nFinal grid (first 10 rows):")
    print(soup.visualize_grid(max_rows=10))
    print()


def preset_demo():
    """Run with a preset configuration."""
    print("\n" + "="*70)
    print("PRESET CONFIGURATION DEMO")
    print("="*70)
    print("\nRunning with 'standard' preset configuration...")
    
    config = get_preset_config('minimal')  # Using minimal for faster demo
    soup = PrimordialSoup(config)
    
    results = soup.run()
    
    print(f"\nResults:")
    print(f"  Replicators: {results['final_replicator_count']}")
    print(f"  Complexity: {results['complexity_score']:.2f}")
    print(f"  Success rate: {results['success_rate']*100:.1f}%")


def save_and_load_demo():
    """Demonstrate saving and loading checkpoints."""
    print("\n" + "="*70)
    print("CHECKPOINT SAVE/LOAD DEMO")
    print("="*70)
    
    # Create and run for 100 epochs
    config = SimulationConfig(
        grid_height=32,
        grid_width=32,
        max_epochs=100,
        executions_per_epoch=50
    )
    
    soup = PrimordialSoup(config)
    print("\nRunning first 100 epochs...")
    soup.run(epochs=100)
    
    # Save checkpoint
    print("\nSaving checkpoint...")
    soup.save_checkpoint("checkpoint_demo.pkl")
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    loaded_soup = PrimordialSoup.load_checkpoint("checkpoint_demo.pkl")
    
    # Continue for another 100 epochs
    print("\nContinuing for another 100 epochs...")
    loaded_soup.run(epochs=100)
    
    print(f"\nFinal epoch: {loaded_soup.current_epoch}")
    print(f"Total executions: {loaded_soup.total_executions}")


def metrics_demo():
    """Demonstrate metrics tracking over time."""
    print("\n" + "="*70)
    print("METRICS TRACKING DEMO")
    print("="*70)
    
    config = SimulationConfig(
        grid_height=32,
        grid_width=32,
        max_epochs=200,
        executions_per_epoch=50,
        record_interval=20
    )
    
    soup = PrimordialSoup(config)
    results = soup.run()
    
    print("\nMetrics History:")
    history = results['metrics_history']
    
    print(f"\nShannon Entropy over time:")
    for i, entropy in enumerate(history['shannon_entropy']):
        epoch = i * config.record_interval
        print(f"  Epoch {epoch:4d}: {entropy:.4f}")
    
    print(f"\nReplicator Count over time:")
    for i, count in enumerate(history['replicator_count']):
        epoch = i * config.record_interval
        print(f"  Epoch {epoch:4d}: {count} unique patterns")
    
    print(f"\nCode Length over time:")
    for i, length in enumerate(history['code_length']):
        epoch = i * config.record_interval
        print(f"  Epoch {epoch:4d}: {length:.2f} average length")


def experiment_demo():
    """Run a full experiment with automatic saving."""
    print("\n" + "="*70)
    print("FULL EXPERIMENT DEMO")
    print("="*70)
    
    config = get_preset_config('minimal')
    results = run_experiment(config, name="demo_experiment")
    
    print("\nExperiment results saved to: experiments/demo_experiment/")


if __name__ == "__main__":
    # Run different demos
    print("\n" + "#"*70)
    print("# PRIMORDIAL SOUP - EMERGENT SELF-REPLICATION")
    print("# PWS Research Implementation")
    print("#"*70)
    
    # Run the simple demo
    simple_demo()
    
    # Uncomment to run other demos:
    # preset_demo()
    # save_and_load_demo()
    # metrics_demo()
    # experiment_demo()
    
    print("\n" + "#"*70)
    print("# All demos complete!")
    print("#"*70)
