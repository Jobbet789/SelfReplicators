"""
Main simulation engine for emergent self-replication.
Implements the primordial soup where replicators can emerge from chaos.
"""

import numpy as np
from typing import Optional, Dict, List
import pickle
from pathlib import Path

from grid import Grid
from brainfuck import GridBrainfuckEngine
from metrics import ComplexityMetrics, ReplicatorDetector
from config import SimulationConfig


class PrimordialSoup:
    """
    The main simulation class implementing the digital primordial soup.
    Random code fragments interact locally, enabling emergent self-replication.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the primordial soup simulation.
        
        Args:
            config: Simulation configuration (uses default if None)
        """
        self.config = config if config is not None else SimulationConfig()
        self.config.validate()
        
        # Initialize grid
        self.grid = Grid(
            height=self.config.grid_height,
            width=self.config.grid_width,
            density=self.config.initial_density
        )
        
        # Initialize execution engine
        self.engine = GridBrainfuckEngine(
            grid=self.grid,
            interaction_radius=self.config.interaction_radius,
            max_program_length=self.config.max_program_length,
            max_steps=self.config.max_steps_per_execution
        )
        
        # Initialize metrics
        self.metrics = ComplexityMetrics(self.grid)
        self.replicator_detector = ReplicatorDetector(
            self.grid,
            min_pattern_length=self.config.min_replicator_length
        )
        
        # Simulation state
        self.current_epoch = 0
        self.total_executions = 0
        self.successful_executions = 0
        self.takeover_detected = False
        self.takeover_epoch = None
        
    def mutate_grid(self):
        """
        Apply random mutations to the grid based on mutation rate.
        Implements point mutations, insertions, deletions, and duplications.
        """
        if self.config.mutation_rate == 0:
            return
        
        total_cells = self.grid.height * self.grid.width
        num_mutations = int(total_cells * self.config.mutation_rate)
        
        if num_mutations == 0:
            return
        
        for _ in range(num_mutations):
            # Random position
            x = np.random.randint(0, self.grid.height)
            y = np.random.randint(0, self.grid.width)
            
            # Random mutation type
            mutation_type = np.random.choice(self.config.mutation_types)
            
            if mutation_type == 'point':
                # Change to random token
                new_token = np.random.choice(self.grid.tokens + [' '])
                self.grid.set_cell(x, y, new_token)
                
            elif mutation_type == 'insertion':
                # Insert a random token (shift right)
                new_token = np.random.choice(self.grid.tokens)
                # Get current row
                row = [self.grid.get_cell(x, i) for i in range(self.grid.width)]
                # Insert and shift
                row.insert(y, new_token)
                row = row[:self.grid.width]  # Truncate
                # Write back
                for i, token in enumerate(row):
                    self.grid.set_cell(x, i, token)
                    
            elif mutation_type == 'deletion':
                # Delete token (shift left)
                row = [self.grid.get_cell(x, i) for i in range(self.grid.width)]
                if y < len(row):
                    row.pop(y)
                    row.append(' ')  # Fill with empty
                for i, token in enumerate(row):
                    self.grid.set_cell(x, i, token)
                    
            elif mutation_type == 'duplication':
                # Duplicate a segment (2-4 tokens)
                dup_length = np.random.randint(2, 5)
                segment = [self.grid.get_cell(x, (y + i) % self.grid.width) 
                          for i in range(dup_length)]
                # Write duplicate next to original
                for i, token in enumerate(segment):
                    self.grid.set_cell(x, (y + dup_length + i) % self.grid.width, token)
    
    def run_epoch(self):
        """
        Run one epoch of the simulation.
        An epoch consists of multiple random program executions and mutations.
        """
        # Execute random programs
        for _ in range(self.config.executions_per_epoch):
            # Pick random starting position
            x = np.random.randint(0, self.grid.height)
            y = np.random.randint(0, self.grid.width)
            
            # Execute
            success = self.engine.execute_at_position(x, y)
            
            self.total_executions += 1
            if success:
                self.successful_executions += 1
        
        # Apply mutations
        self.mutate_grid()
        
        # Record metrics if needed
        if self.current_epoch % self.config.record_interval == 0:
            replicator_count = self.replicator_detector.detect_replicators()
            self.metrics.record_metrics(replicator_count)
            
            # Check for takeover
            if not self.takeover_detected and self.metrics.detect_takeover():
                self.takeover_detected = True
                self.takeover_epoch = self.current_epoch
        
        self.current_epoch += 1
    
    def run(self, epochs: Optional[int] = None, 
            callback: Optional[callable] = None) -> Dict:
        """
        Run the simulation for a specified number of epochs.
        
        Args:
            epochs: Number of epochs to run (uses config default if None)
            callback: Optional callback function called each epoch with (epoch, soup)
            
        Returns:
            Dictionary with simulation results
        """
        if epochs is None:
            epochs = self.config.max_epochs
        
        start_epoch = self.current_epoch
        target_epoch = start_epoch + epochs
        
        while self.current_epoch < target_epoch:
            self.run_epoch()
            
            # Call callback if provided
            if callback is not None:
                callback(self.current_epoch, self)
            
            # Print progress
            if self.current_epoch % 100 == 0:
                self._print_progress()
        
        return self.get_results()
    
    def _print_progress(self):
        """Print current simulation progress."""
        success_rate = (self.successful_executions / self.total_executions * 100 
                       if self.total_executions > 0 else 0)
        
        print(f"Epoch {self.current_epoch}/{self.config.max_epochs} | "
              f"Success rate: {success_rate:.1f}% | "
              f"Replicators: {len(self.replicator_detector.detected_replicators)} | "
              f"Density: {self.metrics.density():.3f} | "
              f"Entropy: {self.metrics.shannon_entropy():.3f}")
        
        if self.takeover_detected and self.takeover_epoch == self.current_epoch:
            print(f">>> TAKEOVER DETECTED AT EPOCH {self.current_epoch}! <<<")
    
    def get_results(self) -> Dict:
        """
        Get comprehensive results of the simulation.
        
        Returns:
            Dictionary with all results and metrics
        """
        # Get final replicator count
        final_replicators = self.replicator_detector.detect_replicators()
        
        return {
            'config': self.config.to_dict(),
            'final_epoch': self.current_epoch,
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'success_rate': (self.successful_executions / self.total_executions 
                           if self.total_executions > 0 else 0),
            'takeover_detected': self.takeover_detected,
            'takeover_epoch': self.takeover_epoch,
            'final_replicator_count': final_replicators,
            'top_replicators': self.replicator_detector.get_most_successful(10),
            'complexity_score': self.replicator_detector.get_complexity_score(),
            'final_metrics': self.metrics.get_summary(),
            'metrics_history': self.metrics.history
        }
    
    def save_checkpoint(self, filepath: str):
        """
        Save simulation checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'config': self.config,
            'grid': self.grid.grid.copy(),
            'current_epoch': self.current_epoch,
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'takeover_detected': self.takeover_detected,
            'takeover_epoch': self.takeover_epoch,
            'metrics_history': self.metrics.history
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'PrimordialSoup':
        """
        Load simulation from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            PrimordialSoup instance restored from checkpoint
        """
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create new instance
        soup = cls(config=checkpoint['config'])
        
        # Restore state
        soup.grid.grid = checkpoint['grid']
        soup.current_epoch = checkpoint['current_epoch']
        soup.total_executions = checkpoint['total_executions']
        soup.successful_executions = checkpoint['successful_executions']
        soup.takeover_detected = checkpoint['takeover_detected']
        soup.takeover_epoch = checkpoint['takeover_epoch']
        soup.metrics.history = checkpoint['metrics_history']
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {soup.current_epoch}")
        
        return soup
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.grid = Grid(
            height=self.config.grid_height,
            width=self.config.grid_width,
            density=self.config.initial_density
        )
        self.engine.grid = self.grid
        self.metrics = ComplexityMetrics(self.grid)
        self.replicator_detector = ReplicatorDetector(
            self.grid,
            min_pattern_length=self.config.min_replicator_length
        )
        self.current_epoch = 0
        self.total_executions = 0
        self.successful_executions = 0
        self.takeover_detected = False
        self.takeover_epoch = None
    
    def visualize_grid(self, max_rows: int = 20) -> str:
        """
        Get a string visualization of the grid.
        
        Args:
            max_rows: Maximum number of rows to display
            
        Returns:
            String representation of grid
        """
        rows_to_show = min(max_rows, self.grid.height)
        lines = []
        
        for i in range(rows_to_show):
            row = ''.join(self.grid.grid[i])
            lines.append(f"{i:3d}: {row}")
        
        if rows_to_show < self.grid.height:
            lines.append(f"... ({self.grid.height - rows_to_show} more rows)")
        
        return '\n'.join(lines)


def run_experiment(config: SimulationConfig, name: str = "experiment") -> Dict:
    """
    Run a complete experiment with the given configuration.
    
    Args:
        config: Simulation configuration
        name: Experiment name for saving
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Starting experiment: {name}")
    print(f"{'='*60}\n")
    
    # Create simulation
    soup = PrimordialSoup(config)
    
    # Run
    results = soup.run()
    
    # Save results
    output_dir = Path("experiments") / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final checkpoint
    soup.save_checkpoint(str(output_dir / "final_checkpoint.pkl"))
    
    # Save config
    config.save(str(output_dir / "config.json"))
    
    # Save results as pickle
    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*60}")
    print(f"Experiment complete: {name}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return results
