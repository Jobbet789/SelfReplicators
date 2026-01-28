"""
Configuration management for the simulation.
Based on optimal parameters from research (Chou 1997, Agüera y Arcas 2024).
"""

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class SimulationConfig:
    """
    Configuration parameters for the primordial soup simulation.
    """
    
    # Grid dimensions
    grid_height: int = 64
    grid_width: int = 64
    
    # Initial conditions
    initial_density: float = 0.2  # 20% from research (10-50% range)
    
    # Execution parameters
    interaction_radius: int = 1  # How far programs can read/write
    max_program_length: int = 64  # Maximum code length to execute
    max_steps_per_execution: int = 100  # Prevent infinite loops
    
    # Mutation parameters
    mutation_rate: float = 0.0001  # 0.01% from Agüera y Arcas (0.024%)
    mutation_types: list = None  # Will be set in __post_init__
    
    # Simulation parameters
    max_epochs: int = 10000  # Total simulation epochs
    executions_per_epoch: int = 100  # Programs executed per epoch
    
    # Metrics and logging
    record_interval: int = 10  # Record metrics every N epochs
    checkpoint_interval: int = 1000  # Save checkpoint every N epochs
    
    # Replicator detection
    min_replicator_length: int = 4  # Minimum pattern length
    
    def __post_init__(self):
        """Initialize default mutation types if not provided."""
        if self.mutation_types is None:
            self.mutation_types = [
                'point',      # Change single token
                'insertion',  # Insert random token
                'deletion',   # Delete token
                'duplication' # Duplicate segment
            ]
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'initial_density': self.initial_density,
            'interaction_radius': self.interaction_radius,
            'max_program_length': self.max_program_length,
            'max_steps_per_execution': self.max_steps_per_execution,
            'mutation_rate': self.mutation_rate,
            'mutation_types': self.mutation_types,
            'max_epochs': self.max_epochs,
            'executions_per_epoch': self.executions_per_epoch,
            'record_interval': self.record_interval,
            'checkpoint_interval': self.checkpoint_interval,
            'min_replicator_length': self.min_replicator_length
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SimulationConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SimulationConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        if self.grid_height <= 0 or self.grid_width <= 0:
            raise ValueError("Grid dimensions must be positive")
        
        if not 0 <= self.initial_density <= 1:
            raise ValueError("Initial density must be between 0 and 1")
        
        if self.interaction_radius < 1:
            raise ValueError("Interaction radius must be at least 1")
        
        if self.mutation_rate < 0 or self.mutation_rate > 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        
        if self.max_epochs <= 0:
            raise ValueError("Max epochs must be positive")
        
        return True


# Preset configurations based on research

def get_preset_config(name: str) -> SimulationConfig:
    """
    Get a preset configuration.
    
    Args:
        name: Preset name ('minimal', 'standard', 'intensive', 'research')
        
    Returns:
        SimulationConfig instance
    """
    presets = {
        'minimal': SimulationConfig(
            grid_height=32,
            grid_width=32,
            initial_density=0.2,
            max_epochs=1000,
            executions_per_epoch=50
        ),
        
        'standard': SimulationConfig(
            grid_height=64,
            grid_width=64,
            initial_density=0.2,
            max_epochs=10000,
            executions_per_epoch=100
        ),
        
        'intensive': SimulationConfig(
            grid_height=128,
            grid_width=128,
            initial_density=0.3,
            max_epochs=50000,
            executions_per_epoch=200,
            mutation_rate=0.0001
        ),
        
        'research': SimulationConfig(
            grid_height=128,
            grid_width=128,
            initial_density=0.4,  # From Agüera y Arcas range
            max_epochs=100000,
            executions_per_epoch=200,
            mutation_rate=0.00024,  # 0.024% from paper
            max_program_length=64,
            interaction_radius=2
        )
    }
    
    if name not in presets:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
    
    return presets[name]


# Configuration for different experimental conditions

def get_density_experiment_configs() -> list:
    """
    Get configs for density experiments (10%, 20%, 40%, 60%).
    Based on Chou (1997) research: 10-50% optimal, >60% overcrowding.
    """
    densities = [0.1, 0.2, 0.4, 0.6]
    configs = []
    
    for density in densities:
        config = SimulationConfig(
            initial_density=density,
            grid_height=64,
            grid_width=64,
            max_epochs=10000
        )
        configs.append((f"density_{int(density*100)}", config))
    
    return configs


def get_mutation_experiment_configs() -> list:
    """
    Get configs for mutation rate experiments.
    Based on Agüera y Arcas: 0.024% helps, >1% destroys.
    """
    mutation_rates = [0.0, 0.0001, 0.001, 0.01]
    configs = []
    
    for rate in mutation_rates:
        config = SimulationConfig(
            mutation_rate=rate,
            grid_height=64,
            grid_width=64,
            max_epochs=10000
        )
        configs.append((f"mutation_{rate}", config))
    
    return configs


def get_radius_experiment_configs() -> list:
    """
    Get configs for interaction radius experiments.
    Based on research: radius ≤2 limits propagation, 3+ may increase complexity.
    """
    radii = [1, 2, 3, 4]
    configs = []
    
    for radius in radii:
        config = SimulationConfig(
            interaction_radius=radius,
            grid_height=64,
            grid_width=64,
            max_epochs=10000
        )
        configs.append((f"radius_{radius}", config))
    
    return configs
