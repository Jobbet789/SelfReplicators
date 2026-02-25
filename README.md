# Primordial Soup - Emergent Self-Replication Simulation

A computational simulation exploring how complex self-replicating structures can emerge from pure chaos. Based on research by Agüera y Arcas (2024) and Chou & Reggia (1997).

## Research Context

This implementation is part of a PWS (Profielwerkstuk) investigating:
- **Main Question**: What is the maximum complexity that emergent self-replicating structures can achieve in a spatial computer simulation?
- **Approach**: Start from random code (primordial soup) and observe emergent replication without pre-programming any replicators

## Features

- **Brainfuck interpreter** with circular tape and local interactions
- **2D spatial grid** where code fragments interact locally
- **Multiple complexity metrics** (Shannon entropy, high-order entropy, pattern diversity)
- **Replicator detection** to identify and track self-copying patterns
- **Configurable experiments** with presets based on research
- **Checkpoint system** for long-running simulations
- **Mutation system** (point mutations, insertions, deletions, duplications)

## Installation

Requirements:
- Python 3.8+
- NumPy

```bash
# Install dependencies
pip install numpy

# Clone or download this repository
git clone https://github.com/Jobbet789/SelfReplicators
cd SelfReplicators
```

## Quick Start

Run a simple demonstration:

```bash
python example.py
```

This will:
1. Initialize a 32×32 grid with 20% random code
2. Run 500 epochs of simulation
3. Detect emergent self-replicating patterns
4. Display metrics and top replicators

## Usage

### Basic Simulation

```python
from simulation import PrimordialSoup
from config import SimulationConfig

# Create configuration
config = SimulationConfig(
    grid_height=64,
    grid_width=64,
    initial_density=0.2,
    max_epochs=10000
)

# Run simulation
soup = PrimordialSoup(config)
results = soup.run()

# View results
print(f"Replicators found: {results['final_replicator_count']}")
print(f"Complexity score: {results['complexity_score']}")
```

### Using Presets

```python
from config import get_preset_config
from simulation import PrimordialSoup

# Available presets: 'minimal', 'standard', 'intensive', 'research'
config = get_preset_config('standard')
soup = PrimordialSoup(config)
results = soup.run()
```

### Running Experiments

```python
from simulation import run_experiment
from config import SimulationConfig

config = SimulationConfig(
    grid_height=128,
    grid_width=128,
    initial_density=0.4,
    max_epochs=50000
)

# Automatically saves results to experiments/my_experiment/
results = run_experiment(config, name="my_experiment")
```

### Checkpoints

```python
# Save checkpoint
soup.save_checkpoint("checkpoint.pkl")

# Load and continue
from simulation import PrimordialSoup
soup = PrimordialSoup.load_checkpoint("checkpoint.pkl")
soup.run(epochs=5000)  # Continue for 5000 more epochs
```

## Configuration Parameters

| Parameter | Description | Default | Research Basis |
|-----------|-------------|---------|----------------|
| `grid_height` | Height of 2D grid | 64 | - |
| `grid_width` | Width of 2D grid | 64 | - |
| `initial_density` | Initial code density (0-1) | 0.2 | Chou (1997): 10-50% optimal |
| `mutation_rate` | Probability of mutation (0-1) | 0.0001 | Agüera y Arcas: 0.024% |
| `interaction_radius` | Local interaction distance | 1 | Research: ≤2 limits propagation |
| `max_program_length` | Max code length to execute | 64 | Agüera y Arcas: 64 bytes |
| `max_steps_per_execution` | Max steps to prevent loops | 100 | - |
| `executions_per_epoch` | Programs run per epoch | 100 | - |
| `max_epochs` | Total simulation epochs | 10000 | Modern: >10^6 possible |

## Metrics Explanation

### Shannon Entropy
Measures disorder/randomness:
- **High**: Chaotic random state
- **Low**: Ordered, uniform replicators

### High-Order Entropy
Shannon entropy adjusted for compression:
- **Pre-transition**: Low (uniform tokens)
- **Takeover**: Peak (diversity from modifications)
- **Post-stabilization**: Moderate (stable diversity)

### Pattern Diversity
Ratio of unique patterns to total patterns (0-1)

### Code Length
Average length of contiguous non-empty code segments

### Complexity Score
Combined metric: unique replicators × average length × diversity

## Research Questions Addressed

1. **What is already known?** - Literature review of cellular automata research
2. **How to measure complexity?** - Entropy + code length + diversity metrics
3. **What conditions increase complexity?** - Density, mutation rate, interaction radius experiments
4. **How does complexity affect entropy?** - Tracking entropy transitions during takeover

## File Structure

```
SelfReplicators/
├── data/                       # Old test data
├── src/
    └── batch_experiments.py    # Batch experiment runner for testing different conditions.
    └── brainfuck.py            # Brainfuck interpreter and execution engine
    └── config.py               # Configuration management
    └── example.py              # Example usage and demos
    └── grid.py                 # 2D grid implementation
    └── metrics.py              # Complexity and entropy measurements
    └── simulation.py           # Main simulation engine
    └── test.py                 # Test before starting to verify if everything is working
├── test/                       # old test files (might remove later)
└── README.md                   # This file
```

## Experimental Configurations

### Density Experiments
```python
from config import get_density_experiment_configs

for name, config in get_density_experiment_configs():
    run_experiment(config, name=name)
# Tests: 10%, 20%, 40%, 60% density
```

### Mutation Rate Experiments
```python
from config import get_mutation_experiment_configs

for name, config in get_mutation_experiment_configs():
    run_experiment(config, name=name)
# Tests: 0%, 0.01%, 0.1%, 1% mutation rates
```

### Interaction Radius Experiments
```python
from config import get_radius_experiment_configs

for name, config in get_radius_experiment_configs():
    run_experiment(config, name=name)
# Tests: radius 1, 2, 3, 4
```

## Understanding Emergent Replication

Self-replicating structures emerge through:

1. **Random Initialization**: Grid filled with random Brainfuck commands
2. **Local Execution**: Programs execute at random positions
3. **Self-Modification**: Programs can read and write to the grid
4. **Concatenation**: Programs combine (A + B → AB)
5. **Replication**: Some combinations copy themselves
6. **Selection**: Successful replicators spread through the grid

Example replicator pattern from research:
```
[[{.>]-]]-]>.[[ 
```
This creates an infinite copying loop via read/write heads.

## Performance Notes

- **Minimal preset**: ~1-2 minutes (32×32, 1000 epochs)
- **Standard preset**: ~10-15 minutes (64×64, 10000 epochs)
- **Intensive preset**: Several hours (128×128, 50000 epochs)
- **Research preset**: Many hours (128×128, 100000 epochs)


## Future Extensions (GUI Coming)

- [ ] Real-time visualization
- [ ] Interactive controls
- [ ] Live metric graphs
- [ ] Grid editing
- [ ] Export animations

## Research References

- Agüera y Arcas, B. et al. (2024). "Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction"
- Chou, H. & Reggia, J.A. (1997). "Emergence of self-replicating structures in a cellular automata space"
- Sipper, M. (1998). "Fifty Years of Research on Self-Replication: An Overview"

## License

This is educational research code for a PWS project.

## Author

Job van der Veen

---

*"How complex can chaos become when it learns to copy itself?"*
