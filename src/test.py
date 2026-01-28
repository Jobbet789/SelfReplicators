"""
Test script to verify all components are working correctly.
"""

from grid import Grid
from brainfuck import BrainfuckInterpreter, GridBrainfuckEngine
from metrics import ComplexityMetrics, ReplicatorDetector
from config import SimulationConfig, get_preset_config
from simulation import PrimordialSoup

def test_grid():
    """Test Grid functionality."""
    print("Testing Grid...")
    grid = Grid(height=10, width=10, density=0.3)
    assert grid.grid.shape == (10, 10)
    assert len(grid.tokens) == 8
    
    # Test get/set
    grid.set_cell(0, 0, '+')
    assert grid.get_cell(0, 0) == '+'
    
    print("✓ Grid tests passed")

def test_brainfuck():
    """Test Brainfuck interpreter."""
    print("Testing Brainfuck Interpreter...")
    interpreter = BrainfuckInterpreter(tape_size=100, max_steps=1000)
    
    # Test basic operations
    output = []
    code = "+++++++++."  # Output 9
    success, steps = interpreter.execute(code, output_buffer=output)
    assert success
    assert output[0] == 9
    
    # Test loops
    interpreter.reset()
    interpreter.tape[0] = 5
    code = "[->+<]"  # Move value from cell 0 to cell 1
    success, steps = interpreter.execute(code)
    assert interpreter.tape[0] == 0
    assert interpreter.tape[1] == 5
    
    print("✓ Brainfuck tests passed")

def test_metrics():
    """Test metrics calculation."""
    print("Testing Metrics...")
    grid = Grid(height=10, width=10, density=0.3)
    metrics = ComplexityMetrics(grid)
    
    # Test entropy calculation
    entropy = metrics.shannon_entropy()
    assert entropy > 0
    
    # Test density
    density = metrics.density()
    assert 0 <= density <= 1
    
    # Test recording
    metrics.record_metrics(replicator_count=5)
    assert len(metrics.history['shannon_entropy']) == 1
    assert metrics.history['replicator_count'][0] == 5
    
    print("✓ Metrics tests passed")

def test_replicator_detection():
    """Test replicator detection."""
    print("Testing Replicator Detection...")
    grid = Grid(height=10, width=10, density=0.0)
    
    # Manually create a repeating pattern
    pattern = "++[->+<]"
    for i in range(len(pattern)):
        grid.set_cell(0, i, pattern[i])
        grid.set_cell(1, i, pattern[i])
        grid.set_cell(2, i, pattern[i])
    
    detector = ReplicatorDetector(grid, min_pattern_length=4)
    count = detector.detect_replicators()
    
    assert count > 0
    top = detector.get_most_successful(1)
    assert len(top) > 0
    
    print(f"✓ Replicator detection tests passed (found {count} patterns)")

def test_config():
    """Test configuration."""
    print("Testing Configuration...")
    
    # Test default config
    config = SimulationConfig()
    assert config.validate()
    
    # Test presets
    for preset in ['minimal', 'standard', 'intensive', 'research']:
        config = get_preset_config(preset)
        assert config.validate()
    
    # Test save/load
    config.save('/tmp/test_config.json')
    loaded = SimulationConfig.load('/tmp/test_config.json')
    assert loaded.grid_height == config.grid_height
    
    print("✓ Configuration tests passed")

def test_simulation():
    """Test full simulation."""
    print("Testing Simulation...")
    
    config = SimulationConfig(
        grid_height=16,
        grid_width=16,
        initial_density=0.3,
        max_epochs=100,
        executions_per_epoch=20,
        record_interval=20
    )
    
    soup = PrimordialSoup(config)
    
    # Run a few epochs
    soup.run(epochs=100)
    
    assert soup.current_epoch == 100
    assert soup.total_executions > 0
    
    # Test checkpoint
    soup.save_checkpoint('/tmp/test_checkpoint.pkl')
    loaded = PrimordialSoup.load_checkpoint('/tmp/test_checkpoint.pkl')
    assert loaded.current_epoch == 100
    
    print("✓ Simulation tests passed")

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    print()
    
    test_grid()
    test_brainfuck()
    test_metrics()
    test_replicator_detection()
    test_config()
    test_simulation()
    
    print()
    print("="*60)
    print("ALL TESTS PASSED!")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()
