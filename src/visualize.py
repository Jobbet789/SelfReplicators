"""
Visualization utilities for the simulation.
These will be useful for building the GUI later.
"""

import numpy as np
from typing import List, Tuple, Dict


class GridVisualizer:
    """
    Utilities for visualizing the grid state.
    Can be used for terminal output now and GUI later.
    """
    
    def __init__(self, grid):
        """
        Initialize visualizer.
        
        Args:
            grid: Grid instance to visualize
        """
        self.grid = grid
        
        # Color codes for terminal (ANSI)
        self.colors = {
            '+': '\033[92m',  # Green
            '-': '\033[91m',  # Red
            '<': '\033[94m',  # Blue
            '>': '\033[94m',  # Blue
            '.': '\033[93m',  # Yellow
            ',': '\033[93m',  # Yellow
            '[': '\033[95m',  # Magenta
            ']': '\033[95m',  # Magenta
            ' ': '\033[90m',  # Gray
        }
        self.reset = '\033[0m'
    
    def render_colored(self, max_rows: int = None) -> str:
        """
        Render grid with ANSI colors for terminal.
        
        Args:
            max_rows: Maximum rows to display (None for all)
            
        Returns:
            Colored string representation
        """
        rows_to_show = min(max_rows or self.grid.height, self.grid.height)
        lines = []
        
        for i in range(rows_to_show):
            row_str = ""
            for j in range(self.grid.width):
                cell = self.grid.grid[i, j]
                color = self.colors.get(cell, '')
                row_str += f"{color}{cell}{self.reset}"
            lines.append(f"{i:3d}: {row_str}")
        
        if rows_to_show < self.grid.height:
            lines.append(f"... ({self.grid.height - rows_to_show} more rows)")
        
        return '\n'.join(lines)
    
    def render_ascii_box(self, row: int, col: int, size: int = 5) -> str:
        """
        Render a small box around a position.
        Useful for showing local neighborhoods.
        
        Args:
            row, col: Center position
            size: Size of box (radius)
            
        Returns:
            String representation of the box
        """
        lines = []
        
        for i in range(row - size, row + size + 1):
            row_chars = []
            for j in range(col - size, col + size + 1):
                if i == row and j == col:
                    # Highlight center
                    cell = self.grid.get_cell(i, j)
                    row_chars.append(f"[{cell}]")
                else:
                    cell = self.grid.get_cell(i, j)
                    row_chars.append(f" {cell} ")
            lines.append(''.join(row_chars))
        
        return '\n'.join(lines)
    
    def get_density_heatmap(self, block_size: int = 4) -> np.ndarray:
        """
        Calculate local density in blocks for heatmap visualization.
        
        Args:
            block_size: Size of blocks to measure
            
        Returns:
            2D array of densities
        """
        h_blocks = self.grid.height // block_size
        w_blocks = self.grid.width // block_size
        heatmap = np.zeros((h_blocks, w_blocks))
        
        for i in range(h_blocks):
            for j in range(w_blocks):
                # Count non-empty cells in block
                block = self.grid.grid[
                    i*block_size:(i+1)*block_size,
                    j*block_size:(j+1)*block_size
                ]
                non_empty = np.sum([1 for cell in block.flatten() if cell != ' '])
                heatmap[i, j] = non_empty / (block_size * block_size)
        
        return heatmap
    
    def render_density_ascii(self, block_size: int = 4) -> str:
        """
        Render density heatmap as ASCII art.
        
        Args:
            block_size: Size of blocks
            
        Returns:
            ASCII heatmap string
        """
        heatmap = self.get_density_heatmap(block_size)
        
        # ASCII gradient
        chars = ' .:-=+*#%@'
        
        lines = []
        for row in heatmap:
            line = ''
            for density in row:
                idx = int(density * (len(chars) - 1))
                line += chars[idx]
            lines.append(line)
        
        return '\n'.join(lines)


class MetricsVisualizer:
    """
    Utilities for visualizing metrics over time.
    Prepares data for plotting (for future GUI).
    """
    
    def __init__(self, metrics):
        """
        Initialize metrics visualizer.
        
        Args:
            metrics: ComplexityMetrics instance
        """
        self.metrics = metrics
    
    def get_plot_data(self) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get all metrics formatted for plotting.
        
        Returns:
            Dictionary of metric_name -> [(epoch, value), ...]
        """
        plot_data = {}
        
        for metric_name, values in self.metrics.history.items():
            # Pair each value with its epoch number
            epochs = list(range(len(values)))
            plot_data[metric_name] = list(zip(epochs, values))
        
        return plot_data
    
    def get_ascii_plot(self, metric_name: str, height: int = 10, 
                       width: int = 60) -> str:
        """
        Create a simple ASCII plot of a metric over time.
        
        Args:
            metric_name: Name of metric to plot
            height: Height of plot in characters
            width: Width of plot in characters
            
        Returns:
            ASCII plot string
        """
        if metric_name not in self.metrics.history:
            return f"Metric '{metric_name}' not found"
        
        values = self.metrics.history[metric_name]
        if not values:
            return "No data to plot"
        
        # Normalize values to fit in height
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1
        
        # Create plot grid
        plot = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot points
        for i, value in enumerate(values):
            x = int((i / len(values)) * (width - 1))
            y = height - 1 - int(((value - min_val) / range_val) * (height - 1))
            y = max(0, min(height - 1, y))
            plot[y][x] = '*'
        
        # Convert to string
        lines = [''.join(row) for row in plot]
        
        # Add axis labels
        result = f"{metric_name}\n"
        result += f"Max: {max_val:.3f} |" + lines[0] + "\n"
        for line in lines[1:-1]:
            result += "            |" + line + "\n"
        result += f"Min: {min_val:.3f} |" + lines[-1] + "\n"
        result += "            +" + "-" * width + "\n"
        result += f"            0{' ' * (width-10)}epochs\n"
        
        return result
    
    def print_summary_table(self):
        """Print a formatted table of current metrics."""
        summary = self.metrics.get_summary()
        
        print("\n" + "="*60)
        print("CURRENT METRICS SUMMARY")
        print("="*60)
        
        for metric, value in summary.items():
            print(f"{metric:25s}: {value:10.4f}")
        
        print("="*60)


class ReplicatorVisualizer:
    """
    Utilities for visualizing replicators.
    """
    
    def __init__(self, detector):
        """
        Initialize replicator visualizer.
        
        Args:
            detector: ReplicatorDetector instance
        """
        self.detector = detector
    
    def render_top_replicators(self, top_n: int = 10) -> str:
        """
        Render a formatted list of top replicators.
        
        Args:
            top_n: Number of replicators to show
            
        Returns:
            Formatted string
        """
        replicators = self.detector.get_most_successful(top_n)
        
        if not replicators:
            return "No replicators found yet."
        
        lines = [f"\nTop {len(replicators)} Replicators:"]
        lines.append("="*70)
        lines.append(f"{'Rank':<6} {'Pattern':<30} {'Copies':<8} {'Length':<8}")
        lines.append("-"*70)
        
        for i, (pattern, count) in enumerate(replicators, 1):
            # Truncate long patterns
            display_pattern = pattern if len(pattern) <= 28 else pattern[:25] + "..."
            lines.append(f"{i:<6} {display_pattern:<30} {count:<8} {len(pattern):<8}")
        
        lines.append("="*70)
        return '\n'.join(lines)
    
    def show_replicator_diversity(self) -> str:
        """
        Show diversity statistics of replicators.
        
        Returns:
            Formatted string
        """
        if not self.detector.detected_replicators:
            return "No replicators to analyze."
        
        # Calculate statistics
        lengths = [len(pattern) for pattern, _ in self.detector.detected_replicators]
        counts = [count for _, count in self.detector.detected_replicators]
        
        lines = ["\nReplicator Diversity Analysis:"]
        lines.append("="*50)
        lines.append(f"Total unique patterns: {len(self.detector.detected_replicators)}")
        lines.append(f"Total copies: {sum(counts)}")
        lines.append(f"Average length: {np.mean(lengths):.1f} tokens")
        lines.append(f"Length range: {min(lengths)} - {max(lengths)}")
        lines.append(f"Most replicated: {max(counts)} copies")
        lines.append(f"Complexity score: {self.detector.get_complexity_score():.2f}")
        lines.append("="*50)
        
        return '\n'.join(lines)


def create_full_report(soup) -> str:
    """
    Create a comprehensive text report of simulation state.
    
    Args:
        soup: PrimordialSoup instance
        
    Returns:
        Full report string
    """
    lines = []
    
    # Header
    lines.append("\n" + "="*70)
    lines.append("PRIMORDIAL SOUP SIMULATION REPORT")
    lines.append("="*70)
    
    # Simulation info
    lines.append(f"\nEpoch: {soup.current_epoch} / {soup.config.max_epochs}")
    lines.append(f"Total executions: {soup.total_executions:,}")
    lines.append(f"Success rate: {soup.successful_executions/soup.total_executions*100:.1f}%")
    
    # Metrics
    metrics_viz = MetricsVisualizer(soup.metrics)
    lines.append("\n" + "-"*70)
    summary = soup.metrics.get_summary()
    lines.append("CURRENT METRICS:")
    for key, value in summary.items():
        lines.append(f"  {key}: {value:.4f}")
    
    # Replicators
    lines.append("\n" + "-"*70)
    rep_viz = ReplicatorVisualizer(soup.replicator_detector)
    lines.append(rep_viz.render_top_replicators(5))
    lines.append(rep_viz.show_replicator_diversity())
    
    # Grid preview
    lines.append("\n" + "-"*70)
    lines.append("GRID PREVIEW (first 10 rows):")
    grid_viz = GridVisualizer(soup.grid)
    lines.append(grid_viz.render_colored(max_rows=10))
    
    # Density map
    lines.append("\n" + "-"*70)
    lines.append("DENSITY HEATMAP:")
    lines.append(grid_viz.render_density_ascii(block_size=4))
    
    lines.append("\n" + "="*70)
    
    return '\n'.join(lines)


# Example usage functions for testing

def demo_visualizations():
    """Demonstrate all visualization capabilities."""
    from grid import Grid
    from metrics import ComplexityMetrics, ReplicatorDetector
    
    print("\nDemonstrating Visualization Tools")
    print("="*70)
    
    # Create test grid
    grid = Grid(height=20, width=40, density=0.3)
    
    # Grid visualization
    print("\n1. COLORED GRID:")
    viz = GridVisualizer(grid)
    print(viz.render_colored(max_rows=10))
    
    # Density heatmap
    print("\n2. DENSITY HEATMAP:")
    print(viz.render_density_ascii(block_size=4))
    
    # Local box
    print("\n3. LOCAL NEIGHBORHOOD (around position 5,5):")
    print(viz.render_ascii_box(5, 5, size=3))
    
    # Metrics
    print("\n4. METRICS SUMMARY:")
    metrics = ComplexityMetrics(grid)
    metrics.record_metrics(replicator_count=5)
    metrics.record_metrics(replicator_count=8)
    metrics.record_metrics(replicator_count=12)
    
    metrics_viz = MetricsVisualizer(metrics)
    metrics_viz.print_summary_table()
    
    # ASCII plot
    print("\n5. ASCII PLOT (replicator count over time):")
    print(metrics_viz.get_ascii_plot('replicator_count'))
    
    print("\n" + "="*70)
    print("Visualization demo complete!")


if __name__ == "__main__":
    demo_visualizations()
