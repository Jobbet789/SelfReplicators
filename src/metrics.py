"""
Metrics for measuring complexity and entropy in the simulation.
Based on the research from Agüera y Arcas (2024) and Chou & Reggia (1997).
"""

import numpy as np
from collections import Counter
import zlib
from typing import Dict, List, Tuple


class ComplexityMetrics:
    """
    Measures complexity and entropy of the simulation state.
    """
    
    def __init__(self, grid):
        """
        Initialize metrics calculator.
        
        Args:
            grid: Grid instance to measure
        """
        self.grid = grid
        self.history = {
            'shannon_entropy': [],
            'high_order_entropy': [],
            'code_length': [],
            'replicator_count': [],
            'diversity_index': []
        }
    
    def shannon_entropy(self) -> float:
        """
        Calculate Shannon entropy of the grid.
        H = -Σ(p_i * log2(p_i))
        
        Returns:
            Shannon entropy value
        """
        # Flatten grid and count token frequencies
        flat_grid = self.grid.grid.flatten()
        total = len(flat_grid)
        
        # Count occurrences
        counts = Counter(flat_grid)
        
        # Calculate probabilities and entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def kolmogorov_complexity_estimate(self) -> float:
        """
        Estimate Kolmogorov complexity via compression (normalized).
        Lower compression ratio = higher complexity.
        
        Returns:
            Normalized compression ratio (0-1)
        """
        # Convert grid to string
        grid_string = ''.join(self.grid.grid.flatten())
        
        # Compress
        compressed = zlib.compress(grid_string.encode('utf-8'))
        
        # Normalize by original size
        original_size = len(grid_string)
        compressed_size = len(compressed)
        
        if original_size == 0:
            return 0.0
        
        return compressed_size / original_size
    
    def high_order_entropy(self) -> float:
        """
        Calculate high-order entropy: Shannon entropy - normalized Kolmogorov complexity.
        
        Pre-transition: Low (uniform tokens)
        Takeover: Peak (diversity from random modifications)
        Post-stabilization: Slight decrease (diverse but stable replicators)
        
        Returns:
            High-order entropy value
        """
        shannon = self.shannon_entropy()
        kolmogorov_norm = self.kolmogorov_complexity_estimate()
        
        # High-order entropy = Shannon - (1 - compression_ratio)
        # Higher compression ratio (lower Kolmogorov) = lower complexity
        # We want: high Shannon + high compression = high order
        hoe = shannon * kolmogorov_norm
        
        return hoe
    
    def pattern_diversity(self, pattern_length: int = 4) -> float:
        """
        Measure diversity of patterns in the grid.
        
        Args:
            pattern_length: Length of patterns to analyze
            
        Returns:
            Diversity index (0-1)
        """
        patterns = []
        
        # Collect horizontal patterns
        for row in self.grid.grid:
            row_str = ''.join(row)
            for i in range(len(row_str) - pattern_length + 1):
                pattern = row_str[i:i+pattern_length]
                if pattern.strip():  # Ignore empty patterns
                    patterns.append(pattern)
        
        if not patterns:
            return 0.0
        
        # Calculate diversity (unique patterns / total patterns)
        unique_patterns = len(set(patterns))
        total_patterns = len(patterns)
        
        return unique_patterns / total_patterns
    
    def average_code_length(self, sample_size: int = 100) -> float:
        """
        Calculate average length of contiguous non-empty code segments.
        
        Args:
            sample_size: Number of random positions to sample
            
        Returns:
            Average code length
        """
        lengths = []
        
        # Sample random positions
        for _ in range(sample_size):
            x = np.random.randint(0, self.grid.height)
            y = np.random.randint(0, self.grid.width)
            
            # Count contiguous non-empty cells
            length = 0
            current_y = y
            while length < self.grid.width:
                cell = self.grid.get_cell(x, current_y)
                if cell != ' ' and cell in self.grid.tokens:
                    length += 1
                    current_y = (current_y + 1) % self.grid.width
                else:
                    break
            
            if length > 0:
                lengths.append(length)
        
        return np.mean(lengths) if lengths else 0.0
    
    def density(self) -> float:
        """
        Calculate the current density of non-empty cells.
        
        Returns:
            Density (0-1)
        """
        flat_grid = self.grid.grid.flatten()
        non_empty = np.sum([1 for cell in flat_grid if cell != ' '])
        return non_empty / len(flat_grid)
    
    def record_metrics(self, replicator_count: int = 0):
        """
        Record current metrics to history.
        
        Args:
            replicator_count: Number of detected replicators
        """
        self.history['shannon_entropy'].append(self.shannon_entropy())
        self.history['high_order_entropy'].append(self.high_order_entropy())
        self.history['code_length'].append(self.average_code_length())
        self.history['replicator_count'].append(replicator_count)
        self.history['diversity_index'].append(self.pattern_diversity())
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of current state.
        
        Returns:
            Dictionary of metric values
        """
        return {
            'shannon_entropy': self.shannon_entropy(),
            'high_order_entropy': self.high_order_entropy(),
            'kolmogorov_estimate': self.kolmogorov_complexity_estimate(),
            'pattern_diversity': self.pattern_diversity(),
            'average_code_length': self.average_code_length(),
            'density': self.density()
        }
    
    def detect_takeover(self, window_size: int = 100) -> bool:
        """
        Detect if a replicator takeover has occurred.
        Indicated by a significant drop in high-order entropy.
        
        Args:
            window_size: Number of recent epochs to analyze
            
        Returns:
            True if takeover is detected
        """
        if len(self.history['high_order_entropy']) < window_size:
            return False
        
        recent = self.history['high_order_entropy'][-window_size:]
        older = self.history['high_order_entropy'][-2*window_size:-window_size] if len(self.history['high_order_entropy']) >= 2*window_size else recent
        
        # Check if recent entropy is significantly lower than older
        if len(older) > 0:
            recent_mean = np.mean(recent)
            older_mean = np.mean(older)
            
            # Takeover if entropy dropped by >30%
            if older_mean > 0 and recent_mean < 0.7 * older_mean:
                return True
        
        return False


class ReplicatorDetector:
    """
    Detects and tracks self-replicating structures in the grid.
    """
    
    def __init__(self, grid, min_pattern_length: int = 4):
        """
        Initialize replicator detector.
        
        Args:
            grid: Grid instance
            min_pattern_length: Minimum length to consider as a replicator
        """
        self.grid = grid
        self.min_pattern_length = min_pattern_length
        self.detected_replicators = []
    
    def find_repeating_patterns(self) -> List[Tuple[str, int]]:
        """
        Find repeating patterns in the grid.
        
        Returns:
            List of (pattern, count) tuples
        """
        patterns = Counter()
        
        # Scan horizontally
        for row in self.grid.grid:
            row_str = ''.join(row).strip()
            if len(row_str) < self.min_pattern_length:
                continue
            
            # Extract all substrings
            for length in range(self.min_pattern_length, len(row_str) + 1):
                for i in range(len(row_str) - length + 1):
                    pattern = row_str[i:i+length]
                    if pattern.strip() and ' ' not in pattern:
                        patterns[pattern] += 1
        
        # Filter for patterns that appear multiple times
        replicators = [(pattern, count) for pattern, count in patterns.items() 
                      if count >= 2]
        
        # Sort by count (most replicated first)
        replicators.sort(key=lambda x: x[1], reverse=True)
        
        return replicators
    
    def detect_replicators(self) -> int:
        """
        Detect current replicators and update tracking.
        
        Returns:
            Number of unique replicators detected
        """
        self.detected_replicators = self.find_repeating_patterns()
        return len(self.detected_replicators)
    
    def get_most_successful(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Get the most successful replicators.
        
        Args:
            top_n: Number of top replicators to return
            
        Returns:
            List of (pattern, count) tuples
        """
        return self.detected_replicators[:top_n]
    
    def get_complexity_score(self) -> float:
        """
        Calculate overall complexity score based on replicators.
        Score = (number of unique replicators) * (average length) * (diversity)
        
        Returns:
            Complexity score
        """
        if not self.detected_replicators:
            return 0.0
        
        num_unique = len(self.detected_replicators)
        avg_length = np.mean([len(pattern) for pattern, _ in self.detected_replicators])
        
        # Diversity: ratio of unique patterns to total occurrences
        total_occurrences = sum(count for _, count in self.detected_replicators)
        diversity = num_unique / total_occurrences if total_occurrences > 0 else 0
        
        return num_unique * avg_length * (1 + diversity)
