"""
Advanced analysis tools for studying emergent replication.
Statistical analysis, pattern matching, and evolution tracking.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

class SequenceAnalyzer:
    """
    Analyzes code sequences for interesting properties.
    """
    
    @staticmethod
    def calculate_compression_ratio(code: str) -> float:
        """
        Calculate how compressible the code is.
        
        Args:
            code: Code string
            
        Returns:
            Compression ratio (lower = more pattern/structure)
        """
        import zlib
        if not code:
            return 1.0
        compressed = zlib.compress(code.encode('utf-8'))
        return len(compressed) / len(code)
    
    @staticmethod
    def analyze_bracket_balance(code: str) -> Dict:
        """
        Analyze bracket pairing and nesting in Brainfuck code.
        
        Args:
            code: Brainfuck code
            
        Returns:
            Dictionary with bracket statistics
        """
        stack = []
        pairs = []
        max_depth = 0
        
        for i, char in enumerate(code):
            if char == '[':
                stack.append(i)
            elif char == ']':
                if stack:
                    open_pos = stack.pop()
                    pairs.append((open_pos, i))
                    max_depth = max(max_depth, len(stack) + 1)
        
        return {
            'balanced': len(stack) == 0,
            'unmatched_opens': len(stack),
            'max_nesting_depth': max_depth,
            'total_pairs': len(pairs),
            'average_loop_size': np.mean([j - i for i, j in pairs]) if pairs else 0
        }

class ReportGenerator:
    """
    Generate comprehensive research reports.
    """
    
    @staticmethod
    def generate_experiment_report(results: Dict, filename: str = None) -> str:
        """
        Generate a detailed experiment report.
        
        Args:
            results: Results dictionary from simulation
            filename: Optional filename to save to
            
        Returns:
            Report string
        """
        lines = []
        
        # Header
        lines.append("="*80)
        lines.append("PRIMORDIAL SOUP EXPERIMENT REPORT")
        lines.append("="*80)
        lines.append("")
        
        # Configuration
        lines.append("CONFIGURATION:")
        lines.append("-"*80)
        config = results['config']
        lines.append(f"Grid Size: {config['grid_height']}×{config['grid_width']}")
        lines.append(f"Initial Density: {config['initial_density']*100:.1f}%")
        lines.append(f"Mutation Rate: {config['mutation_rate']*100:.3f}%")
        lines.append(f"Interaction Radius: {config['interaction_radius']}")
        lines.append(f"Max Program Length: {config['max_program_length']}")
        lines.append(f"Total Epochs: {results['final_epoch']}")
        lines.append("")
        
        # Results
        lines.append("RESULTS:")
        lines.append("-"*80)
        lines.append(f"Total Executions: {results['total_executions']:,}")
        lines.append(f"Successful Executions: {results['successful_executions']:,}")
        lines.append(f"Success Rate: {results['success_rate']*100:.2f}%")
        lines.append(f"Final Replicator Count: {results['final_replicator_count']}")
        lines.append(f"Complexity Score: {results['complexity_score']:.2f}")
        lines.append(f"Takeover Detected: {results['takeover_detected']}")
        if results['takeover_detected']:
            lines.append(f"Takeover at Epoch: {results['takeover_epoch']}")
        lines.append("")
        
        # Top Replicators
        if results['top_replicators']:
            lines.append("TOP REPLICATORS:")
            lines.append("-"*80)
            for i, (pattern, count) in enumerate(results['top_replicators'][:10], 1):
                lines.append(f"{i:2d}. '{pattern}' (length={len(pattern)}, copies={count})")
            lines.append("")
        
        # Final Metrics
        lines.append("FINAL METRICS:")
        lines.append("-"*80)
        metrics = results['final_metrics']
        for key, value in metrics.items():
            lines.append(f"{key:30s}: {value:.4f}")
        lines.append("")
        
        # Metrics History Summary
        history = results['metrics_history']
        if history['shannon_entropy']:
            lines.append("METRICS EVOLUTION:")
            lines.append("-"*80)
            lines.append(f"Shannon Entropy:   {history['shannon_entropy'][0]:.3f} → {history['shannon_entropy'][-1]:.3f}")
            lines.append(f"High-Order Entropy: {history['high_order_entropy'][0]:.3f} → {history['high_order_entropy'][-1]:.3f}")
            lines.append(f"Code Length:       {history['code_length'][0]:.2f} → {history['code_length'][-1]:.2f}")
            lines.append(f"Replicators:       {history['replicator_count'][0]} → {history['replicator_count'][-1]}")
            lines.append("")
        
        lines.append("="*80)
        
        report = '\n'.join(lines)
        
        # Save if filename provided
        if filename:
            with open(filename, 'w') as f:
                f.write(report)
            print(f"Report saved to: {filename}")
        
        return report


# Convenience functions

def analyze_single_experiment(results: Dict) -> None:
    """
    Perform comprehensive analysis of a single experiment.
    
    Args:
        results: Results dictionary from simulation
    """
    print("\n" + "="*80)
    print("EXPERIMENT ANALYSIS")
    print("="*80)
    
    # Generate report
    report = ReportGenerator.generate_experiment_report(results)
    print(report)
    
    # Analyze top replicators
    if results['top_replicators']:
        print("\n" + "="*80)
        print("TOP REPLICATORS BY COPY COUNT (current behaviour)")
        print("="*80)
        for i, (pattern, count) in enumerate(results['top_replicators'][:10], 1):
            print(f"{i:2d}. '{pattern}' × {count:3d} (len={len(pattern):2d})")

        print("\n" + "="*80)
        print("TOP REPLICATORS SORTED BY ESTIMATED STRUCTURAL COMPLEXITY")
        print("="*80)
        
        # Header – make sure total width matches data rows
        print(f"{'Rank':<4} {'Pattern':<20} {'Copies':<8} {'Len':<5} {'Cmds':<6} {'Depth':<6} {'Compr':<7} {'Score':<8}")
        print("-" * 74)   # adjust if you change widths

        scored = []
        for pattern, count in results['top_replicators'][:30]:
            length = len(pattern)
            
            # distinct commands
            cmds = len(set(c for c in pattern if c in '+-<>.,[]'))
            
            # bracket info
            bracket_info = SequenceAnalyzer.analyze_bracket_balance(pattern)
            max_depth = bracket_info['max_nesting_depth']
            
            # compression
            compr_ratio = SequenceAnalyzer.calculate_compression_ratio(pattern)
            complexity_from_compr = 1.0 - compr_ratio if compr_ratio <= 1 else 0
            
            # score (you can still tune these weights)
            score = (
                length * 0.4 +
                cmds * 3.0 +
                max_depth * 8.0 +
                complexity_from_compr * 30.0
            )
            
            scored.append((score, pattern, count, length, cmds, max_depth, compr_ratio))
        
        scored.sort(reverse=True)
        
        for i, (score, pattern, count, length, cmds, depth, compr) in enumerate(scored[:10], 1):
            # Truncate pattern and add ellipsis if needed
            short = (pattern[:17] + "…") if len(pattern) > 17 else pattern
            
            # Now align everything properly
            print(
                f"{i:>2}  "                             # rank, right-aligned, width 2 + space
                f"'{short:<17}' "                       # pattern left-aligned in 19 chars (quote + 17 + quote)
                f"{count:>6}  "                         # copies right-aligned
                f"{length:>4}  " 
                f"{cmds:>4}  " 
                f"{depth:>4}  " 
                f"{compr:>6.3f}  " 
                f"{score:>7.1f}"
            )
    if 'metrics_history' in results and results['metrics_history']:
        print("\n" + "="*80)
        print("GLOBAL METRICS EVOLUTION OVER TIME")
        print("="*80)
        
        mh = results['metrics_history']
        n_steps = len(mh.get('shannon_entropy', []))
        print(f"Recorded over {n_steps} points (up to epoch {results['final_epoch']})")
        
        for key in ['shannon_entropy', 'high_order_entropy', 'code_length', 
                    'replicator_count', 'diversity_index']:
            if key in mh and mh[key]:
                vals = mh[key]
                start, end = vals[0], vals[-1]
                peak = max(vals)
                peak_idx = np.argmax(vals)
                print(f"{key:18}: {start:.3f} → {end:.3f}   (peak {peak:.3f} around step {peak_idx})")
        
        # Simple phase interpretation
        if results['takeover_detected']:
            print(f"\nTakeover phase detected around epoch {results['takeover_epoch']}")
        if results['final_replicator_count'] > 1000:
            print("→ Strong replicator dominance in late simulation")
        elif results['final_replicator_count'] > 100:
            print("→ Moderate replicator activity")

    # ────────────────────────────────────────────────────────────────
    #   VISUALIZATIONS – Metrics over time
    # ────────────────────────────────────────────────────────────────
    if 'metrics_history' in results and results['metrics_history']:
        mh = results['metrics_history']
        
        # Prepare time axis (assuming uniform sampling; step ≈ epoch / (n-1))
        n = len(mh.get('shannon_entropy', []))
        if n < 2:
            print("\nNot enough data points for plotting.")
        else:
            epochs = np.linspace(0, results['final_epoch'], n)
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            fig.suptitle("Emergent Replication Dynamics", fontsize=16, y=0.98)
            
            # Replicator count
            if 'replicator_count' in mh and mh['replicator_count']:
                plt.figure(figsize=(9, 5))
                plt.plot(epochs, mh['replicator_count'], color='darkgreen', lw=2)
                plt.title("Replicator Population Growth")
                plt.xlabel("Epoch")
                plt.ylabel("Replicator Count")
                plt.grid(True, alpha=0.3)
                if results['takeover_detected']:
                    plt.axvline(results['takeover_epoch'], color='red', ls='--', 
                              alpha=0.7, label=f"Takeover @ {results['takeover_epoch']}")
                    plt.legend()
                plt.savefig("../data/replicators_over_time.png", dpi=140, bbox_inches='tight')
                plt.close()
                print("Saved: replicators_over_time.png")

            # Entropies
            if 'shannon_entropy' in mh or 'high_order_entropy' in mh:
                plt.figure(figsize=(9, 5))
                if 'shannon_entropy' in mh:
                    plt.plot(epochs, mh['shannon_entropy'], label='Shannon', color='blue')
                if 'high_order_entropy' in mh:
                    plt.plot(epochs, mh['high_order_entropy'], label='High-Order', color='orange')
                plt.title("Entropy Evolution")
                plt.xlabel("Epoch")
                plt.ylabel("Entropy")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig("../data/entropy_evolution.png", dpi=140, bbox_inches='tight')
                plt.close()
                print("Saved: entropy_evolution.png")

            # Average code length
            if 'code_length' in mh and mh['code_length']:
                plt.figure(figsize=(9, 5))
                plt.plot(epochs, mh['code_length'], color='purple', lw=2)
                plt.title("Average Program Length")
                plt.xlabel("Epoch")
                plt.ylabel("Avg Length")
                plt.grid(True, alpha=0.3)
                plt.savefig("../data/avg_code_length.png", dpi=140, bbox_inches='tight')
                plt.close()
                print("Saved: avg_code_length.png")


if __name__ == "__main__":
    print("Advanced Analysis Tools Module")
    
    result_file = "experiments/long_term_complexity/results.pkl" # hard code change soon!!!!!!

    try:
        with open(result_file, 'rb') as f:
            results = pickle.load(f)
        analyze_single_experiment(results)
    except FileNotFoundError:
        print(f"Result file not found: {result_file}")
        print("Run the long-term complexity experiment first to generate results.")
