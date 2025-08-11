#!/usr/bin/env python3
"""
Performance comparison script for release validation.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
import statistics


def load_benchmark_results(file_path: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading benchmark results from {file_path}: {e}")
        return {}


def extract_benchmark_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract benchmark metrics from results."""
    metrics = {}
    
    # Handle different benchmark result formats
    if 'benchmarks' in results:
        for benchmark in results['benchmarks']:
            name = benchmark.get('name', 'unknown')
            # Extract timing information
            if 'value' in benchmark:
                metrics[name] = benchmark['value']
            elif 'mean' in benchmark:
                metrics[name] = benchmark['mean']
            elif 'median' in benchmark:
                metrics[name] = benchmark['median']
    
    return metrics


def compare_benchmarks(baseline_metrics: Dict[str, float], 
                      current_metrics: Dict[str, float],
                      threshold: float) -> Dict[str, Any]:
    """Compare benchmark results and check for regressions."""
    comparison = {
        'passed': True,
        'regressions': [],
        'improvements': [],
        'summary': {}
    }
    
    all_benchmarks = set(baseline_metrics.keys()) | set(current_metrics.keys())
    
    for benchmark in all_benchmarks:
        baseline_value = baseline_metrics.get(benchmark)
        current_value = current_metrics.get(benchmark)
        
        if baseline_value is None:
            comparison['summary'][benchmark] = {
                'status': 'new',
                'current': current_value,
                'change': None
            }
            continue
        
        if current_value is None:
            comparison['summary'][benchmark] = {
                'status': 'removed',
                'baseline': baseline_value,
                'change': None
            }
            continue
        
        # Calculate performance ratio (lower is better for timing)
        ratio = current_value / baseline_value
        change_percent = (ratio - 1.0) * 100
        
        status = 'maintained'
        if ratio > (1.0 / threshold):  # Performance regression
            status = 'regression'
            comparison['regressions'].append({
                'benchmark': benchmark,
                'baseline': baseline_value,
                'current': current_value,
                'ratio': ratio,
                'change_percent': change_percent
            })
            comparison['passed'] = False
        elif ratio < threshold:  # Performance improvement
            status = 'improvement'
            comparison['improvements'].append({
                'benchmark': benchmark,
                'baseline': baseline_value,
                'current': current_value,
                'ratio': ratio,
                'change_percent': change_percent
            })
        
        comparison['summary'][benchmark] = {
            'status': status,
            'baseline': baseline_value,
            'current': current_value,
            'ratio': ratio,
            'change_percent': change_percent
        }
    
    return comparison


def generate_comparison_report(comparison: Dict[str, Any]) -> str:
    """Generate a human-readable comparison report."""
    report = []
    report.append("# Performance Comparison Report")
    report.append("")
    
    if comparison['passed']:
        report.append("✅ **Performance validation PASSED**")
    else:
        report.append("❌ **Performance validation FAILED**")
    
    report.append("")
    report.append("## Summary")
    report.append(f"- Total benchmarks: {len(comparison['summary'])}")
    report.append(f"- Regressions: {len(comparison['regressions'])}")
    report.append(f"- Improvements: {len(comparison['improvements'])}")
    report.append("")
    
    if comparison['regressions']:
        report.append("## ❌ Performance Regressions")
        for regression in comparison['regressions']:
            report.append(f"- **{regression['benchmark']}**: {regression['change_percent']:+.2f}% "
                         f"({regression['baseline']:.3f}s → {regression['current']:.3f}s)")
        report.append("")
    
    if comparison['improvements']:
        report.append("## ✅ Performance Improvements")
        for improvement in comparison['improvements']:
            report.append(f"- **{improvement['benchmark']}**: {improvement['change_percent']:+.2f}% "
                         f"({improvement['baseline']:.3f}s → {improvement['current']:.3f}s)")
        report.append("")
    
    report.append("## Detailed Results")
    for benchmark, data in comparison['summary'].items():
        status_icon = {
            'regression': '❌',
            'improvement': '✅',
            'maintained': '➖',
            'new': '🆕',
            'removed': '🗑️'
        }.get(data['status'], '❓')
        
        if data['change'] is not None:
            report.append(f"- {status_icon} **{benchmark}**: {data['change_percent']:+.2f}%")
        else:
            report.append(f"- {status_icon} **{benchmark}**: {data['status']}")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Compare performance benchmarks')
    parser.add_argument('baseline_inference', type=Path, help='Baseline inference benchmark results')
    parser.add_argument('current_inference', type=Path, help='Current inference benchmark results')
    parser.add_argument('baseline_kernels', type=Path, help='Baseline kernels benchmark results')
    parser.add_argument('current_kernels', type=Path, help='Current kernels benchmark results')
    parser.add_argument('--output', type=Path, help='Output comparison results to JSON file')
    parser.add_argument('--threshold', type=float, default=0.95, 
                       help='Performance threshold (0.95 = allow 5% regression)')
    
    args = parser.parse_args()
    
    # Load benchmark results
    baseline_inference = load_benchmark_results(args.baseline_inference)
    current_inference = load_benchmark_results(args.current_inference)
    baseline_kernels = load_benchmark_results(args.baseline_kernels)
    current_kernels = load_benchmark_results(args.current_kernels)
    
    # Extract metrics
    baseline_inference_metrics = extract_benchmark_metrics(baseline_inference)
    current_inference_metrics = extract_benchmark_metrics(current_inference)
    baseline_kernels_metrics = extract_benchmark_metrics(baseline_kernels)
    current_kernels_metrics = extract_benchmark_metrics(current_kernels)
    
    # Combine all metrics
    baseline_metrics = {**baseline_inference_metrics, **baseline_kernels_metrics}
    current_metrics = {**current_inference_metrics, **current_kernels_metrics}
    
    # Compare benchmarks
    comparison = compare_benchmarks(baseline_metrics, current_metrics, args.threshold)
    
    # Generate report
    report = generate_comparison_report(comparison)
    print(report)
    
    # Save results if output file specified
    if args.output:
        comparison['report'] = report
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison results saved to: {args.output}")
    
    # Exit with error code if performance regressions detected
    if not comparison['passed']:
        print(f"\n❌ Performance validation failed: {len(comparison['regressions'])} regressions detected")
        sys.exit(1)
    
    print("\n✅ Performance validation passed")


if __name__ == '__main__':
    main()