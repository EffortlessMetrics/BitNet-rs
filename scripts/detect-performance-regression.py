#!/usr/bin/env python3
"""
Performance regression detection for BitNet.rs

This script compares current benchmark results against established baselines
and detects performance regressions, improvements, or anomalies.
"""

import json
import sys
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    IMPROVEMENT = "improvement"


@dataclass
class PerformanceAlert:
    """Performance alert with context"""
    level: AlertLevel
    metric: str
    message: str
    current_value: float
    baseline_value: float
    change_percent: float
    threshold: float


@dataclass
class RegressionResult:
    """Result of regression analysis"""
    status: str  # "pass", "warning", "critical", "improved"
    alerts: List[PerformanceAlert]
    summary: Dict[str, Any]


class PerformanceRegressor:
    """Detects performance regressions in benchmark data"""

    def __init__(self, baselines_file: str = "crossval/baselines.json"):
        self.baselines_file = Path(baselines_file)
        self.baselines = self._load_baselines()

    def _load_baselines(self) -> Dict[str, Any]:
        """Load baseline performance data"""
        try:
            with open(self.baselines_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load baselines from {self.baselines_file}: {e}")
            return {}

    def _get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get performance thresholds from baselines"""
        default_thresholds = {
            "critical": {
                "throughput_decrease_percent": 15.0,
                "latency_increase_percent": 25.0,
                "memory_increase_percent": 30.0,
                "accuracy_decrease": 0.005
            },
            "warning": {
                "throughput_decrease_percent": 8.0,
                "latency_increase_percent": 15.0,
                "memory_increase_percent": 20.0,
                "accuracy_decrease": 0.002
            },
            "improvement": {
                "throughput_increase_percent": 5.0,
                "latency_decrease_percent": 5.0,
                "memory_decrease_percent": 5.0,
                "accuracy_increase": 0.0005
            }
        }

        # Merge with loaded thresholds
        loaded_thresholds = self.baselines.get("alerts", {})
        for level in default_thresholds:
            if level in loaded_thresholds:
                default_thresholds[level].update(loaded_thresholds[level])

        return default_thresholds

    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from benchmark results"""
        metrics = {}

        # Extract from Rust implementation results
        rust_results = results.get("rust_implementation", {})
        if rust_results:
            # Throughput metrics
            if "throughput_tokens_per_second" in rust_results:
                metrics["throughput_tokens_per_second"] = rust_results["throughput_tokens_per_second"]

            # Latency metrics
            for latency_key in ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]:
                if latency_key in rust_results:
                    metrics[latency_key] = rust_results[latency_key]

            # Memory metrics
            if "memory_usage_mb" in rust_results:
                metrics["memory_usage_mb"] = rust_results["memory_usage_mb"]

            # Accuracy metrics
            if "accuracy_score" in rust_results:
                metrics["accuracy_score"] = rust_results["accuracy_score"]

        # Extract from comparison benchmark results (benchmark_comparison.py format)
        if "rust" in results:
            rust_comparison = results["rust"]
            if "mean" in rust_comparison:
                # Convert seconds to tokens per second (rough estimate)
                mean_time = rust_comparison["mean"]
                if mean_time > 0:
                    # Assume 32 tokens generated (default from benchmark script)
                    tokens = 32
                    metrics["comparison_throughput_tokens_per_second"] = tokens / mean_time
                    metrics["comparison_latency_ms"] = mean_time * 1000

        # Extract from Criterion benchmark results
        benchmarks = results.get("benchmarks", [])
        if benchmarks:
            # Calculate average benchmark time
            valid_benchmarks = [
                b for b in benchmarks
                if "mean" in b and "estimate" in b["mean"] and b["mean"]["estimate"] > 0
            ]

            if valid_benchmarks:
                estimates = [b["mean"]["estimate"] for b in valid_benchmarks]
                avg_estimate_ns = statistics.mean(estimates)
                metrics["criterion_average_time_ns"] = avg_estimate_ns
                metrics["criterion_average_time_ms"] = avg_estimate_ns / 1_000_000

        return metrics

    def _calculate_change_percent(self, current: float, baseline: float) -> float:
        """Calculate percentage change from baseline to current"""
        if baseline == 0:
            return 0.0
        return ((current - baseline) / baseline) * 100

    def _check_throughput_regression(self, current_metrics: Dict[str, float],
                                   baseline_metrics: Dict[str, float],
                                   thresholds: Dict[str, Dict[str, float]]) -> List[PerformanceAlert]:
        """Check for throughput-related regressions"""
        alerts = []

        for metric_key in ["throughput_tokens_per_second", "comparison_throughput_tokens_per_second"]:
            if metric_key not in current_metrics or metric_key not in baseline_metrics:
                continue

            current = current_metrics[metric_key]
            baseline = baseline_metrics[metric_key]
            change_percent = self._calculate_change_percent(current, baseline)

            # Check for critical regression
            if change_percent < -thresholds["critical"]["throughput_decrease_percent"]:
                alerts.append(PerformanceAlert(
                    level=AlertLevel.CRITICAL,
                    metric=metric_key,
                    message=f"CRITICAL: Throughput decreased by {abs(change_percent):.1f}%",
                    current_value=current,
                    baseline_value=baseline,
                    change_percent=change_percent,
                    threshold=thresholds["critical"]["throughput_decrease_percent"]
                ))
            # Check for warning regression
            elif change_percent < -thresholds["warning"]["throughput_decrease_percent"]:
                alerts.append(PerformanceAlert(
                    level=AlertLevel.WARNING,
                    metric=metric_key,
                    message=f"WARNING: Throughput decreased by {abs(change_percent):.1f}%",
                    current_value=current,
                    baseline_value=baseline,
                    change_percent=change_percent,
                    threshold=thresholds["warning"]["throughput_decrease_percent"]
                ))
            # Check for improvement
            elif change_percent > thresholds["improvement"]["throughput_increase_percent"]:
                alerts.append(PerformanceAlert(
                    level=AlertLevel.IMPROVEMENT,
                    metric=metric_key,
                    message=f"IMPROVEMENT: Throughput increased by {change_percent:.1f}%",
                    current_value=current,
                    baseline_value=baseline,
                    change_percent=change_percent,
                    threshold=thresholds["improvement"]["throughput_increase_percent"]
                ))

        return alerts

    def _check_latency_regression(self, current_metrics: Dict[str, float],
                                baseline_metrics: Dict[str, float],
                                thresholds: Dict[str, Dict[str, float]]) -> List[PerformanceAlert]:
        """Check for latency-related regressions"""
        alerts = []

        latency_metrics = [
            "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
            "comparison_latency_ms", "criterion_average_time_ms"
        ]

        for metric_key in latency_metrics:
            if metric_key not in current_metrics or metric_key not in baseline_metrics:
                continue

            current = current_metrics[metric_key]
            baseline = baseline_metrics[metric_key]
            change_percent = self._calculate_change_percent(current, baseline)

            # For latency, increase is bad, decrease is good
            if change_percent > thresholds["critical"]["latency_increase_percent"]:
                alerts.append(PerformanceAlert(
                    level=AlertLevel.CRITICAL,
                    metric=metric_key,
                    message=f"CRITICAL: Latency increased by {change_percent:.1f}%",
                    current_value=current,
                    baseline_value=baseline,
                    change_percent=change_percent,
                    threshold=thresholds["critical"]["latency_increase_percent"]
                ))
            elif change_percent > thresholds["warning"]["latency_increase_percent"]:
                alerts.append(PerformanceAlert(
                    level=AlertLevel.WARNING,
                    metric=metric_key,
                    message=f"WARNING: Latency increased by {change_percent:.1f}%",
                    current_value=current,
                    baseline_value=baseline,
                    change_percent=change_percent,
                    threshold=thresholds["warning"]["latency_increase_percent"]
                ))
            elif change_percent < -thresholds["improvement"]["latency_decrease_percent"]:
                alerts.append(PerformanceAlert(
                    level=AlertLevel.IMPROVEMENT,
                    metric=metric_key,
                    message=f"IMPROVEMENT: Latency decreased by {abs(change_percent):.1f}%",
                    current_value=current,
                    baseline_value=baseline,
                    change_percent=change_percent,
                    threshold=thresholds["improvement"]["latency_decrease_percent"]
                ))

        return alerts

    def _check_memory_regression(self, current_metrics: Dict[str, float],
                               baseline_metrics: Dict[str, float],
                               thresholds: Dict[str, Dict[str, float]]) -> List[PerformanceAlert]:
        """Check for memory-related regressions"""
        alerts = []

        if "memory_usage_mb" not in current_metrics or "memory_usage_mb" not in baseline_metrics:
            return alerts

        current = current_metrics["memory_usage_mb"]
        baseline = baseline_metrics["memory_usage_mb"]
        change_percent = self._calculate_change_percent(current, baseline)

        # Memory increase is bad, decrease is good
        if change_percent > thresholds["critical"]["memory_increase_percent"]:
            alerts.append(PerformanceAlert(
                level=AlertLevel.CRITICAL,
                metric="memory_usage_mb",
                message=f"CRITICAL: Memory usage increased by {change_percent:.1f}%",
                current_value=current,
                baseline_value=baseline,
                change_percent=change_percent,
                threshold=thresholds["critical"]["memory_increase_percent"]
            ))
        elif change_percent > thresholds["warning"]["memory_increase_percent"]:
            alerts.append(PerformanceAlert(
                level=AlertLevel.WARNING,
                metric="memory_usage_mb",
                message=f"WARNING: Memory usage increased by {change_percent:.1f}%",
                current_value=current,
                baseline_value=baseline,
                change_percent=change_percent,
                threshold=thresholds["warning"]["memory_increase_percent"]
            ))
        elif change_percent < -thresholds["improvement"]["memory_decrease_percent"]:
            alerts.append(PerformanceAlert(
                level=AlertLevel.IMPROVEMENT,
                metric="memory_usage_mb",
                message=f"IMPROVEMENT: Memory usage decreased by {abs(change_percent):.1f}%",
                current_value=current,
                baseline_value=baseline,
                change_percent=change_percent,
                threshold=thresholds["improvement"]["memory_decrease_percent"]
            ))

        return alerts

    def _check_accuracy_regression(self, current_metrics: Dict[str, float],
                                 baseline_metrics: Dict[str, float],
                                 thresholds: Dict[str, Dict[str, float]]) -> List[PerformanceAlert]:
        """Check for accuracy-related regressions"""
        alerts = []

        if "accuracy_score" not in current_metrics or "accuracy_score" not in baseline_metrics:
            return alerts

        current = current_metrics["accuracy_score"]
        baseline = baseline_metrics["accuracy_score"]
        change = current - baseline

        # Accuracy decrease is bad, increase is good
        if change < -thresholds["critical"]["accuracy_decrease"]:
            alerts.append(PerformanceAlert(
                level=AlertLevel.CRITICAL,
                metric="accuracy_score",
                message=f"CRITICAL: Accuracy decreased by {abs(change):.4f}",
                current_value=current,
                baseline_value=baseline,
                change_percent=(change / baseline) * 100 if baseline > 0 else 0,
                threshold=thresholds["critical"]["accuracy_decrease"]
            ))
        elif change < -thresholds["warning"]["accuracy_decrease"]:
            alerts.append(PerformanceAlert(
                level=AlertLevel.WARNING,
                metric="accuracy_score",
                message=f"WARNING: Accuracy decreased by {abs(change):.4f}",
                current_value=current,
                baseline_value=baseline,
                change_percent=(change / baseline) * 100 if baseline > 0 else 0,
                threshold=thresholds["warning"]["accuracy_decrease"]
            ))
        elif change > thresholds["improvement"]["accuracy_increase"]:
            alerts.append(PerformanceAlert(
                level=AlertLevel.IMPROVEMENT,
                metric="accuracy_score",
                message=f"IMPROVEMENT: Accuracy increased by {change:.4f}",
                current_value=current,
                baseline_value=baseline,
                change_percent=(change / baseline) * 100 if baseline > 0 else 0,
                threshold=thresholds["improvement"]["accuracy_increase"]
            ))

        return alerts

    def analyze_results(self, results_file: str, platform: str = "linux-x86_64") -> RegressionResult:
        """Analyze benchmark results for performance regressions"""

        # Load current results
        try:
            with open(results_file, 'r') as f:
                current_results = json.load(f)
        except Exception as e:
            return RegressionResult(
                status="error",
                alerts=[PerformanceAlert(
                    level=AlertLevel.CRITICAL,
                    metric="file_load",
                    message=f"ERROR: Could not load results file: {e}",
                    current_value=0,
                    baseline_value=0,
                    change_percent=0,
                    threshold=0
                )],
                summary={"error": str(e)}
            )

        # Extract metrics from current results
        current_metrics = self._extract_metrics(current_results)

        # Get baseline metrics for platform
        baseline_data = self.baselines.get("baselines", {}).get(platform, {})
        baseline_metrics = self._extract_metrics(baseline_data)

        if not baseline_metrics:
            return RegressionResult(
                status="no_baseline",
                alerts=[PerformanceAlert(
                    level=AlertLevel.WARNING,
                    metric="baseline",
                    message=f"WARNING: No baseline data available for platform '{platform}'",
                    current_value=0,
                    baseline_value=0,
                    change_percent=0,
                    threshold=0
                )],
                summary={
                    "platform": platform,
                    "current_metrics": current_metrics,
                    "baseline_metrics": {}
                }
            )

        # Get thresholds
        thresholds = self._get_thresholds()

        # Run all regression checks
        all_alerts = []
        all_alerts.extend(self._check_throughput_regression(current_metrics, baseline_metrics, thresholds))
        all_alerts.extend(self._check_latency_regression(current_metrics, baseline_metrics, thresholds))
        all_alerts.extend(self._check_memory_regression(current_metrics, baseline_metrics, thresholds))
        all_alerts.extend(self._check_accuracy_regression(current_metrics, baseline_metrics, thresholds))

        # Determine overall status
        if any(alert.level == AlertLevel.CRITICAL for alert in all_alerts):
            status = "critical"
        elif any(alert.level == AlertLevel.WARNING for alert in all_alerts):
            status = "warning"
        elif any(alert.level == AlertLevel.IMPROVEMENT for alert in all_alerts):
            status = "improved"
        else:
            status = "stable"

        # Create summary
        summary = {
            "platform": platform,
            "current_metrics": current_metrics,
            "baseline_metrics": baseline_metrics,
            "total_alerts": len(all_alerts),
            "critical_alerts": len([a for a in all_alerts if a.level == AlertLevel.CRITICAL]),
            "warning_alerts": len([a for a in all_alerts if a.level == AlertLevel.WARNING]),
            "improvement_alerts": len([a for a in all_alerts if a.level == AlertLevel.IMPROVEMENT])
        }

        return RegressionResult(
            status=status,
            alerts=all_alerts,
            summary=summary
        )


def format_alert_as_dict(alert: PerformanceAlert) -> Dict[str, Any]:
    """Convert PerformanceAlert to dictionary"""
    return {
        "level": alert.level.value,
        "metric": alert.metric,
        "message": alert.message,
        "current_value": alert.current_value,
        "baseline_value": alert.baseline_value,
        "change_percent": alert.change_percent,
        "threshold": alert.threshold
    }


def main():
    parser = argparse.ArgumentParser(description="Detect performance regressions in BitNet.rs benchmarks")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--platform", "-p", default="linux-x86_64",
                       help="Platform to compare against (default: linux-x86_64)")
    parser.add_argument("--baselines", "-b", default="crossval/baselines.json",
                       help="Path to baselines file (default: crossval/baselines.json)")
    parser.add_argument("--output", "-o", help="Output file for regression analysis results")
    parser.add_argument("--format", choices=["json", "human"], default="human",
                       help="Output format (default: human)")
    parser.add_argument("--fail-on-regression", action="store_true",
                       help="Exit with code 1 if critical regressions detected")

    args = parser.parse_args()

    # Create regressor
    regressor = PerformanceRegressor(args.baselines)

    # Analyze results
    result = regressor.analyze_results(args.results_file, args.platform)

    # Generate output
    if args.format == "json":
        output_data = {
            "status": result.status,
            "alerts": [format_alert_as_dict(alert) for alert in result.alerts],
            "summary": result.summary
        }

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
        else:
            print(json.dumps(output_data, indent=2))

    else:  # human format
        print("🔍 BitNet.rs Performance Regression Analysis")
        print("=" * 50)
        print(f"Platform: {args.platform}")
        print(f"Status: {result.status.upper()}")
        print(f"Total alerts: {len(result.alerts)}")
        print()

        if result.alerts:
            # Group alerts by level
            critical_alerts = [a for a in result.alerts if a.level == AlertLevel.CRITICAL]
            warning_alerts = [a for a in result.alerts if a.level == AlertLevel.WARNING]
            improvement_alerts = [a for a in result.alerts if a.level == AlertLevel.IMPROVEMENT]

            if critical_alerts:
                print("🚨 CRITICAL ISSUES:")
                for alert in critical_alerts:
                    print(f"  {alert.message}")
                    print(f"    Current: {alert.current_value:.3f}, Baseline: {alert.baseline_value:.3f}")
                print()

            if warning_alerts:
                print("⚠️  WARNINGS:")
                for alert in warning_alerts:
                    print(f"  {alert.message}")
                    print(f"    Current: {alert.current_value:.3f}, Baseline: {alert.baseline_value:.3f}")
                print()

            if improvement_alerts:
                print("✅ IMPROVEMENTS:")
                for alert in improvement_alerts:
                    print(f"  {alert.message}")
                    print(f"    Current: {alert.current_value:.3f}, Baseline: {alert.baseline_value:.3f}")
                print()

        else:
            print("✅ No performance issues detected")

        if args.output:
            output_data = {
                "status": result.status,
                "alerts": [format_alert_as_dict(alert) for alert in result.alerts],
                "summary": result.summary
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")

    # Exit with appropriate code
    if args.fail_on_regression and result.status == "critical":
        sys.exit(1)
    elif result.status == "error":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()