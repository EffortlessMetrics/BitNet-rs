#!/usr/bin/env python3
"""
Enhanced cross-language validation runner for BitNet Python to Rust migration.

This script runs comprehensive validation tests including:
- Token-level comparison with configurable tolerance
- Performance regression detection
- Edge case and stress testing
- Automated test harness with detailed reporting
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# Add the framework to the path
sys.path.insert(0, str(Path(__file__).parent))

from framework import (
    CrossLanguageValidator,
    ValidationConfig,
    TestCaseGenerator,
    create_default_config,
    run_cross_validation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_custom_config(args) -> ValidationConfig:
    """Create validation configuration from command line arguments."""
    config = create_default_config()

    # Override numerical tolerances if provided
    if args.rtol:
        config.numerical_tolerance["rtol"] = args.rtol
    if args.atol:
        config.numerical_tolerance["atol"] = args.atol
    if args.token_match_ratio:
        config.numerical_tolerance["min_token_match_ratio"] = args.token_match_ratio
    if args.max_token_diff:
        config.numerical_tolerance["max_token_differences"] = args.max_token_diff

    # Override performance tolerances if provided
    if args.regression_threshold:
        config.performance_tolerance["max_regression_percent"] = args.regression_threshold

    # Override timeout if provided
    if args.timeout:
        config.timeout_seconds = args.timeout

    return config

def filter_test_cases(test_cases: List, categories: Optional[List[str]] = None,
                     patterns: Optional[List[str]] = None) -> List:
    """Filter test cases based on categories and name patterns."""
    if not categories and not patterns:
        return test_cases

    filtered = []

    for test_case in test_cases:
        # Check category filter
        if categories:
            test_category = test_case.metadata.get("category", "basic") if test_case.metadata else "basic"
            if test_category not in categories:
                continue

        # Check pattern filter
        if patterns:
            if not any(pattern in test_case.name for pattern in patterns):
                continue

        filtered.append(test_case)

    return filtered

def run_validation_suite(args) -> Dict[str, Any]:
    """Run the complete validation suite."""
    logger.info("Starting cross-language validation suite")

    # Create configuration
    config = create_custom_config(args)
    logger.info(f"Using configuration: {config}")

    # Initialize validator
    validator = CrossLanguageValidator(config)

    # Set up Rust runner if binary path provided
    if args.rust_binary:
        rust_binary_path = Path(args.rust_binary)
        if not rust_binary_path.exists():
            logger.error(f"Rust binary not found: {rust_binary_path}")
            sys.exit(1)
        validator.set_rust_runner(rust_binary_path)
    else:
        # Try to find Rust binary automatically
        validator.set_rust_runner()

    # Generate test cases
    logger.info("Generating test cases...")
    if args.test_type == "basic":
        test_cases = TestCaseGenerator.generate_model_forward_tests()
        test_cases.extend(TestCaseGenerator.generate_quantization_tests())
        test_cases.extend(TestCaseGenerator.generate_inference_tests())
    elif args.test_type == "edge":
        test_cases = TestCaseGenerator.generate_edge_case_tests()
    elif args.test_type == "stress":
        test_cases = TestCaseGenerator.generate_stress_tests()
    elif args.test_type == "all":
        test_cases = TestCaseGenerator.generate_all_tests()
    else:
        logger.error(f"Unknown test type: {args.test_type}")
        sys.exit(1)

    # Filter test cases if requested
    if args.categories or args.patterns:
        test_cases = filter_test_cases(test_cases, args.categories, args.patterns)

    logger.info(f"Running {len(test_cases)} test cases")

    # Run validation
    results = validator.validate_test_suite(test_cases)

    # Generate report
    output_path = Path(args.output) if args.output else Path("cross_validation_report.json")
    report = validator.generate_report(results, output_path)

    # Print summary
    print_summary(report)

    # Check for regressions
    if args.fail_on_regression:
        regressions = validator.performance_analyzer.detect_regressions(
            [r.performance_comparison for r in results]
        )
        if regressions:
            logger.error(f"Performance regressions detected: {len(regressions)}")
            for regression in regressions:
                logger.error(f"  {regression['test_name']}: {regression['regression_percent']:.2f}% regression")
            sys.exit(1)

    return report

def print_summary(report: Dict[str, Any]):
    """Print a detailed summary of validation results."""
    summary = report["summary"]
    perf_summary = report["performance_summary"]

    print("\n" + "="*60)
    print("CROSS-LANGUAGE VALIDATION SUMMARY")
    print("="*60)

    print(f"Total tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed_tests']}")
    print(f"❌ Failed: {summary['failed_tests']}")

    if summary['failed_tests'] > 0:
        print(f"  - Python failures: {summary['python_failures']}")
        print(f"  - Rust failures: {summary['rust_failures']}")
        print(f"  - Output mismatches: {summary['output_mismatches']}")

    print("\nPerformance Analysis:")
    if perf_summary["average_speedup"] > 0:
        print(f"  Average speedup: {perf_summary['average_speedup']:.2f}x")
        print(f"  Max speedup: {perf_summary['max_speedup']:.2f}x")
        print(f"  Min speedup: {perf_summary['min_speedup']:.2f}x")
    else:
        print("  No performance data available")

    # Print failed tests details
    failed_tests = [r for r in report["detailed_results"] if not r["overall_success"]]
    if failed_tests:
        print(f"\nFailed Tests ({len(failed_tests)}):")
        for test in failed_tests[:5]:  # Show first 5 failures
            print(f"  - {test['test_name']}")
            if test['error_messages']:
                print(f"    Error: {test['error_messages'][0]}")

        if len(failed_tests) > 5:
            print(f"  ... and {len(failed_tests) - 5} more")

    print("\n" + "="*60)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run cross-language validation for BitNet Python to Rust migration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic validation tests
  python run_validation.py --test-type basic

  # Run all tests with custom Rust binary
  python run_validation.py --test-type all --rust-binary /path/to/bitnet-cli

  # Run only edge case tests with strict tolerances
  python run_validation.py --test-type edge --rtol 1e-5 --atol 1e-6

  # Run stress tests and fail on any performance regression
  python run_validation.py --test-type stress --fail-on-regression

  # Run specific test categories
  python run_validation.py --categories edge_case stress_test
        """
    )

    parser.add_argument(
        "--test-type",
        choices=["basic", "edge", "stress", "all"],
        default="basic",
        help="Type of tests to run"
    )

    parser.add_argument(
        "--rust-binary",
        type=str,
        help="Path to Rust BitNet binary"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output path for validation report (default: cross_validation_report.json)"
    )

    # Numerical tolerance options
    parser.add_argument(
        "--rtol",
        type=float,
        help="Relative tolerance for numerical comparisons"
    )

    parser.add_argument(
        "--atol",
        type=float,
        help="Absolute tolerance for numerical comparisons"
    )

    parser.add_argument(
        "--token-match-ratio",
        type=float,
        help="Minimum token match ratio for token sequence comparison"
    )

    parser.add_argument(
        "--max-token-diff",
        type=int,
        help="Maximum number of token differences allowed"
    )

    # Performance options
    parser.add_argument(
        "--regression-threshold",
        type=float,
        help="Performance regression threshold percentage"
    )

    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with error code if performance regressions are detected"
    )

    # Filtering options
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Filter tests by categories (e.g., edge_case, stress_test)"
    )

    parser.add_argument(
        "--patterns",
        nargs="+",
        help="Filter tests by name patterns"
    )

    # General options
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for individual tests in seconds"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        report = run_validation_suite(args)

        # Exit with appropriate code
        if report["summary"]["failed_tests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
