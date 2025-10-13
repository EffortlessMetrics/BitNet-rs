#!/usr/bin/env python3
"""
Test runner for BitNet Python baseline validation.
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional

def run_command(cmd: List[str], cwd: Optional[Path] = None, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture_output,
        text=True
    )

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        if result.stderr:
            print(f"Error output: {result.stderr}")

    return result

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    required_packages = [
        "pytest",
        "torch",
        "numpy",
        "hypothesis",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r tests/python_baseline/requirements.txt")
        return False

    return True

def run_unit_tests(args: argparse.Namespace) -> bool:
    """Run unit tests."""
    print("\n" + "="*50)
    print("RUNNING UNIT TESTS")
    print("="*50)

    cmd = [
        "python", "-m", "pytest",
        "tests/python_baseline/test_model_loading.py",
        "tests/python_baseline/test_quantization.py",
        "tests/python_baseline/test_inference.py",
        "-v",
        "-m", "not slow and not gpu",
    ]

    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])

    if args.coverage:
        cmd.extend(["--cov=gpu", "--cov=utils", "--cov-report=term-missing"])

    result = run_command(cmd)
    return result.returncode == 0

def run_property_tests(args: argparse.Namespace) -> bool:
    """Run property-based tests."""
    print("\n" + "="*50)
    print("RUNNING PROPERTY-BASED TESTS")
    print("="*50)

    cmd = [
        "python", "-m", "pytest",
        "tests/python_baseline/test_property_based.py",
        "-v",
        "-m", "property",
    ]

    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])

    result = run_command(cmd)
    return result.returncode == 0

def run_performance_tests(args: argparse.Namespace) -> bool:
    """Run performance benchmark tests."""
    print("\n" + "="*50)
    print("RUNNING PERFORMANCE TESTS")
    print("="*50)

    cmd = [
        "python", "-m", "pytest",
        "tests/python_baseline/test_performance_benchmarks.py",
        "-v",
        "-m", "slow",
        "--tb=short",
    ]

    if args.gpu:
        cmd.extend(["-m", "slow or gpu"])
    else:
        cmd.extend(["-m", "slow and not gpu"])

    result = run_command(cmd)
    return result.returncode == 0

def run_fixture_tests(args: argparse.Namespace) -> bool:
    """Run fixture validation tests."""
    print("\n" + "="*50)
    print("RUNNING FIXTURE VALIDATION TESTS")
    print("="*50)

    cmd = [
        "python", "-m", "pytest",
        "tests/python_baseline/test_fixtures.py",
        "-v",
    ]

    result = run_command(cmd)
    return result.returncode == 0

def run_gpu_tests(args: argparse.Namespace) -> bool:
    """Run GPU-specific tests."""
    print("\n" + "="*50)
    print("RUNNING GPU TESTS")
    print("="*50)

    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, skipping GPU tests")
            return True
    except ImportError:
        print("PyTorch not available, skipping GPU tests")
        return True

    cmd = [
        "python", "-m", "pytest",
        "tests/python_baseline/",
        "-v",
        "-m", "gpu",
    ]

    result = run_command(cmd)
    return result.returncode == 0

def generate_test_report(results: Dict[str, bool], output_file: Path):
    """Generate a test report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "summary": {
            "total_suites": len(results),
            "passed_suites": sum(1 for passed in results.values() if passed),
            "failed_suites": sum(1 for passed in results.values() if not passed),
        }
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nTest report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run BitNet Python baseline tests")

    # Test selection
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--property", action="store_true", help="Run property-based tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--fixtures", action="store_true", help="Run fixture validation tests")
    parser.add_argument("--gpu", action="store_true", help="Run GPU tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    # Test configuration
    parser.add_argument("--parallel", type=int, help="Number of parallel workers")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--report", type=str, help="Output file for test report")

    # Environment
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")

    args = parser.parse_args()

    # Set default report file
    if not args.report:
        args.report = "tests/test_report.json"

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    if args.check_deps:
        print("All dependencies are available!")
        sys.exit(0)

    # Determine which tests to run
    run_tests = []

    if args.all:
        run_tests = ["unit", "property", "fixtures", "performance"]
        if args.gpu:
            run_tests.append("gpu")
    else:
        if args.unit:
            run_tests.append("unit")
        if args.property:
            run_tests.append("property")
        if args.performance:
            run_tests.append("performance")
        if args.fixtures:
            run_tests.append("fixtures")
        if args.gpu:
            run_tests.append("gpu")

    if not run_tests:
        print("No tests specified. Use --all or specify individual test suites.")
        parser.print_help()
        sys.exit(1)

    # Run tests
    results = {}

    if "unit" in run_tests:
        results["unit_tests"] = run_unit_tests(args)

    if "property" in run_tests:
        results["property_tests"] = run_property_tests(args)

    if "fixtures" in run_tests:
        results["fixture_tests"] = run_fixture_tests(args)

    if "performance" in run_tests:
        results["performance_tests"] = run_performance_tests(args)

    if "gpu" in run_tests:
        results["gpu_tests"] = run_gpu_tests(args)

    # Generate report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_test_report(results, report_path)

    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)

    all_passed = True
    for suite_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{suite_name}: {status}")
        if not passed:
            all_passed = False

    print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
