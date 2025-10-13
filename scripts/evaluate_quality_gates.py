#!/usr/bin/env python3
"""
Evaluate release quality gates for release validation.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple


class QualityGate:
    """Represents a single quality gate with pass/fail criteria."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.message = ""
        self.details = {}

    def evaluate(self, data: Any) -> bool:
        """Evaluate the quality gate. Override in subclasses."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'passed': self.passed,
            'message': self.message,
            'details': self.details
        }


class CoverageGate(QualityGate):
    """Quality gate for code coverage threshold."""

    def __init__(self, threshold: float = 85.0):
        super().__init__(
            "Code Coverage",
            f"Code coverage must be >= {threshold}%"
        )
        self.threshold = threshold

    def evaluate(self, coverage_data: Dict[str, Any]) -> bool:
        try:
            if 'files' in coverage_data:
                # Tarpaulin format
                total_lines = 0
                covered_lines = 0

                for file_data in coverage_data['files'].values():
                    if 'coverage' in file_data:
                        for line_data in file_data['coverage']:
                            if line_data is not None:
                                total_lines += 1
                                if line_data > 0:
                                    covered_lines += 1

                if total_lines == 0:
                    self.passed = False
                    self.message = "No coverage data found"
                    return False

                coverage_percentage = (covered_lines / total_lines) * 100
            else:
                coverage_percentage = coverage_data.get('coverage', 0.0)

            self.details = {
                'coverage_percentage': coverage_percentage,
                'threshold': self.threshold
            }

            if coverage_percentage >= self.threshold:
                self.passed = True
                self.message = f"Coverage {coverage_percentage:.2f}% meets threshold {self.threshold}%"
            else:
                self.passed = False
                self.message = f"Coverage {coverage_percentage:.2f}% below threshold {self.threshold}%"

            return self.passed

        except Exception as e:
            self.passed = False
            self.message = f"Error evaluating coverage: {e}"
            return False


class PerformanceGate(QualityGate):
    """Quality gate for performance regression."""

    def __init__(self):
        super().__init__(
            "Performance Regression",
            "No significant performance regressions allowed"
        )

    def evaluate(self, performance_data: Dict[str, Any]) -> bool:
        try:
            self.passed = performance_data.get('passed', False)
            regressions = performance_data.get('regressions', [])

            self.details = {
                'regressions_count': len(regressions),
                'regressions': regressions[:5]  # Limit to first 5 for brevity
            }

            if self.passed:
                self.message = "No performance regressions detected"
            else:
                self.message = f"{len(regressions)} performance regression(s) detected"

            return self.passed

        except Exception as e:
            self.passed = False
            self.message = f"Error evaluating performance: {e}"
            return False


class SecurityGate(QualityGate):
    """Quality gate for security vulnerabilities."""

    def __init__(self):
        super().__init__(
            "Security Vulnerabilities",
            "No high or critical security vulnerabilities allowed"
        )

    def evaluate(self, security_data: Dict[str, Any]) -> bool:
        try:
            vulnerabilities = security_data.get('vulnerabilities', [])

            high_critical_vulns = [
                vuln for vuln in vulnerabilities
                if vuln.get('advisory', {}).get('severity', '').lower() in ['high', 'critical']
            ]

            self.details = {
                'total_vulnerabilities': len(vulnerabilities),
                'high_critical_count': len(high_critical_vulns),
                'high_critical_vulns': high_critical_vulns[:3]  # Limit for brevity
            }

            if len(high_critical_vulns) == 0:
                self.passed = True
                self.message = f"No high/critical vulnerabilities found ({len(vulnerabilities)} total)"
            else:
                self.passed = False
                self.message = f"{len(high_critical_vulns)} high/critical vulnerabilities found"

            return self.passed

        except Exception as e:
            self.passed = False
            self.message = f"Error evaluating security: {e}"
            return False


class CrossPlatformGate(QualityGate):
    """Quality gate for cross-platform build success."""

    def __init__(self):
        super().__init__(
            "Cross-Platform Builds",
            "All target platforms must build successfully"
        )

    def evaluate(self, build_results: List[Path]) -> bool:
        try:
            # Count successful builds based on artifact directories
            successful_builds = []
            failed_builds = []

            for result_dir in build_results:
                if result_dir.is_dir() and any(result_dir.iterdir()):
                    successful_builds.append(result_dir.name)
                else:
                    failed_builds.append(result_dir.name)

            self.details = {
                'successful_builds': successful_builds,
                'failed_builds': failed_builds,
                'total_targets': len(build_results)
            }

            if len(failed_builds) == 0:
                self.passed = True
                self.message = f"All {len(successful_builds)} platform builds successful"
            else:
                self.passed = False
                self.message = f"{len(failed_builds)} platform build(s) failed"

            return self.passed

        except Exception as e:
            self.passed = False
            self.message = f"Error evaluating cross-platform builds: {e}"
            return False


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            return {}
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return {}


def find_build_results(pattern: str) -> List[Path]:
    """Find build result directories matching pattern."""
    current_dir = Path.cwd()
    return list(current_dir.glob(pattern))


def evaluate_all_gates(coverage_file: Path, performance_file: Path,
                      security_file: Path, cross_platform_pattern: str) -> Tuple[bool, List[QualityGate]]:
    """Evaluate all quality gates."""
    gates = []

    # Coverage gate
    coverage_gate = CoverageGate(threshold=85.0)
    coverage_data = load_json_file(coverage_file)
    coverage_gate.evaluate(coverage_data)
    gates.append(coverage_gate)

    # Performance gate
    performance_gate = PerformanceGate()
    performance_data = load_json_file(performance_file)
    performance_gate.evaluate(performance_data)
    gates.append(performance_gate)

    # Security gate
    security_gate = SecurityGate()
    security_data = load_json_file(security_file)
    security_gate.evaluate(security_data)
    gates.append(security_gate)

    # Cross-platform gate
    cross_platform_gate = CrossPlatformGate()
    build_results = find_build_results(cross_platform_pattern)
    cross_platform_gate.evaluate(build_results)
    gates.append(cross_platform_gate)

    # Overall pass/fail
    all_passed = all(gate.passed for gate in gates)

    return all_passed, gates


def generate_summary_report(gates: List[QualityGate], overall_passed: bool) -> str:
    """Generate a summary report of quality gate results."""
    report = []
    report.append("# Release Quality Gates Summary")
    report.append("")

    if overall_passed:
        report.append("## ✅ Overall Status: PASSED")
        report.append("All quality gates have been satisfied.")
    else:
        report.append("## ❌ Overall Status: FAILED")
        report.append("One or more quality gates have failed.")

    report.append("")
    report.append("## Quality Gate Results")
    report.append("")

    for gate in gates:
        status_icon = "✅" if gate.passed else "❌"
        report.append(f"### {status_icon} {gate.name}")
        report.append(f"**Description:** {gate.description}")
        report.append(f"**Status:** {'PASSED' if gate.passed else 'FAILED'}")
        report.append(f"**Message:** {gate.message}")

        if gate.details:
            report.append("**Details:**")
            for key, value in gate.details.items():
                if isinstance(value, list) and len(value) > 0:
                    report.append(f"- {key}: {len(value)} items")
                else:
                    report.append(f"- {key}: {value}")

        report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Evaluate release quality gates')
    parser.add_argument('--coverage-report', type=Path,
                       help='Coverage report JSON file')
    parser.add_argument('--performance-report', type=Path,
                       help='Performance comparison JSON file')
    parser.add_argument('--security-report', type=Path,
                       help='Security audit JSON file')
    parser.add_argument('--cross-platform-results', type=str, default='build-*/',
                       help='Pattern for cross-platform build result directories')
    parser.add_argument('--output', type=Path,
                       help='Output summary report file')

    args = parser.parse_args()

    # Set defaults if not provided
    coverage_file = args.coverage_report or Path('coverage-report/tarpaulin-report.json')
    performance_file = args.performance_report or Path('performance-results/comparison.json')
    security_file = args.security_report or Path('security-results/security-audit.json')

    print("Evaluating release quality gates...")

    # Evaluate all gates
    overall_passed, gates = evaluate_all_gates(
        coverage_file, performance_file, security_file, args.cross_platform_results
    )

    # Generate summary report
    summary_report = generate_summary_report(gates, overall_passed)
    print(summary_report)

    # Save detailed results as JSON
    results = {
        'overall_passed': overall_passed,
        'gates': [gate.to_dict() for gate in gates],
        'summary_report': summary_report
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    # Print final status
    if overall_passed:
        print("\n✅ All quality gates passed - release candidate approved!")
        return 0
    else:
        failed_gates = [gate.name for gate in gates if not gate.passed]
        print(f"\n❌ Quality gates failed: {', '.join(failed_gates)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
