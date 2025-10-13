#!/usr/bin/env python3
"""
Generate comprehensive release validation report.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Release Validation Report - {version}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header .version {{
            font-size: 1.5em;
            opacity: 0.9;
        }}
        .status-passed {{ color: #28a745; }}
        .status-failed {{ color: #dc3545; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #007bff;
        }}
        .summary-card.passed {{
            border-left-color: #28a745;
        }}
        .summary-card.failed {{
            border-left-color: #dc3545;
        }}
        .summary-card h3 {{
            margin: 0 0 15px 0;
            font-size: 1.2em;
        }}
        .summary-card .metric {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .section {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .section-header {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
        }}
        .section-header h2 {{
            margin: 0;
            color: #495057;
        }}
        .section-content {{
            padding: 20px;
        }}
        .quality-gate {{
            display: flex;
            align-items: center;
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        .quality-gate:last-child {{
            border-bottom: none;
        }}
        .quality-gate-icon {{
            font-size: 1.5em;
            margin-right: 15px;
            min-width: 30px;
        }}
        .quality-gate-content {{
            flex: 1;
        }}
        .quality-gate-title {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .quality-gate-description {{
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        .quality-gate-message {{
            font-size: 0.9em;
        }}
        .alert {{
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .alert-success {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }}
        .alert-danger {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }}
        .details-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .details-table th,
        .details-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .details-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #6c757d;
            font-size: 0.9em;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .badge-success {{
            background-color: #28a745;
            color: white;
        }}
        .badge-danger {{
            background-color: #dc3545;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Release Validation Report</h1>
        <div class="version">{version}</div>
        <p>Generated on {timestamp}</p>
        <div style="margin-top: 20px;">
            <span class="badge {overall_status_class}">{overall_status_text}</span>
        </div>
    </div>

    {alert_section}

    <div class="summary-grid">
        {summary_cards}
    </div>

    <div class="section">
        <div class="section-header">
            <h2>Quality Gates</h2>
        </div>
        <div class="section-content">
            {quality_gates_section}
        </div>
    </div>

    {coverage_section}

    {performance_section}

    {security_section}

    <div class="footer">
        <p>This report was generated automatically as part of the release validation pipeline.</p>
        <p>Baseline version: {baseline_version}</p>
    </div>
</body>
</html>
"""


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    except Exception:
        return {}


def generate_summary_cards(coverage_data: Dict, performance_data: Dict,
                          security_data: Dict) -> str:
    """Generate summary cards HTML."""
    cards = []

    # Coverage card
    coverage_percentage = 0.0
    if 'files' in coverage_data:
        total_lines = 0
        covered_lines = 0
        for file_data in coverage_data['files'].values():
            if 'coverage' in file_data:
                for line_data in file_data['coverage']:
                    if line_data is not None:
                        total_lines += 1
                        if line_data > 0:
                            covered_lines += 1
        if total_lines > 0:
            coverage_percentage = (covered_lines / total_lines) * 100

    coverage_status = "passed" if coverage_percentage >= 85.0 else "failed"
    cards.append(f'''
        <div class="summary-card {coverage_status}">
            <h3>Code Coverage</h3>
            <div class="metric status-{coverage_status}">{coverage_percentage:.1f}%</div>
            <p>Target: ≥85%</p>
        </div>
    ''')

    # Performance card
    regressions = performance_data.get('regressions', [])
    performance_status = "passed" if len(regressions) == 0 else "failed"
    cards.append(f'''
        <div class="summary-card {performance_status}">
            <h3>Performance</h3>
            <div class="metric status-{performance_status}">{len(regressions)}</div>
            <p>Regressions detected</p>
        </div>
    ''')

    # Security card
    vulnerabilities = security_data.get('vulnerabilities', [])
    high_critical = [v for v in vulnerabilities
                    if v.get('advisory', {}).get('severity', '').lower() in ['high', 'critical']]
    security_status = "passed" if len(high_critical) == 0 else "failed"
    cards.append(f'''
        <div class="summary-card {security_status}">
            <h3>Security</h3>
            <div class="metric status-{security_status}">{len(high_critical)}</div>
            <p>High/Critical vulnerabilities</p>
        </div>
    ''')

    # Overall status card
    overall_passed = coverage_status == "passed" and performance_status == "passed" and security_status == "passed"
    overall_status = "passed" if overall_passed else "failed"
    cards.append(f'''
        <div class="summary-card {overall_status}">
            <h3>Overall Status</h3>
            <div class="metric status-{overall_status}">{'PASS' if overall_passed else 'FAIL'}</div>
            <p>Release validation</p>
        </div>
    ''')

    return ''.join(cards)


def generate_quality_gates_section(coverage_data: Dict, performance_data: Dict,
                                 security_data: Dict) -> str:
    """Generate quality gates section HTML."""
    gates = []

    # Coverage gate
    coverage_percentage = 0.0
    if 'files' in coverage_data:
        total_lines = 0
        covered_lines = 0
        for file_data in coverage_data['files'].values():
            if 'coverage' in file_data:
                for line_data in file_data['coverage']:
                    if line_data is not None:
                        total_lines += 1
                        if line_data > 0:
                            covered_lines += 1
        if total_lines > 0:
            coverage_percentage = (covered_lines / total_lines) * 100

    coverage_passed = coverage_percentage >= 85.0
    coverage_icon = "✅" if coverage_passed else "❌"
    gates.append(f'''
        <div class="quality-gate">
            <div class="quality-gate-icon">{coverage_icon}</div>
            <div class="quality-gate-content">
                <div class="quality-gate-title">Code Coverage</div>
                <div class="quality-gate-description">Code coverage must be ≥85%</div>
                <div class="quality-gate-message">Coverage: {coverage_percentage:.2f}%</div>
            </div>
        </div>
    ''')

    # Performance gate
    regressions = performance_data.get('regressions', [])
    performance_passed = len(regressions) == 0
    performance_icon = "✅" if performance_passed else "❌"
    gates.append(f'''
        <div class="quality-gate">
            <div class="quality-gate-icon">{performance_icon}</div>
            <div class="quality-gate-content">
                <div class="quality-gate-title">Performance Regression</div>
                <div class="quality-gate-description">No significant performance regressions allowed</div>
                <div class="quality-gate-message">{len(regressions)} regression(s) detected</div>
            </div>
        </div>
    ''')

    # Security gate
    vulnerabilities = security_data.get('vulnerabilities', [])
    high_critical = [v for v in vulnerabilities
                    if v.get('advisory', {}).get('severity', '').lower() in ['high', 'critical']]
    security_passed = len(high_critical) == 0
    security_icon = "✅" if security_passed else "❌"
    gates.append(f'''
        <div class="quality-gate">
            <div class="quality-gate-icon">{security_icon}</div>
            <div class="quality-gate-content">
                <div class="quality-gate-title">Security Vulnerabilities</div>
                <div class="quality-gate-description">No high or critical security vulnerabilities allowed</div>
                <div class="quality-gate-message">{len(high_critical)} high/critical vulnerabilities found</div>
            </div>
        </div>
    ''')

    return ''.join(gates)


def generate_coverage_section(coverage_data: Dict) -> str:
    """Generate coverage details section."""
    if not coverage_data:
        return ""

    return f'''
    <div class="section">
        <div class="section-header">
            <h2>Coverage Details</h2>
        </div>
        <div class="section-content">
            <p>Detailed coverage analysis will be available in the full coverage report.</p>
        </div>
    </div>
    '''


def generate_performance_section(performance_data: Dict) -> str:
    """Generate performance details section."""
    if not performance_data:
        return ""

    regressions = performance_data.get('regressions', [])
    improvements = performance_data.get('improvements', [])

    content = []

    if regressions:
        content.append("<h4>Performance Regressions</h4>")
        content.append('<table class="details-table">')
        content.append('<thead><tr><th>Benchmark</th><th>Change</th><th>Baseline</th><th>Current</th></tr></thead>')
        content.append('<tbody>')
        for reg in regressions[:10]:  # Limit to first 10
            content.append(f'''
                <tr>
                    <td>{reg['benchmark']}</td>
                    <td class="status-failed">{reg['change_percent']:+.2f}%</td>
                    <td>{reg['baseline']:.3f}s</td>
                    <td>{reg['current']:.3f}s</td>
                </tr>
            ''')
        content.append('</tbody></table>')

    if improvements:
        content.append("<h4>Performance Improvements</h4>")
        content.append('<table class="details-table">')
        content.append('<thead><tr><th>Benchmark</th><th>Change</th><th>Baseline</th><th>Current</th></tr></thead>')
        content.append('<tbody>')
        for imp in improvements[:10]:  # Limit to first 10
            content.append(f'''
                <tr>
                    <td>{imp['benchmark']}</td>
                    <td class="status-passed">{imp['change_percent']:+.2f}%</td>
                    <td>{imp['baseline']:.3f}s</td>
                    <td>{imp['current']:.3f}s</td>
                </tr>
            ''')
        content.append('</tbody></table>')

    if not content:
        content.append("<p>No performance data available.</p>")

    return f'''
    <div class="section">
        <div class="section-header">
            <h2>Performance Analysis</h2>
        </div>
        <div class="section-content">
            {''.join(content)}
        </div>
    </div>
    '''


def generate_security_section(security_data: Dict) -> str:
    """Generate security details section."""
    if not security_data:
        return ""

    vulnerabilities = security_data.get('vulnerabilities', [])

    if not vulnerabilities:
        return f'''
        <div class="section">
            <div class="section-header">
                <h2>Security Analysis</h2>
            </div>
            <div class="section-content">
                <p class="status-passed">✅ No security vulnerabilities detected.</p>
            </div>
        </div>
        '''

    content = []
    content.append('<table class="details-table">')
    content.append('<thead><tr><th>Package</th><th>Vulnerability</th><th>Severity</th><th>Description</th></tr></thead>')
    content.append('<tbody>')

    for vuln in vulnerabilities[:20]:  # Limit to first 20
        advisory = vuln.get('advisory', {})
        severity = advisory.get('severity', 'Unknown')
        severity_class = 'failed' if severity.lower() in ['high', 'critical'] else 'passed'

        content.append(f'''
            <tr>
                <td>{vuln.get('package', {}).get('name', 'Unknown')}</td>
                <td>{advisory.get('id', 'Unknown')}</td>
                <td class="status-{severity_class}">{severity}</td>
                <td>{advisory.get('title', 'No description')[:100]}...</td>
            </tr>
        ''')

    content.append('</tbody></table>')

    return f'''
    <div class="section">
        <div class="section-header">
            <h2>Security Analysis</h2>
        </div>
        <div class="section-content">
            {''.join(content)}
        </div>
    </div>
    '''


def main():
    parser = argparse.ArgumentParser(description='Generate release validation report')
    parser.add_argument('--version', required=True, help='Release candidate version')
    parser.add_argument('--baseline', required=True, help='Baseline version')
    parser.add_argument('--coverage-report', type=Path, help='Coverage report JSON file')
    parser.add_argument('--performance-report', type=Path, help='Performance report JSON file')
    parser.add_argument('--security-report', type=Path, help='Security report JSON file')
    parser.add_argument('--output', type=Path, required=True, help='Output HTML file')

    args = parser.parse_args()

    # Load data files
    coverage_data = load_json_file(args.coverage_report) if args.coverage_report else {}
    performance_data = load_json_file(args.performance_report) if args.performance_report else {}
    security_data = load_json_file(args.security_report) if args.security_report else {}

    # Determine overall status
    coverage_percentage = 0.0
    if 'files' in coverage_data:
        total_lines = 0
        covered_lines = 0
        for file_data in coverage_data['files'].values():
            if 'coverage' in file_data:
                for line_data in file_data['coverage']:
                    if line_data is not None:
                        total_lines += 1
                        if line_data > 0:
                            covered_lines += 1
        if total_lines > 0:
            coverage_percentage = (covered_lines / total_lines) * 100

    regressions = performance_data.get('regressions', [])
    vulnerabilities = security_data.get('vulnerabilities', [])
    high_critical = [v for v in vulnerabilities
                    if v.get('advisory', {}).get('severity', '').lower() in ['high', 'critical']]

    overall_passed = (coverage_percentage >= 85.0 and
                     len(regressions) == 0 and
                     len(high_critical) == 0)

    overall_status_class = "badge-success" if overall_passed else "badge-danger"
    overall_status_text = "PASSED" if overall_passed else "FAILED"

    # Generate alert section
    if overall_passed:
        alert_section = '''
        <div class="alert alert-success">
            <strong>✅ Release validation passed!</strong> All quality gates have been satisfied.
        </div>
        '''
    else:
        alert_section = '''
        <div class="alert alert-danger">
            <strong>❌ Release validation failed!</strong> One or more quality gates have failed.
        </div>
        '''

    # Generate HTML content
    html_content = HTML_TEMPLATE.format(
        version=args.version,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        overall_status_class=overall_status_class,
        overall_status_text=overall_status_text,
        alert_section=alert_section,
        summary_cards=generate_summary_cards(coverage_data, performance_data, security_data),
        quality_gates_section=generate_quality_gates_section(coverage_data, performance_data, security_data),
        coverage_section=generate_coverage_section(coverage_data),
        performance_section=generate_performance_section(performance_data),
        security_section=generate_security_section(security_data),
        baseline_version=args.baseline
    )

    # Write to output file
    try:
        with open(args.output, 'w') as f:
            f.write(html_content)
        print(f"Release validation report generated: {args.output}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
