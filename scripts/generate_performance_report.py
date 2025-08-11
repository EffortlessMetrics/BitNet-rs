#!/usr/bin/env python3
"""
Generate HTML performance report for release validation.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Validation Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .status-passed { color: #28a745; }
        .status-failed { color: #dc3545; }
        .status-improvement { color: #28a745; }
        .status-regression { color: #dc3545; }
        .status-maintained { color: #6c757d; }
        .status-new { color: #17a2b8; }
        .status-removed { color: #ffc107; }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            font-size: 2em;
        }
        .benchmark-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }
        .benchmark-table th,
        .benchmark-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .benchmark-table th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .benchmark-table tr:hover {
            background-color: #f5f5f5;
        }
        .change-positive { color: #28a745; font-weight: bold; }
        .change-negative { color: #dc3545; font-weight: bold; }
        .change-neutral { color: #6c757d; }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .alert {
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .alert-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .alert-danger {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Validation Report</h1>
        <p>Generated on {timestamp}</p>
        <p><strong>Status:</strong> <span class="status-{status_class}">{status_text}</span></p>
    </div>

    {alert_section}

    <div class="section">
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>{total_benchmarks}</h3>
                <p>Total Benchmarks</p>
            </div>
            <div class="summary-card">
                <h3 class="status-regression">{regressions}</h3>
                <p>Regressions</p>
            </div>
            <div class="summary-card">
                <h3 class="status-improvement">{improvements}</h3>
                <p>Improvements</p>
            </div>
            <div class="summary-card">
                <h3 class="status-maintained">{maintained}</h3>
                <p>Maintained</p>
            </div>
        </div>
    </div>

    {regressions_section}

    {improvements_section}

    <div class="section">
        <h2>All Benchmark Results</h2>
        <table class="benchmark-table">
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>Status</th>
                    <th>Baseline</th>
                    <th>Current</th>
                    <th>Change</th>
                </tr>
            </thead>
            <tbody>
                {benchmark_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds:.3f}s"


def format_change(change_percent: float) -> tuple[str, str]:
    """Format change percentage with appropriate class."""
    if change_percent > 0:
        return f"+{change_percent:.2f}%", "change-negative"
    elif change_percent < 0:
        return f"{change_percent:.2f}%", "change-positive"
    else:
        return "0.00%", "change-neutral"


def generate_html_report(comparison_data: dict) -> str:
    """Generate HTML performance report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Determine overall status
    status_class = "passed" if comparison_data['passed'] else "failed"
    status_text = "PASSED" if comparison_data['passed'] else "FAILED"
    
    # Generate alert section
    if comparison_data['passed']:
        alert_section = '''
        <div class="alert alert-success">
            <strong>✅ Performance validation passed!</strong> No significant performance regressions detected.
        </div>
        '''
    else:
        alert_section = f'''
        <div class="alert alert-danger">
            <strong>❌ Performance validation failed!</strong> 
            {len(comparison_data['regressions'])} performance regression(s) detected.
        </div>
        '''
    
    # Count different status types
    status_counts = {'regression': 0, 'improvement': 0, 'maintained': 0, 'new': 0, 'removed': 0}
    for benchmark_data in comparison_data['summary'].values():
        status_counts[benchmark_data['status']] += 1
    
    # Generate regressions section
    regressions_section = ""
    if comparison_data['regressions']:
        regressions_section = '''
        <div class="section">
            <h2>❌ Performance Regressions</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Baseline</th>
                        <th>Current</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
        '''
        for regression in comparison_data['regressions']:
            change_text, change_class = format_change(regression['change_percent'])
            regressions_section += f'''
                    <tr>
                        <td><strong>{regression['benchmark']}</strong></td>
                        <td>{format_duration(regression['baseline'])}</td>
                        <td>{format_duration(regression['current'])}</td>
                        <td class="{change_class}">{change_text}</td>
                    </tr>
            '''
        regressions_section += '''
                </tbody>
            </table>
        </div>
        '''
    
    # Generate improvements section
    improvements_section = ""
    if comparison_data['improvements']:
        improvements_section = '''
        <div class="section">
            <h2>✅ Performance Improvements</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Baseline</th>
                        <th>Current</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
        '''
        for improvement in comparison_data['improvements']:
            change_text, change_class = format_change(improvement['change_percent'])
            improvements_section += f'''
                    <tr>
                        <td><strong>{improvement['benchmark']}</strong></td>
                        <td>{format_duration(improvement['baseline'])}</td>
                        <td>{format_duration(improvement['current'])}</td>
                        <td class="{change_class}">{change_text}</td>
                    </tr>
            '''
        improvements_section += '''
                </tbody>
            </table>
        </div>
        '''
    
    # Generate benchmark rows
    benchmark_rows = ""
    for benchmark, data in comparison_data['summary'].items():
        status_icons = {
            'regression': '❌',
            'improvement': '✅',
            'maintained': '➖',
            'new': '🆕',
            'removed': '🗑️'
        }
        
        status_icon = status_icons.get(data['status'], '❓')
        status_text = f"{status_icon} {data['status'].title()}"
        
        if data.get('change_percent') is not None:
            change_text, change_class = format_change(data['change_percent'])
            baseline_text = format_duration(data['baseline'])
            current_text = format_duration(data['current'])
        else:
            change_text = "N/A"
            change_class = "change-neutral"
            baseline_text = format_duration(data.get('baseline', 0)) if data.get('baseline') else "N/A"
            current_text = format_duration(data.get('current', 0)) if data.get('current') else "N/A"
        
        benchmark_rows += f'''
            <tr>
                <td><strong>{benchmark}</strong></td>
                <td class="status-{data['status']}">{status_text}</td>
                <td>{baseline_text}</td>
                <td>{current_text}</td>
                <td class="{change_class}">{change_text}</td>
            </tr>
        '''
    
    # Fill in the template
    return HTML_TEMPLATE.format(
        timestamp=timestamp,
        status_class=status_class,
        status_text=status_text,
        alert_section=alert_section,
        total_benchmarks=len(comparison_data['summary']),
        regressions=len(comparison_data['regressions']),
        improvements=len(comparison_data['improvements']),
        maintained=status_counts['maintained'],
        regressions_section=regressions_section,
        improvements_section=improvements_section,
        benchmark_rows=benchmark_rows
    )


def main():
    parser = argparse.ArgumentParser(description='Generate HTML performance report')
    parser.add_argument('comparison_file', type=Path, help='Performance comparison JSON file')
    parser.add_argument('--output', type=Path, required=True, help='Output HTML file')
    
    args = parser.parse_args()
    
    # Load comparison data
    try:
        with open(args.comparison_file, 'r') as f:
            comparison_data = json.load(f)
    except Exception as e:
        print(f"Error loading comparison file: {e}")
        return 1
    
    # Generate HTML report
    html_content = generate_html_report(comparison_data)
    
    # Write to output file
    try:
        with open(args.output, 'w') as f:
            f.write(html_content)
        print(f"Performance report generated: {args.output}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())