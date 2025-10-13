#!/usr/bin/env python3
"""
Analyze unsafe code usage in the codebase for release validation.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any
import subprocess


def find_rust_files(root_dir: Path) -> List[Path]:
    """Find all Rust source files in the project."""
    rust_files = []

    # Search in src/ and crates/ directories
    for pattern in ["src/**/*.rs", "crates/**/src/**/*.rs"]:
        rust_files.extend(root_dir.glob(pattern))

    # Filter out target directory and test files for main analysis
    filtered_files = []
    for file_path in rust_files:
        if "target/" not in str(file_path) and not str(file_path).endswith("test.rs"):
            filtered_files.append(file_path)

    return filtered_files


def analyze_unsafe_usage(file_path: Path) -> Dict[str, Any]:
    """Analyze unsafe code usage in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e),
            'unsafe_blocks': [],
            'unsafe_functions': [],
            'unsafe_traits': [],
            'unsafe_impls': []
        }

    lines = content.split('\n')

    # Patterns for different unsafe constructs
    unsafe_block_pattern = re.compile(r'\bunsafe\s*\{')
    unsafe_fn_pattern = re.compile(r'\bunsafe\s+fn\s+(\w+)')
    unsafe_trait_pattern = re.compile(r'\bunsafe\s+trait\s+(\w+)')
    unsafe_impl_pattern = re.compile(r'\bunsafe\s+impl')

    analysis = {
        'file': str(file_path.relative_to(Path.cwd())),
        'unsafe_blocks': [],
        'unsafe_functions': [],
        'unsafe_traits': [],
        'unsafe_impls': []
    }

    for line_num, line in enumerate(lines, 1):
        line_stripped = line.strip()

        # Skip comments
        if line_stripped.startswith('//') or line_stripped.startswith('/*'):
            continue

        # Find unsafe blocks
        if unsafe_block_pattern.search(line):
            # Try to extract context
            context_start = max(0, line_num - 3)
            context_end = min(len(lines), line_num + 3)
            context = lines[context_start:context_end]

            analysis['unsafe_blocks'].append({
                'line': line_num,
                'content': line.strip(),
                'context': context
            })

        # Find unsafe functions
        unsafe_fn_match = unsafe_fn_pattern.search(line)
        if unsafe_fn_match:
            function_name = unsafe_fn_match.group(1)
            analysis['unsafe_functions'].append({
                'line': line_num,
                'name': function_name,
                'content': line.strip()
            })

        # Find unsafe traits
        unsafe_trait_match = unsafe_trait_pattern.search(line)
        if unsafe_trait_match:
            trait_name = unsafe_trait_match.group(1)
            analysis['unsafe_traits'].append({
                'line': line_num,
                'name': trait_name,
                'content': line.strip()
            })

        # Find unsafe impls
        if unsafe_impl_pattern.search(line):
            analysis['unsafe_impls'].append({
                'line': line_num,
                'content': line.strip()
            })

    return analysis


def get_crate_info() -> Dict[str, Any]:
    """Get information about the crate structure."""
    try:
        # Run cargo metadata to get crate information
        result = subprocess.run(
            ['cargo', 'metadata', '--format-version', '1'],
            capture_output=True,
            text=True,
            check=True
        )
        metadata = json.loads(result.stdout)

        crate_info = {
            'workspace_root': metadata['workspace_root'],
            'packages': []
        }

        for package in metadata['packages']:
            if package['source'] is None:  # Local packages only
                crate_info['packages'].append({
                    'name': package['name'],
                    'version': package['version'],
                    'manifest_path': package['manifest_path']
                })

        return crate_info
    except Exception as e:
        return {
            'error': str(e),
            'workspace_root': str(Path.cwd()),
            'packages': []
        }


def generate_unsafe_report(analyses: List[Dict[str, Any]], crate_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive unsafe code report."""
    total_unsafe_blocks = sum(len(analysis['unsafe_blocks']) for analysis in analyses)
    total_unsafe_functions = sum(len(analysis['unsafe_functions']) for analysis in analyses)
    total_unsafe_traits = sum(len(analysis['unsafe_traits']) for analysis in analyses)
    total_unsafe_impls = sum(len(analysis['unsafe_impls']) for analysis in analyses)

    files_with_unsafe = [
        analysis for analysis in analyses
        if (analysis['unsafe_blocks'] or analysis['unsafe_functions'] or
            analysis['unsafe_traits'] or analysis['unsafe_impls'])
    ]

    # Categorize by crate
    by_crate = {}
    for analysis in analyses:
        file_path = analysis['file']
        if 'crates/' in file_path:
            crate_name = file_path.split('crates/')[1].split('/')[0]
        else:
            crate_name = 'root'

        if crate_name not in by_crate:
            by_crate[crate_name] = {
                'files': [],
                'unsafe_blocks': 0,
                'unsafe_functions': 0,
                'unsafe_traits': 0,
                'unsafe_impls': 0
            }

        by_crate[crate_name]['files'].append(analysis)
        by_crate[crate_name]['unsafe_blocks'] += len(analysis['unsafe_blocks'])
        by_crate[crate_name]['unsafe_functions'] += len(analysis['unsafe_functions'])
        by_crate[crate_name]['unsafe_traits'] += len(analysis['unsafe_traits'])
        by_crate[crate_name]['unsafe_impls'] += len(analysis['unsafe_impls'])

    report = {
        'timestamp': str(subprocess.run(['date', '-u'], capture_output=True, text=True).stdout.strip()),
        'crate_info': crate_info,
        'summary': {
            'total_files_analyzed': len(analyses),
            'files_with_unsafe': len(files_with_unsafe),
            'total_unsafe_blocks': total_unsafe_blocks,
            'total_unsafe_functions': total_unsafe_functions,
            'total_unsafe_traits': total_unsafe_traits,
            'total_unsafe_impls': total_unsafe_impls,
            'unsafe_percentage': (len(files_with_unsafe) / len(analyses) * 100) if analyses else 0
        },
        'by_crate': by_crate,
        'detailed_analysis': analyses,
        'recommendations': generate_recommendations(analyses)
    }

    return report


def generate_recommendations(analyses: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on unsafe code analysis."""
    recommendations = []

    total_unsafe_blocks = sum(len(analysis['unsafe_blocks']) for analysis in analyses)
    files_with_unsafe = [
        analysis for analysis in analyses
        if analysis['unsafe_blocks'] or analysis['unsafe_functions']
    ]

    if total_unsafe_blocks == 0:
        recommendations.append("✅ No unsafe blocks found - excellent memory safety!")
    elif total_unsafe_blocks < 10:
        recommendations.append("✅ Low unsafe code usage - good safety practices")
    elif total_unsafe_blocks < 50:
        recommendations.append("⚠️ Moderate unsafe code usage - consider safety review")
    else:
        recommendations.append("❌ High unsafe code usage - requires thorough safety audit")

    if len(files_with_unsafe) > 0:
        recommendations.append(f"📋 Review {len(files_with_unsafe)} files containing unsafe code")
        recommendations.append("📝 Ensure all unsafe blocks have safety comments")
        recommendations.append("🧪 Add comprehensive tests for unsafe code paths")
        recommendations.append("👥 Consider peer review for all unsafe code changes")

    return recommendations


def main():
    root_dir = Path.cwd()

    print("Analyzing unsafe code usage...")

    # Find all Rust files
    rust_files = find_rust_files(root_dir)
    print(f"Found {len(rust_files)} Rust files to analyze")

    # Get crate information
    crate_info = get_crate_info()

    # Analyze each file
    analyses = []
    for file_path in rust_files:
        analysis = analyze_unsafe_usage(file_path)
        analyses.append(analysis)

    # Generate report
    report = generate_unsafe_report(analyses, crate_info)

    # Output JSON report
    print(json.dumps(report, indent=2))

    # Print summary to stderr for visibility
    summary = report['summary']
    print(f"\n=== Unsafe Code Analysis Summary ===", file=sys.stderr)
    print(f"Files analyzed: {summary['total_files_analyzed']}", file=sys.stderr)
    print(f"Files with unsafe code: {summary['files_with_unsafe']}", file=sys.stderr)
    print(f"Unsafe blocks: {summary['total_unsafe_blocks']}", file=sys.stderr)
    print(f"Unsafe functions: {summary['total_unsafe_functions']}", file=sys.stderr)
    print(f"Unsafe traits: {summary['total_unsafe_traits']}", file=sys.stderr)
    print(f"Unsafe impls: {summary['total_unsafe_impls']}", file=sys.stderr)
    print(f"Unsafe percentage: {summary['unsafe_percentage']:.1f}%", file=sys.stderr)

    print("\nRecommendations:", file=sys.stderr)
    for rec in report['recommendations']:
        print(f"  {rec}", file=sys.stderr)


if __name__ == '__main__':
    main()
