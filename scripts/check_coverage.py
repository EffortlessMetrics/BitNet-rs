#!/usr/bin/env python3
"""
Coverage threshold checker for release validation.
"""

import json
import sys
import argparse
from pathlib import Path


def check_coverage_threshold(coverage_file: Path, threshold: float) -> bool:
    """Check if coverage meets the minimum threshold."""
    try:
        with open(coverage_file, 'r') as f:
            coverage_data = json.load(f)
        
        # Extract coverage percentage from tarpaulin report
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
                print("No coverage data found")
                return False
            
            coverage_percentage = (covered_lines / total_lines) * 100
        else:
            # Alternative format
            coverage_percentage = coverage_data.get('coverage', 0.0)
        
        print(f"Coverage: {coverage_percentage:.2f}%")
        print(f"Threshold: {threshold:.2f}%")
        
        if coverage_percentage >= threshold:
            print("✅ Coverage threshold met")
            return True
        else:
            print(f"❌ Coverage below threshold ({coverage_percentage:.2f}% < {threshold:.2f}%)")
            return False
            
    except Exception as e:
        print(f"Error reading coverage file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Check coverage threshold')
    parser.add_argument('coverage_file', type=Path, help='Coverage report file (JSON)')
    parser.add_argument('threshold', type=float, help='Minimum coverage threshold (percent)')
    
    args = parser.parse_args()
    
    if not args.coverage_file.exists():
        print(f"Coverage file not found: {args.coverage_file}")
        sys.exit(1)
    
    if not check_coverage_threshold(args.coverage_file, args.threshold):
        sys.exit(1)
    
    print("Coverage check passed")


if __name__ == '__main__':
    main()