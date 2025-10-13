#!/usr/bin/env python3
"""
Check greedy argmax invariant from CLI JSON output.
Ensures that when --greedy is used, the chosen token at each step
matches the argmax of the logits.
"""

import sys
import json
import argparse
from typing import Dict, List, Any

def check_greedy_invariant(json_path: str) -> bool:
    """
    Check that all steps satisfy greedy argmax invariant.
    Returns True if valid, False otherwise.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract logits dump
    logits_dump = data.get('logits_dump', [])
    if not logits_dump:
        print("No logits dump found in JSON", file=sys.stderr)
        return False

    all_valid = True
    for step_idx, step in enumerate(logits_dump):
        top_logits = step.get('top_logits', [])
        chosen_id = step.get('chosen_id')

        if not top_logits or chosen_id is None:
            print(f"Step {step_idx}: Missing data", file=sys.stderr)
            continue

        # Find the argmax token
        argmax_token = None
        max_logit = float('-inf')

        # Check all top logits
        for entry in top_logits:
            token_id = entry['token_id']
            logit_val = entry['logit']

            # Handle NaN/inf
            if not isinstance(logit_val, (int, float)) or not (-1e10 < logit_val < 1e10):
                continue

            if logit_val > max_logit:
                max_logit = logit_val
                argmax_token = token_id

        # Verify chosen matches argmax
        if argmax_token != chosen_id:
            print(f"Step {step_idx}: Greedy violation! argmax={argmax_token} (logit={max_logit:.4f}) but chosen={chosen_id}", file=sys.stderr)

            # Show top-5 for debugging
            print(f"  Top logits at step {step_idx}:", file=sys.stderr)
            for i, entry in enumerate(top_logits[:5]):
                marker = " <-- CHOSEN" if entry['token_id'] == chosen_id else ""
                print(f"    {i+1}. token={entry['token_id']} logit={entry['logit']:.4f}{marker}", file=sys.stderr)

            all_valid = False

    if all_valid:
        print(f"✓ Greedy invariant holds for all {len(logits_dump)} steps")

    return all_valid

def main():
    parser = argparse.ArgumentParser(description='Check greedy argmax invariant')
    parser.add_argument('json_file', help='Path to CLI JSON output')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    if not check_greedy_invariant(args.json_file):
        sys.exit(7)  # EXIT_ARGMAX_MISMATCH

if __name__ == '__main__':
    main()
