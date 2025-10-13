#!/usr/bin/env python3
"""
Replay parity failures for rapid debugging.

Reads a JSONL artifact row and re-runs both Rust and HF sides to compute:
- Median tau-b correlation
- Mean NLL delta
- Detailed step-by-step comparison
- Performance regression analysis

Usage:
    python scripts/replay_parity.py artifact.jsonl --row 5
    python scripts/replay_parity.py artifact.jsonl --prompt "Test prompt"
    python scripts/replay_parity.py parity_failures.jsonl --max-failures 10
    python scripts/replay_parity.py parity_failures.jsonl --filter-tau-b 0.90
"""

import json
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.stats import kendalltau


def load_artifact_row(jsonl_path: str, row: Optional[int] = None,
                      prompt: Optional[str] = None) -> Dict[str, Any]:
    """Load a specific row from JSONL artifact."""
    with open(jsonl_path) as f:
        lines = [json.loads(line) for line in f]

    if prompt:
        # Find row by prompt
        for line in lines:
            if line.get('prompt') == prompt:
                return line
        raise ValueError(f"No row found with prompt: {prompt}")
    elif row is not None:
        if row >= len(lines):
            raise ValueError(f"Row {row} out of range (file has {len(lines)} rows)")
        return lines[row]
    else:
        # Default to first row
        return lines[0]


def run_rust_eval(model_path: str, tokenizer_path: str, tf_ids: list[int],
                  bitnet_bin: str = "target/release/bitnet") -> Dict[str, Any]:
    """Run BitNet eval with teacher-forcing."""
    tf_ids_str = ','.join(map(str, tf_ids))

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        cmd = [
            bitnet_bin, 'eval',
            '--model', model_path,
            '--tokenizer', tokenizer_path,
            '--teacher-force-ids', tf_ids_str,
            '--dump-logit-steps', str(min(24, len(tf_ids) - 1)),
            '--logits-topk', '10',
            '--json-out', tmp.name
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Rust eval failed: {result.stderr}", file=sys.stderr)
            return {}

        with open(tmp.name) as f:
            return json.load(f)


def run_hf_eval(model_id: str, tf_ids: list[int]) -> Dict[str, Any]:
    """Run HF model evaluation with teacher-forcing."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    input_ids = torch.tensor([tf_ids[:-1]])  # All but last token
    labels = torch.tensor([tf_ids[1:]])  # All but first token

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        nll = outputs.loss.item()

        # Get logits for tau-b comparison
        logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

    return {
        'mean_nll': nll,
        'perplexity': np.exp(nll),
        'logits': logits.numpy()
    }


def compute_tau_b(rust_topk: list, hf_logits: np.ndarray) -> float:
    """Compute Kendall's tau-b between Rust top-k and HF logits."""
    rust_ids = [t[0] for t in rust_topk]
    rust_scores = [t[1] for t in rust_topk]

    hf_scores = [hf_logits[tid] for tid in rust_ids]

    tau, _ = kendalltau(rust_scores, hf_scores)
    return tau


def replay_comparison(artifact: Dict[str, Any], model_path: str,
                     tokenizer_path: str, hf_model_id: str,
                     bitnet_bin: str = "target/release/bitnet") -> None:
    """Replay and compare Rust vs HF."""
    tf_ids = artifact['tf_path']
    prompt = artifact.get('prompt', 'N/A')

    print(f"\n{'='*60}")
    print(f"Replaying: {prompt[:50]}...")
    print(f"TF path length: {len(tf_ids)} tokens")
    print(f"{'='*60}\n")

    # Run Rust eval
    print("Running Rust evaluation...")
    rust_result = run_rust_eval(model_path, tokenizer_path, tf_ids, bitnet_bin)

    if not rust_result:
        print("Failed to get Rust results")
        return

    # Run HF eval
    print("Running HF evaluation...")
    hf_result = run_hf_eval(hf_model_id, tf_ids)

    # Compare NLL
    rust_nll = rust_result['mean_nll']
    hf_nll = hf_result['mean_nll']
    nll_delta = abs(rust_nll - hf_nll)

    print(f"\n{'--- Results ---':^40}")
    print(f"Rust NLL:      {rust_nll:.6f}")
    print(f"HF NLL:        {hf_nll:.6f}")
    print(f"Delta NLL:     {nll_delta:.6f}")
    print(f"Rust PPL:      {rust_result['perplexity']:.2f}")
    print(f"HF PPL:        {hf_result['perplexity']:.2f}")

    # Compare tau-b for logit steps
    if 'logits_dump' in rust_result and rust_result['logits_dump']:
        taus = []
        for step in rust_result['logits_dump'][:10]:
            if step['step'] < len(hf_result['logits']):
                tau = compute_tau_b(step['topk'], hf_result['logits'][step['step']])
                taus.append(tau)

        if taus:
            median_tau = np.median(taus)
            print(f"\nMedian tau-b:  {median_tau:.4f} (n={len(taus)} steps)")
            print(f"Tau-b range:   [{min(taus):.4f}, {max(taus):.4f}]")

    # Status
    print(f"\n{'--- Status ---':^40}")
    if nll_delta < 0.01:
        print("✅ NLL parity PASS")
    else:
        print(f"❌ NLL parity FAIL (delta={nll_delta:.4f} > 0.01)")

    if 'taus' in locals() and median_tau > 0.60:
        print(f"✅ Tau-b parity PASS (median={median_tau:.4f})")
    else:
        print(f"❌ Tau-b parity FAIL")


def main():
    parser = argparse.ArgumentParser(description='Replay parity test from artifact')
    parser.add_argument('artifact', help='Path to JSONL artifact file')
    parser.add_argument('--row', type=int, help='Row number to replay (0-indexed)')
    parser.add_argument('--prompt', help='Find row by prompt text')
    parser.add_argument('--model', default='models/bitnet/model.gguf',
                       help='Path to GGUF model')
    parser.add_argument('--tokenizer', default='models/bitnet/tokenizer.json',
                       help='Path to tokenizer')
    parser.add_argument('--hf-model', default='1bitLLM/bitnet_b1_58-3B',
                       help='HuggingFace model ID')
    parser.add_argument('--bitnet-bin', default='target/release/bitnet',
                       help='Path to bitnet binary')

    args = parser.parse_args()

    # Load artifact row
    artifact = load_artifact_row(args.artifact, args.row, args.prompt)

    # Run comparison
    replay_comparison(
        artifact,
        args.model,
        args.tokenizer,
        args.hf_model,
        args.bitnet_bin
    )


if __name__ == '__main__':
    main()
