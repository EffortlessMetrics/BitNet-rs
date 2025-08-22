"""
Teacher-forcing NLL parity tests.

Tests that BitNet's mean negative log-likelihood matches reference
implementations when computing perplexity on text corpora.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from hypothesis import given, settings, strategies as st, HealthCheck

# Configuration
BITNET_BIN = os.environ.get("BITNET_BIN", "target/release/bitnet")
MODEL_PATH = os.environ.get("MODEL_PATH")
TOKENIZER = os.environ.get("TOKENIZER")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID")
PPL_FILE = os.environ.get("PPL_FILE", "crossval/data/ppl_smoke.txt")
DELTA_NLL_MAX = float(os.environ.get("DELTA_NLL_MAX", "1e-2"))


def run_bitnet_eval(text_path: str) -> float:
    """
    Run BitNet eval command and extract mean NLL.
    """
    if not MODEL_PATH or not TOKENIZER:
        raise ValueError("MODEL_PATH and TOKENIZER must be set")
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.close()
    
    try:
        args = [
            BITNET_BIN, "eval",
            "--model", MODEL_PATH,
            "--tokenizer", TOKENIZER,
            "--text-file", text_path,
            "--deterministic",
            "--threads", "1",
            "--json-out", tmp.name
        ]
        
        p = subprocess.run(
            args, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=60
        )
        
        if p.returncode != 0:
            stderr = p.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"BitNet eval failed: {stderr}")
        
        with open(tmp.name, "r", encoding="utf-8") as f:
            result = json.load(f)
        
        # Extract mean NLL from results
        if isinstance(result, dict):
            # Try different possible keys
            if "mean_nll" in result:
                return float(result["mean_nll"])
            elif "perplexity" in result:
                # Convert perplexity back to NLL
                import math
                return math.log(float(result["perplexity"]))
            elif "results" in result and "mean_nll" in result["results"]:
                return float(result["results"]["mean_nll"])
        
        raise ValueError(f"Could not extract mean_nll from result: {result}")
        
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def run_hf_eval(text_path: str, model_id: str) -> float:
    """
    Compute mean NLL using HuggingFace transformers.
    
    Uses teacher-forcing to compute cross-entropy loss.
    """
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError:
        raise ImportError("transformers and torch required for HF evaluation")
    
    # Set deterministic mode
    set_seed(42)  # Match BitNet seed
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32
    ).eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Get special tokens
    bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id
    
    total_nll = 0.0
    total_tokens = 0
    
    # Process each line
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            
            # Tokenize without special tokens to match BitNet
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            
            # Add BOS manually (matching BitNet policy)
            if bos_id is not None:
                ids = [bos_id] + ids
            
            # Convert to tensor
            input_ids = torch.tensor([ids], dtype=torch.long).to(device)
            
            if input_ids.size(1) < 2:
                continue
            
            # Teacher-forcing: compute loss manually for exact control
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits  # [B, T, V]
                
                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                # Compute log softmax for numerical stability
                log_probs = F.log_softmax(shift_logits, dim=-1)
                
                # Gather the log probs for actual next tokens
                flat_log_probs = log_probs.view(-1, log_probs.size(-1))
                flat_labels = shift_labels.view(-1)
                
                # Compute NLL (negative log likelihood)
                nll = -flat_log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
                
                # Mask out any padding tokens if present
                if pad_id is not None:
                    mask = flat_labels != pad_id
                    nll = nll * mask.float()
                    valid_tokens = mask.sum().item()
                else:
                    valid_tokens = flat_labels.numel()
                
                # Accumulate
                total_nll += nll.sum().item()
                total_tokens += valid_tokens
    
    if total_tokens == 0:
        return 0.0
    
    return total_nll / total_tokens


@given(st.just(0))  # Single test case
@settings(
    max_examples=1,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow]
)
def test_mean_nll_parity(_):
    """
    Test that BitNet's mean NLL matches reference implementation.
    
    This uses teacher-forcing to compute perplexity on a test corpus,
    which is a more robust measure than sampling-based comparisons.
    """
    # Check configuration
    assert MODEL_PATH, "MODEL_PATH environment variable must be set"
    assert TOKENIZER, "TOKENIZER environment variable must be set"
    assert HF_MODEL_ID, "HF_MODEL_ID environment variable must be set for parity testing"
    
    # Check test file exists
    if not Path(PPL_FILE).exists():
        # Create a simple test file if it doesn't exist
        Path(PPL_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(PPL_FILE, "w") as f:
            f.write("The quick brown fox jumps over the lazy dog.\n")
            f.write("BitNet is a 1-bit transformer architecture.\n")
            f.write("Machine learning models can be quantized for efficiency.\n")
            f.write("Rust provides memory safety without garbage collection.\n")
            f.write("Property-based testing helps find edge cases.\n")
    
    # Run evaluations
    bitnet_nll = run_bitnet_eval(PPL_FILE)
    hf_nll = run_hf_eval(PPL_FILE, HF_MODEL_ID)
    
    # Compute delta
    delta = abs(bitnet_nll - hf_nll)
    
    # Save artifacts on failure
    if delta > DELTA_NLL_MAX:
        import tempfile
        
        artifact = {
            "test_file": str(PPL_FILE),
            "bitnet_nll": bitnet_nll,
            "hf_nll": hf_nll,
            "delta": delta,
            "threshold": DELTA_NLL_MAX,
            "model_path": MODEL_PATH,
            "hf_model_id": HF_MODEL_ID
        }
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_nll_parity_fail.json', delete=False) as f:
            json.dump(artifact, f, indent=2)
            print(f"Failure artifact saved to: {f.name}")
        
        # Also append to cumulative log if specified
        failure_log = os.environ.get("PARITY_FAILURE_LOG")
        if failure_log:
            with open(failure_log, "a") as f:
                f.write(json.dumps(artifact) + "\n")
    
    # Check parity
    assert delta <= DELTA_NLL_MAX, (
        f"NLL parity failed: |Δ| = {delta:.4f} > {DELTA_NLL_MAX}\n"
        f"BitNet NLL: {bitnet_nll:.6f}\n"
        f"HF NLL: {hf_nll:.6f}\n"
        f"Test file: {PPL_FILE}"
    )
    
    print(f"✓ NLL parity passed: BitNet={bitnet_nll:.4f}, HF={hf_nll:.4f}, Δ={delta:.4f}")


if __name__ == "__main__":
    # Allow running directly for debugging
    test_mean_nll_parity(0)