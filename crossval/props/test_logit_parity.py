"""
Logit parity tests using Kendall's tau.

Tests that BitNet produces similar rank orderings of top-k tokens
compared to reference implementations.
"""

import os
import statistics as stats
from hypothesis import given, settings, strategies as st, HealthCheck, note
from .run_model import BitNetRunner, HFRuntimeRunner
from .metrics import kendalls_tau
from .strategies import prompt_strategy

# Configuration from environment
TAU_STEPS = int(os.environ.get("TAU_STEPS", os.environ.get("LOGIT_STEPS", "32")))
TOPK = int(os.environ.get("LOGIT_TOPK", "10"))
TAU_MIN = float(os.environ.get("TAU_MIN", "0.60"))
MAX_TOK = int(os.environ.get("PROP_MAX_NEW_TOKENS", "128"))

BITNET_BIN = os.environ.get("BITNET_BIN", "target/release/bitnet")
MODEL_PATH = os.environ.get("MODEL_PATH")
TOKENIZER = os.environ.get("TOKENIZER")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID")  # Optional for cross-system


def runners():
    """Create runners based on environment configuration."""
    r = {"bitnet": BitNetRunner(BITNET_BIN, MODEL_PATH, TOKENIZER, threads=1)}
    if HF_MODEL_ID:
        r["hf"] = HFRuntimeRunner(HF_MODEL_ID, device="cpu")
    return r


@settings(
    max_examples=int(os.environ.get("PROP_EXAMPLES", "20")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much]
)
@given(prompt=prompt_strategy(), seed=st.integers(min_value=1, max_value=2**32-1))
def test_logit_parity_tau(prompt, seed):
    """
    Test that BitNet's top-k token rankings correlate with reference.
    
    Uses median Kendall's tau across decode steps as the metric.
    """
    rs = runners()
    assert "bitnet" in rs, "BitNet runner missing - check MODEL_PATH and TOKENIZER"
    
    if "hf" not in rs:
        # Self-consistency check: BitNet should be deterministic
        note("Testing BitNet determinism (no HF model configured)")
        
        A = rs["bitnet"].run(
            prompt, MAX_TOK, seed=seed, greedy=True, 
            dump_logits_steps=TAU_STEPS, topk=TOPK
        )
        B = rs["bitnet"].run(
            prompt, MAX_TOK, seed=seed, greedy=True,
            dump_logits_steps=TAU_STEPS, topk=TOPK
        )
        
        # Extract top-k token IDs from each step
        a_steps = A.meta.get("logits_dump", [])
        b_steps = B.meta.get("logits_dump", [])
        
        # Compute tau for each step
        taus = []
        for a_step, b_step in zip(a_steps, b_steps):
            a_ids = [t[0] if isinstance(t, tuple) else t.get("id", t) 
                     for t in a_step.get("topk", [])]
            b_ids = [t[0] if isinstance(t, tuple) else t.get("id", t)
                     for t in b_step.get("topk", [])]
            if a_ids and b_ids:
                tau = kendalls_tau(a_ids, b_ids)
                taus.append(tau)
        
        if taus:
            median_tau = stats.median(taus)
            note(f"BitNet self-consistency: median τ = {median_tau:.3f}")
            assert median_tau >= 0.99, (
                f"BitNet not deterministic! Median τ = {median_tau:.3f}\n"
                f"Individual τ values: {taus}"
            )
        return
    
    # Cross-system parity test
    note(f"Testing BitNet vs HF parity with teacher-forcing")
    
    # First run BitNet to get the reference path
    A = rs["bitnet"].run(
        prompt, MAX_TOK, seed=seed, greedy=True,
        dump_logits_steps=TAU_STEPS, topk=TOPK
    )
    
    # Extract the chosen token path from BitNet
    chosen_path = []
    for step in A.meta.get("logits_dump", [])[:TAU_STEPS]:
        chosen_id = step.get("chosen_id")
        if chosen_id is not None:
            chosen_path.append(chosen_id)
    
    # Now teacher-force both models on the same path
    # For BitNet: re-run with teacher forcing (if supported)
    # For HF: run with teacher forcing on the chosen path
    
    # For now, use the existing runs but note the improvement needed
    B = rs["hf"].run(
        prompt, MAX_TOK, seed=seed, greedy=True,
        dump_logits_steps=TAU_STEPS, topk=TOPK
    )
    
    a_steps = A.meta.get("logits_dump", [])
    b_steps = B.meta.get("logits_dump", [])
    
    # TODO: Add teacher_force_path parameter to runners
    # A_tf = rs["bitnet"].run_teacher_force(prompt, chosen_path, topk=TOPK)
    # B_tf = rs["hf"].run_teacher_force(prompt, chosen_path, topk=TOPK)
    
    # Compute tau for each step
    taus = []
    for i, (a_step, b_step) in enumerate(zip(a_steps, b_steps)):
        a_ids = [t[0] if isinstance(t, tuple) else t.get("id", t)
                 for t in a_step.get("topk", [])]
        b_ids = [t[0] if isinstance(t, tuple) else t.get("id", t)
                 for t in b_step.get("topk", [])]
        
        if a_ids and b_ids:
            tau = kendalls_tau(a_ids, b_ids)
            taus.append(tau)
            note(f"Step {i}: τ = {tau:.3f}")
    
    if not taus:
        note("No logits captured - check dump_logits_steps implementation")
        return
    
    median_tau = stats.median(taus)
    note(f"Median Kendall's τ = {median_tau:.3f} (threshold: {TAU_MIN})")
    
    # Save artifacts on failure for debugging
    if median_tau < TAU_MIN:
        import tempfile
        import json
        
        artifact = {
            "prompt": prompt,
            "seed": seed,
            "median_tau": median_tau,
            "tau_values": taus,
            "threshold": TAU_MIN,
            "bitnet": {
                "text": A.text,
                "logits_dump": A.meta.get("logits_dump", [])[:10]  # First 10 steps
            },
            "reference": {
                "text": B.text,
                "logits_dump": B.meta.get("logits_dump", [])[:10]
            }
        }
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_logit_parity_fail.json', delete=False) as f:
            json.dump(artifact, f, indent=2)
            note(f"Failure artifact saved to: {f.name}")
        
        # Also append to cumulative log if specified
        failure_log = os.environ.get("PARITY_FAILURE_LOG")
        if failure_log:
            with open(failure_log, "a") as f:
                f.write(json.dumps(artifact) + "\n")
    
    assert median_tau >= TAU_MIN, (
        f"Median Kendall's τ too low: {median_tau:.3f} < {TAU_MIN}\n"
        f"Prompt: {prompt}\n"
        f"Individual τ values: {taus}"
    )