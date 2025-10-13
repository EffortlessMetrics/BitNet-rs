"""
Logit parity tests using Kendall's tau.

Tests that BitNet produces similar rank orderings of top-k tokens
compared to reference implementations.
"""

import os
import statistics as stats
from hypothesis import given, settings, strategies as st, HealthCheck, note, assume
from .run_model import BitNetRunner, HFRuntimeRunner
from .metrics import kendalls_tau_b_scored
from .strategies import prompt_strategy

# Configuration from environment
TAU_STEPS = int(os.environ.get("TAU_STEPS", os.environ.get("LOGIT_STEPS", "32")))
TOPK = int(os.environ.get("LOGIT_TOPK", "10"))
TAU_MIN = float(os.environ.get("TAU_MIN", "0.60"))
MAX_TOK = int(os.environ.get("PROP_MAX_NEW_TOKENS", "128"))

MIN_INFORMATIVE_STEPS = 8  # Minimum number of informative steps required

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
                # For determinism check, use regular tau since same model
                from .metrics import kendalls_tau
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

    # Cross-system parity test using teacher-forcing on shared path
    note(f"Testing BitNet vs HF parity with teacher-forcing")

    # 1) Run greedy on BitNet to get chosen token IDs
    greedy = rs["bitnet"].run(
        prompt, MAX_TOK, seed=seed, greedy=True,
        dump_logits_steps=TAU_STEPS, topk=TOPK
    )

    # Extract chosen token IDs from BitNet's greedy run
    chosen = [s["chosen_id"] for s in greedy.meta.get("logits_dump", [])[:TAU_STEPS]
              if s.get("chosen_id") is not None]

    if len(chosen) < 2:
        assume(False)  # Skip trivial cases

    # 2) Tokenize prompt on HF (no specials) and prepend BOS if defined
    tok = rs["hf"].tokenizer()
    bos = tok.bos_token_id
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_path = ([bos] if bos is not None else []) + prompt_ids + chosen

    # Ensure we have enough tokens
    assume(len(full_path) > 1)

    # 3) Teacher-force both sides on the same path
    A = rs["bitnet"].run_teacher_force(full_path, steps=TAU_STEPS, topk=TOPK)
    B = rs["hf"].run_teacher_force(full_path, steps=TAU_STEPS, topk=TOPK)

    # 4) Compute τ-b per step over intersection, keep informative steps
    TAU_TIE_EPS = float(os.environ.get("TAU_TIE_EPS", "1e-6"))

    taus = []
    for a_step, b_step in zip(A, B):
        # Get top-k pairs: [(token_id, logit)]
        a_pairs = a_step["topk"]  # Already in descending order
        b_pairs = b_step["topk"]

        # Skip steps with low intersection (not informative)
        a_ids = set(tid for tid, _ in a_pairs)
        b_ids = set(tid for tid, _ in b_pairs)
        inter = a_ids & b_ids
        if len(inter) < 3:
            continue

        # Compute score-aware Kendall's tau-b that handles ties properly
        tau = kendalls_tau_b_scored(a_pairs, b_pairs, eps=TAU_TIE_EPS)
        taus.append(tau)
        note(f"Step {a_step['step']}: τ-b = {tau:.3f} (|intersection| = {len(inter)})")

    # Require minimum informative steps
    assume(len(taus) >= min(MIN_INFORMATIVE_STEPS, TAU_STEPS // 3))

    # Check median tau
    median_tau = stats.median(taus)
    note(f"Median Kendall's τ-b = {median_tau:.3f} (threshold: {TAU_MIN})")

    # Save artifacts on failure for debugging
    if median_tau < TAU_MIN:
        from crossval.props.util import append_jsonl

        artifact = {
            "test": "logit_parity",
            "prompt": prompt,
            "seed": seed,
            "median_tau": median_tau,
            "tau_values": taus,
            "threshold": TAU_MIN,
            "full_path": full_path[:50],  # First 50 tokens of path
            "bitnet_logits": A[:5] if A else [],  # First 5 steps
            "hf_logits": B[:5] if B else [],
            "greedy_text": greedy.text[:200]
        }

        # Use unified JSONL artifact persistence
        artifact_path = os.environ.get("PARITY_ARTIFACT", "artifacts/parity_failures.jsonl")
        append_jsonl(artifact_path, artifact)
        note(f"Failure artifact appended to: {artifact_path}")

    assert median_tau >= TAU_MIN, (
        f"Median Kendall's τ-b too low: {median_tau:.3f} < {TAU_MIN}\n"
        f"Prompt: {prompt}\n"
        f"Informative steps: {len(taus)}/{TAU_STEPS}\n"
        f"Tau values: {[f'{t:.3f}' for t in taus]}"
    )
