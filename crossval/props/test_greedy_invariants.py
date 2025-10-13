"""
Greedy argmax invariant tests.

Tests that greedy decoding always selects the highest-probability token
and that teacher-forcing reproduces the same path.
"""

import os
from hypothesis import given, settings, strategies as st, HealthCheck, note, assume
from .run_model import BitNetRunner
from .strategies import prompt_strategy

BITNET_BIN = os.environ.get("BITNET_BIN", "target/release/bitnet")
MODEL_PATH = os.environ.get("MODEL_PATH")
TOKENIZER = os.environ.get("TOKENIZER")


@settings(
    max_examples=int(os.environ.get("PROP_EXAMPLES", "10")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow]
)
@given(prompt=prompt_strategy(), seed=st.integers(min_value=1, max_value=2**32-1))
def test_greedy_argmax_invariant(prompt, seed):
    """
    Test that greedy decoding always selects argmax(logits).
    """
    if not MODEL_PATH or not TOKENIZER:
        return  # Skip if not configured

    runner = BitNetRunner(BITNET_BIN, MODEL_PATH, TOKENIZER, threads=1)

    # Run with greedy and capture logits
    result = runner.run(
        prompt,
        max_new_tokens=32,
        seed=seed,
        greedy=True,
        dump_logits_steps=32,
        topk=1  # Only need the top token
    )

    # Check that chosen_id == argmax for each step
    violations = []
    for step in result.meta.get("logits_dump", []):
        topk = step.get("topk", [])
        chosen = step.get("chosen_id")

        if topk and chosen is not None:
            argmax_id = topk[0][0] if isinstance(topk[0], tuple) else topk[0]
            if chosen != argmax_id:
                violations.append({
                    "step": step.get("step", -1),
                    "chosen": chosen,
                    "argmax": argmax_id
                })

    if violations:
        note(f"Greedy argmax violations: {violations}")

    assert not violations, (
        f"Greedy decoding did not select argmax at {len(violations)} steps\n"
        f"First violation: {violations[0] if violations else None}"
    )


@settings(
    max_examples=int(os.environ.get("PROP_EXAMPLES", "5")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow]
)
@given(prompt=prompt_strategy(), seed=st.integers(min_value=1, max_value=2**32-1))
def test_teacher_forcing_consistency(prompt, seed):
    """
    Test that teacher-forcing on a greedy path reproduces the same chosen_ids.
    """
    if not MODEL_PATH or not TOKENIZER:
        return  # Skip if not configured

    runner = BitNetRunner(BITNET_BIN, MODEL_PATH, TOKENIZER, threads=1)

    # 1) Run greedy to get a path
    greedy = runner.run(
        prompt,
        max_new_tokens=16,
        seed=seed,
        greedy=True,
        dump_logits_steps=16,
        topk=1
    )

    # Extract the full token path (need to tokenize prompt + chosen ids)
    chosen = [s["chosen_id"] for s in greedy.meta.get("logits_dump", [])[:16]
              if s.get("chosen_id") is not None]

    if len(chosen) < 2:
        assume(False)  # Skip trivial cases

    # Tokenize prompt and prepend BOS if defined
    tok = runner.tokenizer()
    bos = tok.bos_token_id
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_path = ([bos] if bos is not None else []) + prompt_ids + chosen

    # 2) Teacher-force on the same path
    tf_result = runner.run_teacher_force(full_path, steps=len(full_path)-1, topk=1)

    # 3) Check that argmax at each step matches the original chosen_id
    mismatches = []
    for i, tf_step in enumerate(tf_result):
        if i < len(chosen) - 1:
            expected = chosen[i + 1]  # Next token in path
            if tf_step["topk"]:
                argmax = tf_step["topk"][0][0]
                if argmax != expected:
                    mismatches.append({
                        "step": i,
                        "expected": expected,
                        "argmax": argmax
                    })

    if mismatches:
        note(f"Teacher-forcing mismatches: {mismatches}")

    assert not mismatches, (
        f"Teacher-forcing did not reproduce greedy path\n"
        f"{len(mismatches)} mismatches found\n"
        f"First mismatch: {mismatches[0] if mismatches else None}"
    )
