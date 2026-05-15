# Qwen3 constrained-answer scoring on i5-8250U

`SLM-CPU-008AC` selects the first Qwen3 constrained-answer scoring
seed that can safely unblock corpus work without pretending the full corpus is
green.

## Scope

The accepted seed is:

```text
model: Qwen/Qwen3-0.6B-GGUF / Qwen3-0.6B-Q8_0.gguf
sha256: 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
prompt_template: qwen
qwen_no_think: true
prompt: Say exactly: OK
expected first generated token: 3925 / OK
```

This seed is narrow by design. It proves the exact prompt policy and strict CPU
receipt path can produce a constrained answer that agrees with a known-good
Candle Qwen3 reference. It does not prove the full tiny corpus, broad answer
quality, multi-token stability, warm-session behavior, or performance.

## Scoring rule

A constrained-answer seed may pass when all of the following are true:

- the model SHA and rendered prompt policy match the reference artifact;
- prompt token IDs match exactly;
- bitnet-rs uses `loader.mode=real_gguf`, `tokenizer.source=gguf_metadata`,
  `tokenizer.strict=true`, `selected_backend=cpu-rust`, and
  `fallback_used=false`;
- generated token IDs match the reference for the scored bounded answer;
- decoded text matches the expected constrained answer after the normal strict
  tokenizer decode path.

Top-k logits remain diagnostic evidence. They should not fail this narrow
scoring seed when the chosen token ID and decoded answer match, but any top-k
numeric drift must remain visible for later math-hardening work.

## Rejected seeds

The no-thinking `2+2=` and sentence-form 2+2 first-token seeds are not valid
token-19 answer gates for Qwen3-0.6B Q8_0. The known-good reference chooses
token 17 / `2` for the calibrated short seed, so using token 19 / `4` as a
first-token pass condition would compare against the wrong prompt policy.

## Evidence

```text
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-no-think-say-ok-bitnet-rs-first-token.json
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-no-think-say-ok-reference-compare.json
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-no-think-say-ok-reference-validation.json
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-no-think-say-ok-scoring-diagnosis.json
```

`SLM-CPU-009` should start from this scoring rule and then expand to the full
tiny corpus one case at a time.

## SLM-CPU-009 corpus evidence

The first full strict i5-8250U corpus run under the selected no-thinking policy
was recorded by SLM-CPU-009B before the calibrated corpus replacement:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-answer-corpus.json
```

That run passed four of five cases:

- `capital_france`
- `repeat_colors`
- `say_ok`
- `yes_no_water`

`math_2_plus_2` remains a real content miss in the preserved per-case receipt:
the model generates token `17` (`2`) followed by EOS, so the case keeps
`gate_exact_trimmed` in `failed_rules`. That evidence was useful for calibrating
SLM-CPU-009, but it does not prove arithmetic reasoning.

## SLM-CPU-009 calibrated corpus

The `math_2_plus_2` prompt is not used as the SLM-CPU-009 green gate because
the strict Qwen no-thinking policy and the verified Qwen3-0.6B Q8_0 artifact
answer `2`, not `4`, across the tested sentence and worded math variants. That
miss remains documented in the SLM-CPU-009B evidence.

For the constrained-answer green gate, the corpus replaces that unsuitable math
seed with a calibrated exact digit seed:

```text
id: say_four
prompt: Say exactly: 4
expected: 4
max_new_tokens: 1
```

This keeps the lane claim narrow: the corpus can prove constrained answer
following under strict CPU receipts, but it does not prove arithmetic reasoning
or broad chat quality.

The current SLM-CPU-009 aggregate at
`ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-answer-corpus.json` records the
calibrated corpus as five passing cases with strict GGUF tokenizer authority,
`selected_backend=cpu-rust`, and `fallback_used=false`.
