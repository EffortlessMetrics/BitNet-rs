# SLM Reference Divergence Artifacts

SLM reference divergence artifacts compare a `bitnet-rs` run against a known-good external run. They are offline diagnostics: the CLI validates and normalizes the artifact, but it does not run the external engine.

## Artifact Shape

Use `artifact_kind = "backend_reference_compare"`:

```json
{
  "schema_version": "1.0.0",
  "artifact_kind": "backend_reference_compare",
  "model_sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
  "model_family": "qwen3",
  "prompt_text": "What is 2+2?",
  "prompt_template": "qwen",
  "bos": false,
  "reference": {
    "backend": "known-good-external",
    "kernel": "reference",
    "prompt_ids": [1, 2, 3],
    "generated_ids": [4],
    "text": "4",
    "topk_step0": [[4, 10.0], [5, 1.0]],
    "chosen_id": 4
  },
  "bitnet_rs": {
    "backend": "cpu-rust",
    "kernel": "dense-q8_0-reference",
    "prompt_ids": [1, 2, 3],
    "generated_ids": [5],
    "text": "5",
    "topk_step0": [[5, 10.0], [4, 1.0]],
    "chosen_id": 5
  }
}
```

`candidate` is accepted as an alias for `bitnet_rs` so early hand-written artifacts can use the same shape as older notes.

## Validation

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features "cpu,full-cli" -- `
  reference-compare `
  --artifact ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-reference-divergence-example.json `
  --json-out target\bitnet\receipts\qwen3-reference-divergence.json
```

Add `--require-match` when a lane is expected to match the reference exactly. Without it, validation succeeds for schema-valid divergence artifacts and records the first divergence.

## First-Token Logit Capture

For `SLM-CPU-006`, capture the first generated token with top-k logits on the
Rust side before comparing against an external reference:

```powershell
$env:BITNET_STRICT_MODE = "1"
$env:BITNET_DISABLE_MINIMAL_LOADER = "1"
$env:RAYON_NUM_THREADS = "8"

cargo run --locked -p bitnet-cli --no-default-features --features "cpu,full-cli" -- `
  --device cpu `
  run `
  --model models\slm\Qwen3-0.6B-Q8_0.gguf `
  --prompt-template qwen `
  --prompt "What is 2+2? Answer with only the number." `
  --max-new-tokens 1 `
  --temperature 0.0 `
  --greedy `
  --deterministic `
  --strict-loader `
  --strict-tokenizer `
  --logits-dump-steps 1 `
  --logits-topk 10 `
  --assert-greedy `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-bitnet-rs-first-token-topk.json
```

`--logits-dump-steps` is an alias for the existing `--dump-logit-steps`
receipt field. The run receipt must include `tokens.prompt_ids`,
`tokens.generated_ids`, `text`, `logits_dump[0].chosen_id`, and
`logits_dump[0].top_logits`.

The external reference artifact should use the same model SHA, prompt text,
Qwen template, BOS policy, prompt IDs, and greedy-equivalent first-token
settings. If the external runner cannot emit logits, keep the artifact
diagnostic-only and do not close `SLM-CPU-006` as a logit-localized result.

## First-Drift Checkpoint Trace

For `SLM-CPU-007`, capture bounded bitnet-rs checkpoint summaries for the first
generated token. This uses the same strict run shape as first-token top-k capture
and writes JSONL tensor summaries instead of full tensors:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features "cpu,full-cli" -- `
  --device cpu `
  run `
  --model models\slm\Qwen3-0.6B-Q8_0.gguf `
  --prompt-template qwen `
  --prompt "What is 2+2? Answer with only the number." `
  --max-new-tokens 1 `
  --temperature 0.0 `
  --greedy `
  --deterministic `
  --strict-loader `
  --strict-tokenizer `
  --logits-dump-steps 1 `
  --logits-topk 10 `
  --assert-greedy `
  --qwen-trace-jsonl ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-bitnet-rs-checkpoints.jsonl `
  --qwen-trace-layer 0 `
  --qwen-trace-full-prompt `
  --json-out ci\slm-cpu\intel-i5-8250u\2026-05-07\qwen3-bitnet-rs-first-token-topk.json
```

Each `qwen_trace_tensor` row records the checkpoint name, step, optional layer,
shape, dtype, finite/nonfinite counts, mean, RMS, min/max, checksum, and a small
sample. The intended first pass is layer 0 with stages such as
`decode.input_embedding`, `block.attention_norm`, `attention.q_proj`,
`attention.k_proj`, `attention.v_proj`, `attention.q_rope`,
`attention.k_rope`, `attention.scores_post_mask`, `attention.weights`,
`attention.o_proj`, `block.ffn_norm`, `mlp.gate_proj`, `mlp.up_proj`,
`mlp.down_proj`, `block.output`, `model.final_norm`, and `lm_head.logits`.

If the external reference provides prompt token IDs, pass them through
`--qwen-trace-prompt-ids 1,2,3` to force the bitnet-rs trace onto exactly the
same prompt IDs before comparing checkpoint summaries. The trace remains a
diagnostic artifact: it localizes the first drift point and does not claim Qwen
answer quality or throughput.

The `SLM-CPU-007B` i5-8250U capture records the first layer-0 trace at:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-bitnet-rs-layer0-checkpoints.jsonl
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-bitnet-rs-first-token-layer0.json
ci/slm-cpu/intel-i5-8250u/2026-05-07/qwen3-first-drift-checkpoint-classification.json
```

That artifact confirms prompt IDs still match the reference and the first
comparable mismatch remains `lm_head.top_logits`: the reference chooses token
`19` (`4`) while bitnet-rs chooses token `4594` (`ł`). Because the known-good
reference does not yet include internal checkpoint dumps, the internal first
drift is recorded as `reference-missing`, not as an attention, MLP, or output
head root-cause claim.

## Divergence Classification

The validator records a `classification` alongside
`comparison.first_divergence.phase`:

| Phase | Classification | Meaning |
| --- | --- | --- |
| `prompt` | `prompt_tokenizer_template` | Prompt IDs differ; audit tokenizer source, Qwen template, BOS/EOS policy, or special-token handling first. |
| `logits` | `logits_or_shared_transformer_math` | Prompt IDs match, but first-step top-k logits differ; inspect Q8_0 dequantization, tensor orientation, Q/K/V/O projections, RoPE, RMSNorm, GQA, output head, and vocab indexing. |
| `sampler` | `sampler` | Top-k evidence matches, but the chosen/generated token differs; inspect greedy argmax, temperature-zero behavior, tie-breaking, and stop/EOS handling. |
| `decode` | `output_head_vocab_indexing_or_shared_transformer_math` | Generated IDs differ and top-k evidence is missing or inconclusive; collect first-token top-k before assigning blame. |
| `text` | `tokenizer_decode` | Token IDs match, but decoded text differs; inspect decode cleanup, byte fallback, and special-token filtering. |

## Claim Boundary

This artifact may show whether the first mismatch is in prompt IDs, generated IDs, decoded text, or top-k logits. It does not prove general answer quality, sustained 8250U throughput, server inference, GPU execution, OpenVINO execution, UHD 620 execution, or NPU execution.
