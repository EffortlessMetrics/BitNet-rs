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

## Claim Boundary

This artifact may show whether the first mismatch is in prompt IDs, generated IDs, decoded text, or top-k logits. It does not prove general answer quality, sustained 8250U throughput, server inference, GPU execution, OpenVINO execution, UHD 620 execution, or NPU execution.
