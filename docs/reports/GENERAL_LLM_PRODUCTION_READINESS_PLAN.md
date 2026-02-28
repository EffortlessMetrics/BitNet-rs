# BitNet-rs General LLM Production Readiness Plan

## 1) Objective and Scope

This document turns the executive summary into an implementation blueprint for extending BitNet-rs from optimized BitNet inference to a production-ready, **general LLM platform**.

### In scope
- Standard model loading (HF SafeTensors, GGUF full precision, optional ONNX runtime path).
- Tokenizer compatibility beyond current BitNet defaults.
- A practical training/fine-tuning pipeline (Python-first, Rust-integrated artifact flow).
- High-throughput inference service (request validation, auth, batching, observability, SLOs).
- Quantization and performance roadmap (FP16/BF16/INT8/INT4).
- Deployment, rollout, and backward compatibility for legacy BitNet models.

### Out of scope (initial phases)
- Re-implementing full distributed training in Rust.
- Supporting every model architecture on day 1.
- Building a new orchestration platform from scratch.

## 2) Current State (Condensed)

BitNet-rs already has strong foundations:
- Modular Rust workspace and microcrates.
- BitNet-centric model loading and quantization paths.
- Optimized CPU/GPU kernels for BitNet formats.
- CLI inference and chat workflows.
- Server health endpoints and monitoring scaffolding.

Primary gap: generalized model/training/serving stack for mainstream LLM workflows.

## 3) Target Architecture (Pragmatic)

```text
Data -> Tokenizer build/validation -> Training/Fine-tuning (Python stack)
     -> Export/convert artifacts -> Model registry -> BitNet-rs Server
     -> Batched inference (GPU/CPU) -> Receipts + Metrics + Logs
```

### Architectural principles
1. **Backward compatible by default**: existing BitNet workflows remain untouched unless explicitly opted into new paths.
2. **Format adapters over rewrites**: add adapters/loaders rather than replacing stable modules.
3. **Python for training, Rust for serving**: leverage ecosystem maturity while preserving BitNet-rs runtime strengths.
4. **Observable-first production posture**: every request path emits traces/metrics/logs.
5. **Incremental delivery**: production features behind flags and explicit acceptance criteria.

## 4) Implementation Workstreams

## WS-A: Model Format & Runtime Compatibility

### Deliverables
- Add `bitnet-models` support for:
  - SafeTensors full-precision tensor maps.
  - Expanded GGUF metadata mapping for non-BitNet transformer variants.
  - Optional ONNX backend adapter (feature gated).
- Unified model descriptor:
  - Architecture type, precision, kv-cache policy, tokenizer requirements.
- Converter extensions in `bitnet-st2gguf`:
  - Validate tensor naming contracts.
  - Emit compatibility report (missing/unmapped tensors).

### New modules (proposed)
- `crates/bitnet-models/src/loaders/safetensors_full.rs`
- `crates/bitnet-models/src/loaders/gguf_transformer.rs`
- `crates/bitnet-models/src/compat/model_descriptor.rs`
- `crates/bitnet-st2gguf/src/report.rs`

### Acceptance criteria
- Load GPT-2 style and LLaMA-style checkpoints from SafeTensors (smoke-level).
- Correctly fail with actionable diagnostics for unsupported tensor layouts.
- Legacy BitNet model loading tests remain green.

## WS-B: Tokenization and Vocabulary

### Deliverables
- Standardized tokenizer loader contract in `bitnet-tokenizers`:
  - JSON tokenizer files.
  - SentencePiece `.model` files.
- Tokenizer manifest object persisted with model artifacts.
- `xtask train-tokenizer` scaffold:
  - Input corpus path(s), vocab size, normalization mode, special tokens.
  - Output tokenizer + deterministic training metadata.

### New modules (proposed)
- `crates/bitnet-tokenizers/src/manifest.rs`
- `crates/bitnet-tokenizers/src/loader/universal.rs`
- `xtask/src/train_tokenizer.rs`

### Acceptance criteria
- Round-trip encode/decode tests for BPE + SentencePiece examples.
- Runtime mismatch check between model and tokenizer manifests.

## WS-C: Training & Fine-Tuning Pipeline (Python-first)

### Deliverables
- `training/` directory with:
  - `finetune.py` (HF Transformers + Accelerate/DeepSpeed configurable).
  - `evaluate.py` (perplexity + benchmark harness wrappers).
  - `export.py` (convert checkpoints to BitNet-rs consumable forms).
  - `configs/` with baseline DeepSpeed and LoRA/QLoRA profiles.
- Artifact contract between Python outputs and Rust loaders.
- Reproducibility package:
  - seeds, env capture, model card metadata, dataset hashes.

### Acceptance criteria
- One documented end-to-end fine-tune + export + serve flow on a reference model.
- Deterministic run metadata generated for every training job.

## WS-D: Inference Service Completion

### Deliverables
- Extend `bitnet-server` with inference API endpoints:
  - `POST /v1/generate`
  - `POST /v1/chat/completions` (optional compatible schema)
  - `GET /v1/models`
- Request validation:
  - max tokens bounds, prompt length limits, schema validation.
- Auth middleware (API key in phase 1).
- Streaming response mode (SSE or chunked text).
- Dynamic batching queue with configurable latency budget.

### New modules (proposed)
- `crates/bitnet-server/src/api/inference.rs`
- `crates/bitnet-server/src/api/models.rs`
- `crates/bitnet-server/src/middleware/auth.rs`
- `crates/bitnet-server/src/runtime/batching_queue.rs`

### Acceptance criteria
- Stable JSON contracts and integration tests.
- Batch throughput improvement demonstrated over single-request path.
- Health/readiness semantics include model-loaded status.

## WS-E: Performance, Quantization, and Memory Efficiency

### Deliverables
- Precision modes for inference: `fp16`, `bf16`, `int8` (phase 1), `int4` (phase 2).
- KV-cache management policy + telemetry.
- Micro-benchmark suite updates covering:
  - token latency,
  - tokens/s,
  - memory footprint,
  - quality deltas against fp16 baseline.

### Acceptance criteria
- Precision selectable via CLI/server config.
- Published benchmark table for at least one 7B-equivalent workload and one small model.

## WS-F: Security, Policy, and Honest Compute

### Deliverables
- Preserve and expose honest-compute receipts for API generation requests.
- Input/output policy hooks:
  - prompt validation,
  - optional content filters,
  - abuse controls (rate limits, token quotas).
- Security baseline:
  - secrets handling,
  - minimal container privilege,
  - dependency scanning in CI.

### Acceptance criteria
- Auth + rate limiting enforced in integration tests.
- Receipt ID returned per generation and retrievable for audits.

## WS-G: Observability, Reliability, and SRE

### Deliverables
- Prometheus metrics:
  - request count, error count, p50/p95/p99 latency,
  - queue depth, tokens/sec, model load time.
- OpenTelemetry traces:
  - tokenization, prefill, decode, postprocess spans.
- Structured JSON logging with request IDs.
- SLO definitions and alerts.

### Acceptance criteria
- Dashboards for latency/throughput/error budget in staging.
- Alert runbook and on-call checklist in docs.

## 5) Milestones and Timeline (18–24 weeks)

| Milestone | Duration | Output |
|---|---:|---|
| M0 Design freeze + contracts | 2 weeks | API/model/tokenizer contracts approved |
| M1 Model/tokenizer compatibility | 4 weeks | SafeTensors + tokenizer compatibility in place |
| M2 Inference API beta | 4 weeks | `/v1/generate` + auth + validation + tests |
| M3 Performance phase 1 | 4 weeks | fp16/bf16/int8 path + batch queue + metrics |
| M4 Training integration | 4 weeks | fine-tune/export/reload reference flow |
| M5 Hardening + canary rollout | 2–6 weeks | SLO-backed production deployment |

## 6) Ownership and Team Shape

Minimum team (2–3 engineers):
- **Infra/Serving engineer**: server API, batching, auth, observability.
- **ML systems engineer**: model loading, quantization, benchmark quality/perf.
- **MLOps engineer (or shared role)**: training pipeline, artifact contracts, CI/CD + deployment.

## 7) Detailed Backlog (Initial 12 tickets)

1. Add `ModelDescriptor` and compatibility report structs.
2. Implement SafeTensors full-precision loader with mapping diagnostics.
3. Add universal tokenizer loader + manifest.
4. Add tokenizer mismatch runtime guard.
5. Implement `/v1/generate` endpoint (non-streaming).
6. Add request schema validation + bounds.
7. Add API-key middleware.
8. Add batching queue abstraction and config knobs.
9. Add Prometheus counters/histograms for inference path.
10. Add OpenTelemetry spans around generation stages.
11. Add training scaffold (`training/finetune.py`, `export.py`, `configs/`).
12. Add end-to-end smoke CI: train(or load)-export-serve-infer.

## 8) CI/CD Additions

- New required checks:
  - `cargo test -p bitnet-server`
  - `cargo test -p bitnet-tokenizers`
  - contract tests for loader/tokenizer compatibility.
- Nightly performance job:
  - run microbenchmarks and compare against threshold regressions.
- Security gate:
  - dependency audit + container scan + secret scan.

## 9) Benchmark and Evaluation Matrix

### Quality
- Perplexity: WikiText-103/LAMBADA.
- Reasoning/knowledge: MMLU, ARC, HellaSwag.
- Code generation: HumanEval.

### Performance
- Single-request latency (prefill + decode).
- Multi-tenant throughput under concurrent load.
- Memory footprint by precision mode.

### Release gate recommendation
- No more than 1% absolute quality regression from fp16 baseline for approved quantization modes.
- p95 latency and error-rate SLOs met for two consecutive canary windows.

## 10) Backward Compatibility Strategy

- Keep current BitNet code path as default.
- Introduce new behavior via explicit configuration:
  - model family (`bitnet`, `transformer`),
  - precision mode,
  - backend selection.
- Add compatibility tests that run both legacy and new paths in CI.
- Document migration with examples and expected pitfalls.

## 11) Deployment Plan

1. Containerize server with model volume mounts and immutable image tags.
2. Deploy to staging on GPU node pool.
3. Run synthetic + benchmark workloads.
4. Canary production rollout (5% -> 25% -> 100%).
5. Auto-rollback on SLO breaches.

## 12) Risks and Mitigations

- **Model format fragmentation** -> strict compatibility report + converter tests.
- **GPU memory pressure** -> batching budgets, kv-cache policies, quantization fallback.
- **Training reproducibility drift** -> seed + dataset hash + config capture.
- **Security regressions** -> auth/rate limits first, policy hooks before public exposure.
- **Scope creep** -> milestone gates with frozen acceptance criteria.

## 13) Definition of Done (Platform v1)

BitNet-rs is considered “general-LLM production-ready v1” when:
- A reference mainstream model can be fine-tuned, exported, loaded, and served end-to-end.
- The server exposes stable inference APIs with auth, validation, batching, and observability.
- Performance/quality gates are measured and published.
- Legacy BitNet workflows remain fully functional and tested.
