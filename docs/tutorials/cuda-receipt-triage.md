# CUDA Receipt Triage

Use this guide when a 9950X3D + RTX 5070 Ti CUDA run produces a surprising
answer, a rejected claim, or a support issue. The goal is to turn a receipt into
an actionable diagnosis without broadening the claim beyond what the receipt
proves.

Start from:

```powershell
bitnet model status --device nvidia-rtx-5070-ti-cuda --format json
bitnet receipts explain --latest --format json
```

For issue reports, prefer the single support bundle. It includes model status,
the latest receipt explanation, route/backend/fallback summary, quality gate,
server-readiness scope, proof-family booleans, binary identity, and runtime
identity when the receipt exposes it:

```powershell
bitnet support bundle --latest --device nvidia-rtx-5070-ti-cuda --format json
```

Or explain a specific receipt:

```powershell
bitnet receipts explain <path-to-receipt.json> --format json
```

The model status command tells you what the repo currently allows each model
row to claim. The receipt tells you what the last command actually did.

## What To Paste In An Issue

Paste the `support bundle` JSON when available. If bundle creation fails, paste
the `receipts explain` summary plus:

```text
model_coverage_row
current_tier
model id or artifact SHA
requested backend
selected_backend
runtime API
selected_route
fallback_used
quality gate result
speedup_claim
server_ready
server_scope
server_endpoint
server_streaming
server_smoke
server_reason
full_residency_claim
bitnet_packed_i2s_qk256_proof
dense_regular_llm_cuda_proof
receipt path
claim boundary / not allowed claims
```

Do not paste private paths, tokens, credentials, or model files.

## Server Response Metadata

For `/v1/chat/completions`, the response metadata links the response to the
same receipt that is embedded in the response body:

```json
{
  "metadata": {
    "receipt_id": "uuid",
    "receipt_path": "/receipts/uuid",
    "latest_receipt_path": "/receipts/latest",
    "readiness_path": "/readiness",
    "model_coverage_row": "dense_qwen25_05b_q8_cuda",
    "model_coverage_tier": "product_cli_ready",
    "selected_backend": "nvidia-rtx-5070-ti-cuda",
    "selected_route": "dense_regular_llm_cuda",
    "fallback_used": false
  }
}
```

Use these server endpoints for support triage:

- `GET /receipts/latest`: latest retained server shared-engine receipt.
- `GET /receipts/{receipt_id}`: retained receipt by response metadata id.
- `GET /readiness`: readiness and claim-boundary state.
- `GET /v1/readiness`: versioned readiness alias.
- `GET /v1/models`: loaded model inventory.

## If `fallback_used=true`

The run is not strict RTX 5070 Ti CUDA proof. Treat it as a rejected or
diagnostic receipt until the fallback reason is fixed.

Check:

- whether the command requested `nvidia-rtx-5070-ti-cuda`;
- whether strict CUDA mode was enabled for the path;
- which operation or route caused fallback;
- whether unsupported strict ops were nonzero;
- whether CPU fallback counts were recorded.

Allowed claim: diagnostic failure evidence.
Not allowed: CUDA answer readiness, CUDA speed, BitNet QK256 proof, or dense
CUDA proof for that run.

## If The Selected Backend Is Generic `cuda`

Generic `cuda` is not enough for selected-device proof. A strict CUDA claim must
resolve to:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
```

If the receipt keeps only a generic CUDA label, file the issue as backend
identity ambiguity. Include the requested backend, selected backend, device
probe details, and receipt path.

## If Tokenizer Authority Is Missing

Artifact loading is not answer readiness. A receipt that lacks tokenizer or
pre-tokenizer authority cannot support a coherent answer claim.

Check:

- the model coverage row;
- tokenizer authority field;
- prompt-template authority field;
- answer artifact gate status;
- whether an external tokenizer was supplied and recorded.

Allowed claim: structural or diagnostic evidence.
Not allowed: coherent BitNet answer, dense SLM answer, benchmark speed based on
generated text, or server readiness.

## If The Prompt Template Is Wrong

A valid model and tokenizer can still fail answer quality when the prompt
template or stop policy is wrong.

Check:

- prompt authority in the model coverage row;
- rendered prompt or template family when recorded;
- stop-token policy;
- prompt suite or corpus case that failed;
- first generated token or first divergence.

Do not promote model quality from a receipt that used an unapproved prompt
template.

## If The Quality Gate Failed

Quality-gate failure is useful proof, but it is failure proof. Keep the claim
diagnostic until a passing artifact or backend receipt exists.

Check:

- failed prompt id;
- expected answer class;
- generated text and token ids;
- first divergence if recorded;
- whether CPU and CUDA both failed or only one backend failed;
- whether the artifact is answer-ready for this model family.

If CPU passes and CUDA fails, route the issue to backend/kernel/parity triage. If
both fail, route it to artifact, tokenizer, prompt, or model-quality triage.

## If `speedup_claim=false`

This is usually the correct state. CUDA execution proof and answer quality do
not imply speedup.

Speed can be claimed only when a governed benchmark qualification receipt
accepts the exact model/profile. Check:

- profile name;
- CPU and CUDA p50/p95/mean;
- first-token latency;
- steady decode timing;
- kernel time;
- H2D and D2H timing sources;
- VRAM high-water mark;
- power and thermal context when available;
- accepted or rejected speedup decision and reason.

Do not turn a benchmark baseline, one fast run, or a report-only receipt read
into a global CUDA speed claim.

## If `server_ready=false`

CLI readiness is not server readiness. A bounded server-smoke receipt is useful,
but it does not automatically promote broad server support.

Check:

- server route;
- selected backend;
- fallback status;
- model coverage row;
- `server_smoke`, `server_scope`, `server_endpoint`, and `server_streaming`
  from `bitnet model status --format json`;
- response quality;
- receipt emission path;
- whether the row was explicitly promoted by an exact-profile readiness gate.

Allowed claim: bounded server-smoke evidence when a receipt exists.
Not allowed: production server readiness, global dense server readiness, or
BitNet server readiness unless the exact route has its own receipt and row.

For the RTX 5070 Ti lane, Qwen2.5 exact-profile server readiness should show
`server_ready=true`, `server_scope=exact_profile`, endpoint
`/v1/chat/completions`, and `server_streaming=false`. Official BitNet QK256
server smoke should show `server_smoke=true`, `server_ready=false`, and
`server_reason=broad production readiness not qualified`.

## If Dense Proof Is Mistaken For BitNet Proof

Keep route families separate:

```text
bitnet_qk256_cuda       -> official BitNet packed I2_S/QK256 only
dense_regular_llm_cuda  -> dense SLM / small dense LLM only
```

Dense Qwen2.5, Qwen3, SmolLM2, Llama, Gemma, and Phi receipts never prove
BitNet packed I2_S/QK256 behavior. Official BitNet QK256 receipts never prove
dense SLM behavior.

If the issue mixes these claims, paste the route and model coverage row first.
The route usually determines whether the issue belongs to BitNet QK256 CUDA,
dense regular-LLM CUDA, CPU reference, server, or benchmark triage.

## If Qwen2.5 Proof Is Mistaken For Qwen3 Proof

Qwen2.5 and Qwen3 are separate model coverage rows. A Qwen2.5 receipt can prove
only the Qwen2.5 artifact/profile it names. It does not prove Qwen3 artifact
identity, tokenizer/prompt authority, user-path ask/chat behavior, server
readiness, speedup, or residency.

Check:

- `model_coverage_row`;
- model id and artifact checksum;
- selected route;
- prompt and tokenizer authority;
- whether the receipt path names Qwen2.5 or Qwen3;
- whether the model coverage row already earned the claim being discussed.

Allowed claim: the exact model/profile named by the receipt.
Not allowed: inheriting Qwen2.5 server readiness, speed, or quality proof for
Qwen3.

## If The Receipt Is Missing

Re-run with an explicit receipt output path when the command supports it:

```powershell
bitnet receipts explain --latest --format json
```

If `--latest` cannot find the expected file, include:

- command run;
- working directory;
- expected receipt directory;
- whether the command failed before receipt creation;
- console summary if available.

No receipt means no durable proof. Keep the issue about reproducibility or
receipt emission until a receipt exists.

## Stop Lines

Do not use a receipt to claim:

- generic `cuda` as strict RTX 5070 Ti proof;
- CPU fallback as CUDA proof;
- WGPU, Vulkan, or D3D12 as CUDA proof;
- dense SLM proof as BitNet QK256 proof;
- BitNet QK256 proof as dense SLM proof;
- answer quality as speedup;
- CLI readiness as server readiness;
- one bounded server smoke as production readiness;
- crates.io or docs.rs publication.

When in doubt, demote the claim and keep the receipt as diagnostic evidence.
