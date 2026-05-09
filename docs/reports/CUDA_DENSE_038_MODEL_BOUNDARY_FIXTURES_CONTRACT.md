# CUDA-DENSE-038 Model-Boundary Fixtures Contract

`CUDA-DENSE-038` is the next dense CUDA proof slice after
`CUDA-DENSE-037` made the Qwen-family transformer-block plan inspectable across
all layers.

The next implementation should cover model-boundary fixtures that are outside
the transformer block plan:

```text
token embedding lookup
final model norm
LM head / output projection
logits vector shape
logits hash / top-k diagnostics
```

## Claim Boundary

The future receipt may claim:

```text
dense_gguf_model_boundary_fixtures recorded
token embedding fixture exists
final norm fixture exists
lm_head/logits fixture exists
fallback_used=false for the fixture route
```

It must not claim:

```text
Qwen one-token CUDA proof
Qwen short decode or chat
general dense GGUF inference
speedup
persistent or full CUDA residency
KV cache policy
sampling integration
BitNet packed I2_S / QK256 proof
```

## Remaining Blockers After CUDA-DENSE-038

Even after model-boundary fixtures land, dense CUDA still needs:

```text
KV cache policy and receipt
sampling/logit selection boundary
one-token strict CUDA proof
short decode proof
warm-session receipts
benchmark qualification
```
