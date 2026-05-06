# BitNet Runtime Phases

## Purpose

Every BitNet proof or benchmark must name the runtime phase it covers. A kernel smoke, prefill run, first-token run, and steady decode benchmark prove different things.

## Phase Contract

```yaml
runtime_phases:
  load_model:
    includes:
      - file open
      - GGUF parse
      - metadata validation
      - tensor map
      - loader mode

  tokenize_prompt:
    includes:
      - tokenizer source
      - prompt template
      - prompt token count

  prefill:
    includes:
      - full prompt forward
      - RoPE positions
      - KV cache creation

  first_token:
    includes:
      - first decode step
      - logits
      - sampling or greedy selection

  decode_steady_state:
    includes:
      - repeated single-token decode
      - KV cache reuse
      - tokens_per_second

  sampling:
    includes:
      - temperature
      - top_k
      - top_p
      - seed
      - greedy flag

  total_generation:
    includes:
      - full user-visible generation path
      - all available phase timings
```

## Claim Rules

- A prefill benchmark is not a decode benchmark.
- A first-token result is not steady-state throughput.
- A kernel smoke test is not model inference.
- A full inference receipt must include at least load, tokenize, prefill, first token, decode, and sampling fields where available.
- Missing phase labels invalidate performance claims.

## Receipt Fields

```json
{
  "execution": {
    "phase": "load_model|tokenize_prompt|prefill|first_token|decode_steady_state|sampling|total_generation|full",
    "prompt_tokens": 0,
    "generated_tokens": 0,
    "batch_size": 1
  }
}
```

## Related Docs

- `docs/bitnet/BITNET_BENCHMARK_PROTOCOL.md`
- `docs/bitnet/BITNET_RECEIPT_FIELDS.md`
