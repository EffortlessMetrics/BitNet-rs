# Apple M4 Operator Workload Suite

`M4-WORKLOAD-001` starts with a model-free workload-suite contract:

```bash
bitnet mac workload --suite m4-operator --json-out target/apple-m4-inference-excellence/workload/summary.json
bitnet mac receipts-check target/apple-m4-inference-excellence/workload/summary.json --json
```

The receipt artifact is `apple_m4_operator_workload_suite`. It enumerates the
operator workflows, enabled route surfaces, route-state boundaries, required
mechanical checks, and receipt obligations that later live M4 workload runs must
fill in.

This command is generic-PR-safe. It does not fetch models, run generation, start
the local server, or enable BitNet chat or serve. Live workload receipts belong
under `ci/hardware/apple-m4-mac-mini/**/workload/**` after the suite contract is
merged and run on the M4 Mac mini.

## Workflows

| Workflow | Task family | Mechanical checks |
|---|---|---|
| `summarize` | constrained summary | required keywords, forbidden-token checks |
| `extract` | extraction | JSON/schema validation, required keywords |
| `classify` | classification | exact match, forbidden-token checks |
| `json_schema` | JSON/schema output | JSON/schema validation, numeric tolerance, forbidden-token checks |
| `rewrite` | rewrite | normalized match, required keywords, forbidden-token checks |
| `table_qa` | fixed-table QA | exact match, numeric tolerance |

## Route Scope

The suite covers these enabled routes from the route-state matrix:

| Route | Class | Runtime proof source |
|---|---|---|
| `dense_slm.ask` | interactive or advisory by selected model | dense ask/local-answer receipts |
| `dense_slm.chat` | interactive or advisory by selected model | dense chat receipts |
| `dense_slm.warm_session` | interactive or advisory by selected model | dense warm-session receipts |
| `dense_slm.serve` | advisory | dense local-server receipts |
| `bitnet.ask` | batch | accepted BitNet one-shot receipts |
| `bitnet.warm_session` | batch | accepted BitNet warm-session receipts |

BitNet chat, serve, and streaming stay disabled unless their ready gate receipts
exist. Dense SLM evidence does not prove BitNet behavior, and BitNet evidence
does not prove dense SLM behavior.

## Receipt Obligations

Each future live workload case must preserve:

- model identity and tokenizer authority
- requested and selected backend
- `fallback_used=false`
- prompt hash
- output text and token IDs
- timing, throughput, and memory fields
- explicit claim boundaries

The suite is a contract for later live receipts, not a broad model-quality,
production-service, Metal, QK256, Neural Engine, MPSGraph, MacBook, speedup, or
broad Apple Silicon performance claim.
