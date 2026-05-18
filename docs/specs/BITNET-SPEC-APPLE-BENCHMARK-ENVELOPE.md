# BITNET-SPEC-APPLE-BENCHMARK-ENVELOPE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: [Apple reproducible run identity](BITNET-SPEC-APPLE-REPRODUCIBLE-RUN-IDENTITY.md)
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; benchmark contract only
Policy impact: no policy exception

## Purpose

Define Apple benchmark envelopes that are useful to operators without
claiming broad Apple Silicon performance from one machine, backend, or profile.

## Required metrics

Benchmark receipts and summaries must record, where supported by the profile:

- cold load;
- tokenizer load;
- prompt render;
- prompt tokenization;
- prefill/input tok/s;
- TTFT;
- first streamed chunk;
- decode/output tok/s;
- sampling overhead;
- total wall time;
- peak memory;
- memory drift;
- disk/cache context;
- thermal pressure where available;
- power state where available;
- p50/p90/p99/min/max;
- repeat count;
- outlier policy;
- matching identity context.

## Required profiles

Apple benchmark profiles include:

- `short_prompt_16_out`;
- `short_prompt_64_out`;
- `long_prompt_16_out`;
- `long_prompt_128_out`;
- `context_1k`;
- `context_4k`;
- `resident_25`;
- `resident_50`;
- `resident_100`;
- `mixed_model_switch`;
- `bitnet_one_shot`;
- `bitnet_warm`.

## Envelope rules

- Every benchmark comparison must include reproducible run identity.
- Dense SLM and BitNet benchmark envelopes remain separate.
- Profiles with different model, tokenizer, prompt, backend, fallback, machine,
  OS, binary/build, corpus/profile, or seed identity are not matching-history
  comparisons.
- No broad Apple Silicon benchmark claim may be made from one M4 Mac Mini
  profile.
- Live hardware/model timing is not required in ordinary generic PR CI.
