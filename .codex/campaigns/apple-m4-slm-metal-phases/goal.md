# apple-m4-slm-metal-phases

Goal: expand Apple M4 dense SLM Metal participation through named,
phase-scoped, parity-gated prefill/projection phases while keeping CPU/NEON as
the default full-pipeline route and avoiding any full `apple-m4-metal`
inference claim until a later strict receipt proves it.

Every phase must record CPU parity, Metal `fallback_used=false`, explicit CPU
routing for remaining phases, timing deltas, and claim boundaries.
