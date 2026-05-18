# BITNET-SPEC-APPLE-QUALITY-CORPUS

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: [Apple M4 dense SLM appliance](BITNET-SPEC-APPLE-M4-DENSE-SLM-APPLIANCE.md), [Apple M4 BitNet CPU/NEON](BITNET-SPEC-APPLE-M4-BITNET-CPU-NEON.md)
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; corpus contract only
Policy impact: no policy exception

## Purpose

Unify Apple quality-corpus rules while keeping dense SLM and BitNet evidence
separate. Corpus receipts report mechanical quality envelopes for exact model,
tokenizer, prompt, backend, fallback, machine, and corpus identities; they do
not make broad quality claims.

## Dense SLM corpus

Dense SLM evidence uses:

- 500 deterministic cases where promoted;
- mechanical scoring only;
- task-family pass rates;
- failure taxonomy;
- matching-history refresh;
- no broad quality claim.

Dense Qwen evidence remains in the dense SLM proof family and cannot be reused
as BitNet proof.

## BitNet corpus

BitNet evidence uses:

- 100 to 250 to optional 500-case decision progression;
- BitNet-specific tasks;
- reference-vs-Rust deltas;
- task-family taxonomy;
- evidence separated from dense Qwen evidence.

## Mechanical scoring modes

Apple quality corpora may use these mechanical checks:

- exact match;
- normalized match;
- numeric tolerance;
- JSON/schema validation;
- required keywords;
- forbidden tokens;
- closed-label classification;
- stop/special-token checks;
- timeout/error/fallback taxonomy.

LLM judging is not required for these corpora. If future qualitative review is
added, it must be labeled advisory and cannot replace the mechanical corpus
contract.
