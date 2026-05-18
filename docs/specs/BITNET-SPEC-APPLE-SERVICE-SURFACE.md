# BITNET-SPEC-APPLE-SERVICE-SURFACE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: [Apple Silicon route contract](BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md), [Apple reproducible run identity](BITNET-SPEC-APPLE-REPRODUCIBLE-RUN-IDENTITY.md)
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no service promotion; readiness contract only
Policy impact: no policy exception

## Purpose

Define Mac ask/chat/serve readiness for exact Apple profiles without implying
production hosting, broad Apple Silicon support, unsupported BitNet service
status, or unproven acceleration.

## Surfaces

Apple service-surface work covers:

- `bitnet mac doctor`;
- `bitnet mac evidence`;
- `bitnet mac ask`;
- `bitnet mac chat`;
- `bitnet mac serve`;
- `bitnet mac receipts-check`;
- `bitnet mac regression`;
- `bitnet mac report-refresh`;
- `bitnet mac benchmark`.

## Serve readiness

Serve readiness for an exact profile requires:

- health endpoint;
- ready endpoint;
- streaming semantics;
- timeout behavior;
- cancellation behavior;
- per-request receipt;
- `fallback_used = false` for backend proof routes;
- model/tokenizer/backend identity;
- exact-profile scope.

## Status rules

- A surface may be enabled for dense SLM without being enabled for BitNet.
- BitNet chat or serve cannot be claimed until the relevant campaign gate has
  matching receipts.
- Service readiness is exact-profile readiness, not broad production hosting.
- Operator output must expose model, tokenizer, requested backend, selected
  backend, runtime API, fallback, machine, and proof-family status.
