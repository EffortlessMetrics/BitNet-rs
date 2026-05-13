# BITNET-PROP-0001: Proof Convergence and CI Economics

Status: proposed
Owner: release/ci
Type: proposal

## Problem

BitNet-rs has many real proof surfaces: model artifact manifests, tokenizer
authority, prompt-template checks, hardware receipts, backend diagnostics, CI
policy ledgers, campaign trackers, and generated reports. Those surfaces are
useful, but they are easy for humans and agents to misread if they are treated
as interchangeable.

The repo is still pre-alpha. It supports dense SLM work as a first-class local
inference path where the relevant artifact, tokenizer, backend, and receipt
gates pass, while coherent Rust BitNet answer quality is still under strict
validation. A structurally valid GGUF is not automatically answer-ready. A CUDA
or Metal receipt is not automatically coherent answer proof. A dense SLM answer
receipt is not BitNet or 1-bit proof.

At the same time, BitNet-rs has enough CI, model, hardware, and receipt
surfaces that verification cost is now part of correctness. If ordinary PRs
spend expensive lanes on unrelated work, maintainers and agents get slower
feedback without better evidence. If CI is cheap but weak, claims become easy to
overstate. The repo needs one operating model that keeps product claims,
campaign work, CI economics, and proof artifacts aligned.

## Proposal

Create a proof-convergence lane that turns BitNet-rs into a proof-first local
inference repo operating system:

```text
model artifact proof
+ tokenizer and prompt authority
+ backend and hardware receipt
+ answer corpus or diagnostic corpus
+ CI economics and risk routing
+ campaign source of truth
= honest user-facing capability claims
```

The lane does not add a new hidden goal system. BitNet-rs already has the right
execution authority in campaign-local tracking:

```text
docs/tracking/campaigns/<campaign>/CAMPAIGN.md
docs/tracking/campaigns/<campaign>/active.toml
docs/tracking/campaigns/<campaign>/events/
docs/tracking/campaigns/<campaign>/generated/
```

The lane defines how existing proof surfaces fit together:

- Proposals explain why a long-lived effort exists.
- Specs define what must be true before a behavior or claim is accepted.
- ADRs record durable architecture and proof decisions.
- Plans define PR order, proof commands, non-goals, and rollback.
- Campaign `active.toml` files define current executable work.
- Status documents map user-facing claims to proof commands and artifacts.
- Policy TOMLs define enforceable CI, exception, allowlist, and routing ledgers.
- Receipts and artifacts prove what actually happened.
- README and roadmap summarize; they do not replace the proof map.

## Source-Of-Truth Links

This proposal uses current BitNet-rs authorities rather than introducing a
parallel tracker:

- [README](../../README.md) for high-level user positioning and current
  pre-alpha claim boundaries.
- [ROADMAP](../../ROADMAP.md) for release direction, current limitations, and
  milestone planning.
- [Answer Artifact Gate](../model-artifacts/ANSWER_ARTIFACT_GATE.md) for model
  answer-readiness, tokenizer authority, prompt-template authority, reference
  runner proof, and artifact identity.
- [Hardware Matrix](../hardware/HARDWARE_MATRIX.md) for hardware lane identity,
  runtime identity, selected backend identity, fallback behavior, proof stage,
  and allowed claims.
- [CI Cost and Verification Policy](../ci/cost-and-verification-policy.md) for
  ordinary PR economics, risk-routed expensive lanes, and ripr's role as static
  mutation-exposure analysis.
- [Tracker Model](../tracking/TRACKER_MODEL.md) for campaign-local execution
  state, work item fields, lifecycle events, generated dashboards, and merge
  policy.
- [Status Documents](../status/README.md) for the planned user-facing claim map.

## Goals

- Give users, maintainers, and agents one durable way to understand which model
  families are usable, diagnostic, experimental, or unsupported.
- Make every user-facing capability claim traceable to the model artifact,
  tokenizer authority, backend receipt, hardware lane, CI lane, or campaign
  item that justifies it.
- Preserve dense SLM support as a first-class product path without letting dense
  SLM proof become BitNet or 1-bit proof.
- Preserve BitNet I2_S and QK256 proof as separate model-family work with its
  own artifact, tokenizer, quantization, backend, and answer-readiness gates.
- Keep hardware receipts useful without letting selected-device execution imply
  answer quality or speed qualification.
- Keep ordinary PR CI cheap, deterministic, and high-signal while routing
  mutation, coverage, hardware, model downloads, GPU, macOS, Windows, and
  performance lanes to risk surfaces that justify them.
- Teach agents to use campaign-local `active.toml` and events instead of chat
  logs, README prose, or another global hand-edited tracker.
- Prepare the next minor release to state only claims backed by receipts,
  policy ledgers, and proof commands.

## Non-Goals

- Do not create `.adze/goals`, `.bitnet/goals`, or another hidden global goal
  manifest.
- Do not move model artifacts, receipts, hardware manifests, CI workflows, or
  generated dashboards as part of this proposal.
- Do not duplicate the full answer artifact gate inside specs or status pages.
- Do not duplicate the full hardware matrix inside specs or status pages.
- Do not duplicate CI lane tables from policy TOMLs inside narrative docs.
- Do not promote diagnostic BitNet output to coherent local-answer support.
- Do not promote dense SLM proof to BitNet, QK256, I2_S, 1-bit, kernel, or
  speed proof.
- Do not make macOS, Windows, GPU, model-validation, coverage, or full mutation
  lanes default ordinary-PR requirements.

## Claim Boundaries

The proof-convergence lane should make these rules hard to miss:

- Structural model validity proves bytes can be parsed; it does not prove answer
  readiness.
- Tokenizer metadata checks prove authority or gaps; they do not prove coherent
  answers on their own.
- Prompt-template authority is a precondition for answer claims; it is not a
  backend proof.
- Hardware receipts prove lane-specific execution metadata; they do not prove
  answer quality unless the shared answer gate and lane-specific proof pass.
- CUDA, Metal, OpenCL, OpenVINO, NPU, WGPU, AVX2, AVX-512, and NEON claims must
  keep requested backend, selected backend, runtime, hardware identity, and
  fallback status separate.
- Dense SLM local-answer evidence is valuable product evidence for dense SLMs;
  it is not BitNet or 1-bit evidence.
- Speed claims require profile-qualified evidence; successful output alone is
  not a throughput claim.
- Skipped expensive lanes must be reported as skipped-by-policy, not hidden as
  passed proof.

## CI Economics

CI cost control is part of the proof model. BitNet-rs should verify more, not
less, by spending expensive checks where they buy relevant signal.

Default PRs should prefer cheap, deterministic proof:

```text
formatting
compile checks
lint checks
crate/risk-scoped tests
policy checks
file and process guardrails
ripr static mutation-exposure signal
```

Risk-routed lanes should add expensive proof when the changed surface warrants
it:

```text
targeted mutation
coverage
GPU or hardware receipts
model validation
crossval
macOS or Windows
performance qualification
release readiness
```

The corrected ripr doctrine matters here: ripr is static mutation-exposure
analysis. It catches much of the same weak-test and weak-oracle signal that
mutation testing catches, but earlier and cheaper because it runs statically
and can run per PR. Mutation testing remains the runtime empirical backstop for
risk PRs, nightly, and release readiness.

## Success Criteria

This lane succeeds when BitNet-rs has:

- A capability matrix that maps each model family and hardware lane to a tier,
  proof command, proof artifact, claim allowed, and claim not allowed.
- Specs that define source-of-truth boundaries, default PR CI economics, answer
  artifact gates, hardware proof stages, dense SLM versus BitNet claims, and
  receipt contracts.
- ADRs that lock the key proof decisions before runtime and release work depend
  on them.
- A proof-convergence plan with PR-sized work items, proof commands, non-goals,
  and rollback paths.
- A campaign-local `active.toml` for proof convergence that agents can use as
  the executable state of the lane.
- Policy ledgers or checks that make the source-of-truth model machine-readable
  once the docs have stabilized.
- Release notes and README summaries that only claim capabilities backed by the
  relevant proof surfaces.

## Exit Criteria

The proposal can be closed when:

- The planned proposal, specs, ADRs, status docs, plan files, campaign manifest,
  and policy ledger exist.
- The docs source-of-truth check exists in advisory/reporting form.
- The next minor release prep can link each user-visible claim to its proof
  artifact, command, CI lane, or explicit unsupported/diagnostic boundary.

## Rollback

Rollback is documentation-only:

- Remove or revert the proposal, specs, ADRs, plan files, status docs, campaign
  manifest, or policy ledger added by this lane.
- Leave runtime code, CI behavior, model artifacts, hardware receipts, and
  generated dashboards unchanged unless a later implementation PR explicitly
  changed them.
- If a source-of-truth check becomes too noisy, keep it advisory or remove it
  from reporting until the dataset and conventions stabilize.
