# Swarm Runner Rollout Plan Across Repositories

Status: Draft
Owner: CI/Swarm Infrastructure
Created: 2026-05-21
Linked proposal: n/a
Linked specs: n/a
Linked ADRs: n/a
Linked plan: docs/development/swarm-runner-rollout-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: n/a
Policy impact: CI routing and guardrail policy updates per repo

Authority note: this is a rollout planning document. It does not supersede a
repository's proposal, spec, ADR, active goal, campaign tracker, PR template, or
cutover/sync policy. Each repository must land its own source-of-truth update
before treating this plan as implementation authority.

## Reference implementation

Use **HL7v2 PR #73** as the reference pattern:

- Correct runner discovery surface: `gh api "orgs/${ORG}/actions/runners?per_page=100"`
- Preserved routing chain: `CPX42 -> CX43 -> CX53 -> GitHub-hosted`
- Proof shape: route job succeeds, selected lane succeeds, fallback lanes skip, normalized result passes.

Operational lessons to preserve:

1. Repository runner endpoint was the wrong discovery surface.
2. CPX42 should **not** assume local `em-ci-rust:1.95` Docker image; use direct Rust 1.95 toolchain on host.
3. CX43/CX53 keep the known-good Docker image path where already working.

## Standard swarm contract

For each migrated repo:

- Before a documented cutover, source repo `EffortlessMetrics/<repo>` remains
  the source-of-truth repository and swarm changes are CI/workflow candidates
  that still need an explicit promotion or sync path.
- After a documented cutover, swarm repo `EffortlessMetrics/<repo>-swarm` is
  the active development authority. The source repo is limited to release,
  publish, emergency hotfix, final source-of-truth correction, or explicit
  promotion/sync work as recorded by that repo's cutover docs.
- Swarm repo `EffortlessMetrics/<repo>-swarm` is public, trusted same-repo
  swarm CI/work.

Self-hosted routing permissions:

- Same-repo PR: allowed
- `workflow_dispatch`: allowed
- `merge_group`: allowed
- Fork PR: never self-hosted (hosted-only or guarded skip)

First-pass exclusions:

- release, publish, signing, tagging, deployment
- registry uploads, TestPyPI/PyPI, npm
- GPU/model/cache-heavy lanes
- full platform matrices

Required check policy:

- One normalized required check per repo: `<Repo> Rust Small Result`
- Never require conditional implementation jobs directly.

## Runner discovery contract

Every routed workflow must use organization runner discovery:

```bash
gh api "orgs/${ORG}/actions/runners?per_page=100"
```

with:

```yaml
env:
  GH_TOKEN: ${{ secrets.EM_RUNNER_READ_TOKEN }}
  ORG: EffortlessMetrics
```

Prohibited endpoint:

```text
repos/<owner>/<repo>/actions/runners
```

Stable router outputs:

- `router_target=cpx42|cx43|cx53|github`
- `router_reason=cpx42_idle|cx43_idle|cx53_idle|no_idle_runner|runner_token_missing|runner_token_unauthorized|runner_token_forbidden|runner_api_failed|parse_failed|fork_pr`
- `router_error=false|true`

## Execution model by runner

### CPX42

- Use direct Rust toolchain install/use (`1.95.0`)
- No local `em-ci-rust` image assumption
- Prepare scratch paths before rust-toolchain action
- Run repo small gate directly on host
- Cleanup scratch paths at end

Required CPX42 labels:

- `self-hosted`, `linux`, `x64`, `em-ci`, `cpx42`, `rust-16gb`, `rust-medium`, `trusted-pr`

### CX43/CX53

- Preserve existing Docker-image flow where already working
- Keep `em-ci-rust:1.95` pattern
- Do not rewrite working jobs to host-toolchain unless lane-specific failure requires it

## Route templates

### Medium Rust repos

Route: `CPX42 -> CX43 -> CX53 -> GitHub-hosted`

Candidates:

- `hl7v2-rs-swarm`
- `tokmd-swarm`
- `OpenRacing-swarm`
- `unsafe-review-swarm`
- `ripr-swarm`
- `uselesskey-swarm`
- `perfgate-swarm`
- `shiplog-swarm`
- `shipper-swarm`
- `atlasctl-swarm`

### Heavy Rust repos

Route: `CX53 -> CX43 -> GitHub-hosted`

Candidates:

- `bitnet-rs-swarm` current Rust Small lane
- `perl-lsp-swarm` full/corpus lane
- model/cache-heavy lanes

### Split-lane repos

- Tiny lane: `CPX42 -> CX43 -> GitHub-hosted`
- Small/heavy lane: `CX53 -> CX43 -> GitHub-hosted`

Candidates:

- `perl-lsp-swarm`
- `bitnet-rs-swarm` (after true Tiny lane exists)

## Rollout order

1. **Batch 1 (medium pilots)**
   - `tokmd-swarm`
   - `OpenRacing-swarm`
2. **Batch 2 (higher-value medium/heavy)**
   - `perl-lsp-swarm` (split lanes)
   - `adze-swarm` (lane-size-aware route)
3. **Batch 3 (BitNet special case)**
   - keep current heavy lane on `CX53 -> CX43 -> GitHub-hosted`
   - do not add CPX42 to current BitNet Rust Small lane
4. **Batch 4 (long tail standardization)**
   - `uselesskey-swarm`, `ripr-swarm`, `unsafe-review-swarm`, `perfgate-swarm`, `shiplog-swarm`, `shipper-swarm`, `atlasctl-swarm`, optionally `flow-studio-swarm`

## Per-repo checklist

### Admin setup

- [ ] swarm repo public
- [ ] repo selected in `em-ci-small`
- [ ] `EM_RUNNER_READ_TOKEN` selected-repo access includes swarm repo
- [ ] source repo remains canonical release mirror
- [ ] branch protection deferred until proof complete

### Workflow patch

- [ ] org runner discovery used
- [ ] same-repo trust guard present
- [ ] fork PR blocked from self-hosted
- [ ] CPX42 route added for medium/tiny
- [ ] CPX42 host-toolchain model implemented
- [ ] CPX42 scratch prepared before toolchain action
- [ ] CX43/CX53 paths preserved
- [ ] hosted fallback preserved
- [ ] normalized result aggregates all conditional jobs

### Guardrails

- [ ] policy/xtask rejects repository runner endpoint
- [ ] policy/xtask requires org runner endpoint
- [ ] policy/xtask verifies CPX42 labels
- [ ] policy/xtask verifies scratch-before-toolchain
- [ ] policy/xtask verifies Rust 1.95 toolchain usage
- [ ] policy/xtask verifies CX43/CX53 jobs exist
- [ ] policy/xtask verifies hosted fallback exists
- [ ] docs enforce only normalized result as required check

### Pre-merge proof

- [ ] `router_target=cpx42` (for medium lanes)
- [ ] `router_reason=cpx42_idle`
- [ ] selected lane succeeds
- [ ] fallback implementation lanes skip
- [ ] normalized result succeeds

### Post-merge proof

- [ ] `workflow_dispatch` on `main` succeeds
- [ ] forced CX43 fallback succeeds
- [ ] forced CX53 fallback succeeds (if included)
- [ ] hosted fallback succeeds
- [ ] only then enable branch protection requiring normalized result

## BitNet-specific constraint

For `bitnet-rs-swarm` current lane:

- Keep route as `CX53 -> CX43 -> GitHub-hosted`
- Do **not** add CPX42 to current BitNet Rust Small lane
- Defer CPX42 until a true BitNet Tiny lane exists

## Branch protection rule

Enable branch protection only after primary + fallback + hosted proofs are complete and source/sync boundary checks are clean.

Require exactly:

- `<Repo> Rust Small Result`

Do not require implementation lane checks directly.

## Stop conditions

Do not merge when any are true:

- repository runner endpoint used
- CPX42 assumes local `em-ci-rust:1.95`
- scratch dir setup occurs after rust-toolchain action
- normalized result is missing conditional needs
- fork PR can route to self-hosted
- hosted fallback removed
- branch protection changed in same PR
- release/publish/signing workflows touched
- source-affecting changes bypass the repo-specific cutover, promotion, or sync
  policy

## Merge conditions

Merge when all are true:

- org discovery is in place and functioning
- selected route target behavior matches lane type
- selected lane succeeds
- fallback lanes skip or succeed by forced validation as required
- normalized result succeeds
- guardrails pass
- no branch protection/release/publish changes are included
