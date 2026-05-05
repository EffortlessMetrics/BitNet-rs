# PR Playbook

## PR size

One PR should do one of:

- add or update tracking
- collapse one leaf crate
- collapse one small cluster into a destination module
- fix one truth-boundary bug
- add one verification gate
- wire one runtime path

Avoid mixing structural collapse with behavior changes.

## PR body template

```md
## Summary

- ...

## Work item

- `TRUTH-001`

## Scope

Allowed paths:
- ...

Out of scope:
- ...

## Acceptance

- [ ] ...

## Verification

Verification result:
- passed:
  - `...`
- failed:
  - `...`
- blocked locally:
  - `...`
- not run:
  - `...`

## Notes

- No behavior change / behavior change explained
- Follow-up items
```

## State transitions

```text
proposed -> ready       only when scope, acceptance, and verification are clear
ready -> in_progress    when a branch starts work
in_progress -> pr_open  when a PR is opened
pr_open -> merged       only after merge
pr_open -> blocked      when waiting on an external blocker
pr_open -> superseded   when replaced by another PR
```

Work item state must match `status.md`. A PR must not mark itself merged. Only
update a merged state after the merge SHA exists.

## Scope exceptions

Every work item may update these tracker files even when they are not listed in
`scope.allowed_paths`:

- `docs/tracking/bitnet-alignment/status.md`
- `docs/tracking/bitnet-alignment/workstream-ledger.yaml`

Use the exception only for state, PR number, verification notes, and follow-up items.
Do not reshape unrelated tracker sections inside implementation PRs.

## Follow-up rule

If a PR discovers adjacent work, do not implement it unless it is required for the
current item acceptance. Add a `proposed` ledger item with scope, acceptance, and
verification instead.

## Merge rules

- Prefer small PRs that keep main green.
- Close duplicates aggressively.
- Do not reopen superseded cleanup.
- If a PR discovers a larger issue, add a new ledger item instead of expanding scope.
- If CI fails outside the changed scope, record it in `status.md`; do not opportunistically rewrite unrelated systems.

## Collapse rules

- Move code first.
- Preserve tests.
- Update imports.
- Remove workspace member.
- Remove dependency edges.
- Keep module names SRP.
- Avoid public re-export sprawl.
