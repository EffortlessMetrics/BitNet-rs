# Rust Migration Candidates

This inventory tracks non-Rust helper code that is a good fit for moving into
workspace crates, `xtask`, or existing CLI commands. The goal is to keep durable
logic in the core Rust design and leave shell/Python only as thin compatibility
entry points when needed.

## Selection criteria

Prioritize migration when a helper:

- parses repository-owned JSON, GGUF, tokenizer, or receipt data,
- enforces CI or release policy,
- duplicates logic already present in a Rust crate,
- runs in GitHub Actions where Python dependencies add failure modes, or
- would benefit from Rust types and workspace tests.

## High-value candidates

| Current helper | Proposed Rust home | Why it fits |
| --- | --- | --- |
| `scripts/ripr-annotations.py` | `cargo xtask ripr-annotations` | CI annotation emission is pure JSON formatting and belongs beside the existing `ripr-pr` / `ripr-review-comments` control-plane commands. Migrated. |
| `scripts/check_greedy_argmax.py` | `bitnet-scoring-core` plus an `xtask` validation command | Greedy argmax invariants are inference/scoring semantics, not ad hoc scripting. Moving them would let Rust tests share the same oracle. |
| `scripts/render_perf_md.py` | `bitnet-inference-metrics-core` plus `xtask` report rendering | Performance receipts and Markdown rendering should use the same typed metric DTOs as the runtime/reporting crates. |
| `scripts/inspect_gguf.py` | `bitnet-gguf` / existing `inspect_gguf_metadata` example | GGUF parsing already has Rust ownership; remaining inspection logic should reuse the crate instead of maintaining a second parser. |
| `scripts/fix_gguf_tokenizer.py` | `bitnet-compat` / `bitnet-compat-core` | GGUF compatibility repair should live with compatibility diagnostics and preserve the “never mutate in place” repository contract. |
| `scripts/validate_readme_examples.py` | `xtask` documentation validation | README code-block validation is release policy and can be tested without Python subprocess conventions. |
| `scripts/check_doc_links.py` | `xtask lint-docs` or docs policy crate | Link extraction and internal path checks are deterministic repository policy; only optional external HTTP checks need network-aware handling. |

## Migration pattern

1. Extract reusable rules into the smallest relevant `*-core` crate when the
   logic is domain semantics.
2. Put repository orchestration, file walking, and CI output formatting in
   `xtask`.
3. Keep shell wrappers only for backwards-compatible command names, and make
   them delegate to `cargo xtask ...`.
4. Add Rust integration tests for CLI behavior before removing the old helper
   from workflows.
