<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-a770 | A770-023 | #300 | `codex/intel-a770/A770-023-first-mismatch-logit-margin` | Add a compact first-mismatch cross-chosen logit-margin frontier to the CPU/A770 answer-parity receipt so the generated-output divergent yes_no_water case reports whether both chosen-token logits are available, whether the lanes have opposite argmax choices, and whether the A770-side chosen-token margin is a near tie, without changing runtime math or promoting CPU/A770 parity. |
