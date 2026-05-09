<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-silicon-macbook | MB-AS-001 | #4184 | `codex/apple-silicon-macbook/MB-AS-001-machine-profile` | Add a MacBook Apple Silicon machine/storage/profile receipt contract that records chip, memory, macOS, free disk, cache root, thermal/mobile context when available, and CPU/NEON, Metal, and MPSGraph visibility without running model inference. |
| intel-258v-platform | CPU258V-018 | #4178 | `codex/lunar-lake/CPU258V-018-hf-prompt-token-parity` | Compare official HF AutoTokenizer.apply_chat_template rendered prompts and token IDs against BitNet-rs metadata-authoritative prompt-authority-audit output for math_2_plus_2, say_ok, capital_france, and yes_no_water, recording mismatch indices and preserving no inference, logits, answer-quality, speed, Arc/NPU, or QK256 claims. |
| nvidia-5070ti | CUDA-UX-001 | #4185 | `codex/cuda-ux-receipts-explain` | Add a user-facing `bitnet receipts explain` command that reads existing BitNet and dense CUDA JSON receipts, supports `--latest` under target/bitnet/receipts, and prints or emits a compact proof summary covering artifact kind, claim, model, backend, execution plan route, kernels, quality, timing, residency, and claim limits without changing inference, tokenizer, loader, kernel, transformer, benchmark, or server behavior. |
| tracker-infra | TRACKER-003 | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |
