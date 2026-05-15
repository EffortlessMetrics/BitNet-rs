# Reference Instrumentation

This directory holds target-local diagnostic instrumentation for external
BitNet/llama.cpp reference builds. These patches are not part of the default
`patches/` directory and are not applied by `ci/fetch_bitnet_cpp.ps1`.

Use these only to localize shared Rust/reference numerical divergence. They do
not promote reference parity, A770 semantic quality, selected attention, KV
residency, or full device residency.

