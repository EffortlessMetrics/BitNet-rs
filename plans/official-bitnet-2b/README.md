# Official BitNet 2B Productization Plan

This plan governs `microsoft/BitNet-b1.58-2B-4T` as the official BitNet-rs
reference model family.

The current I2_S/QK256 GGUF route is already the bounded answer-ready and
product-CLI-ready route for CPU/CUDA surfaces. This plan keeps that route intact
while adding the governance and later proof steps needed for CPU excellence,
CUDA speed/residency review, Apple/ARM promotion, A770/OpenCL promotion, TL1/TL2
expansion, BF16-to-GPU-int2 research, and user-visible status/receipt polish.

Read the implementation sequence in:

- `plans/official-bitnet-2b/implementation-plan.md`

Primary source map:

- `docs/bitnet/official-2b/README.md`

Active campaign manifest:

- `docs/tracking/campaigns/official-bitnet-2b/active.toml`
