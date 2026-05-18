# Intel GPU plan index

This directory sequences the shared Intel GPU documentation, proof, and UX work.
It coordinates the existing A770 and Lunar Lake 258V campaigns without merging
their claim families.

Start with:

- `implementation-plan.md` for the PR-by-PR rollout.
- `docs/intel-gpu/README.md` for the source-of-truth map.

Hard boundaries for every plan item:

- A770 native OpenCL proof is not Arc 140V proof.
- Arc 140V OpenCL proof is not A770 proof.
- OpenVINO GPU proof is not native OpenCL proof or NPU proof.
- Dense SLM proof is not BitNet QK256/I2_S proof.
- CPU fallback cannot satisfy Intel GPU execution.
- No performance or residency claim is valid without profile-specific,
  quality-gated receipts and explicit not-claims.
