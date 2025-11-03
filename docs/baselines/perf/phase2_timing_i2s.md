=== Timing Summary ===
Model: models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
Date: 2025-10-22T07:01:37Z

timing: embed_us=26
timing: forward_us=1856157
timing: logits_us=72092
timing: sample_us=179
timing: embed_us=21
timing: forward_us=1865375
timing: logits_us=85374
timing: sample_us=155
timing: embed_us=26
timing: forward_us=1886460
timing: logits_us=71297
timing: sample_us=137

=== Analysis (Median of 3 runs) ===

Performance Breakdown:
  Embedding:        26 μs    (0.026 ms)  -  0.00%
  Forward Pass:  1,865 μs  (1,865.375 ms)  - 95.61%
  Logits:           72 μs   (72.092 ms)  -  3.70%
  Sampling:        155 μs    (0.155 ms)  -  0.01%
  ────────────────────────────────────────────────
  Total:         1,951 μs  (1,950.925 ms)

Throughput: 0.5126 tokens/second

Key Observations:
- Forward pass dominates at 95.61% of total time
- Logits computation is 3.70% overhead
- Embedding and sampling are negligible (<0.02% combined)

=== System Configuration ===

Hardware:
  CPU: AMD Ryzen 9 9950X3D 16-Core Processor
  Cores: 16 physical / 32 logical (2 threads per core)
  Architecture: x86_64
  Cache: L1d=768 KiB, L1i=512 KiB, L2=16 MiB, L3=96 MiB
  
CPU Features (relevant):
  SIMD: AVX, AVX2, AVX-512 (F, DQ, IFMA, CD, BW, VL, VBMI, VBMI2, 
        VNNI, BITALG, VPOPCNTDQ, VP2INTERSECT, BF16)
  Other: FMA, BMI1, BMI2, SHA-NI, VAES, VPCLMULQDQ

Software:
  Rust: rustc 1.92.0-nightly (4082d6a3f 2025-09-27)
  Cargo: 1.92.0-nightly (f2932725b 2025-09-24)
  OS: Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)

Build Configuration:
  RUSTFLAGS: -C target-cpu=native -C opt-level=3
  Features: cpu,full-cli
  Profile: release

Model:
  Path: models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf
  Size: 1.2 GB
  Format: I2_S (BitNet32-F16 2-bit signed quantization)
  Parameters: ~2B

Test Configuration:
  Prompt: "2+2="
  Max tokens: 1
  Sampling: Greedy (temperature=0.0)
  Iterations: 3

=== Notes ===

1. This is Phase 2 timing measurement using the optimized release binary
   with native CPU features enabled (AVX-512).

2. The ~0.5 tok/s performance is expected for I2_S format on CPU inference
   with the current implementation.

3. The forward pass bottleneck (95.61%) indicates optimization focus should
   be on the transformer computation kernels.

4. AVX-512 instructions are available but may not be fully utilized in the
   current implementation. Further SIMD optimization could improve throughput.

5. Performance is measured on WSL2, which may have slight overhead compared
   to native Linux.
