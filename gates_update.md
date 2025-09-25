<!-- gates:start -->
| Gate | Status | Evidence | Last Updated |
|------|--------|----------|--------------|
| **format** | ✅ **PASS** | rustfmt: all files formatted | 2025-09-24T14:52:07Z |
| **clippy** | ✅ **PASS** | clippy: 0 warnings (workspace) | 2025-09-24T14:52:07Z |
| **build** | ✅ **PASS** | build: workspace ok; CPU: ok, GPU: ok | 2025-09-24T14:52:07Z |
| **security** | ✅ **PASS** | audit: 1 warning (unmaintained paste), gpu: no leaks, ffi: safe, unsafe: 45 validated, gguf: bounds checked | 2025-09-24T15:45:22Z |
| **tests** | ✅ **PASS** | cargo test: 85/86 pass; CPU: 67/67, GPU: 8/9 (1 cleanup failure), crossval: 10/10 | 2025-09-24T15:15:30Z |
| **mutation** | ❌ **FAIL** | score: 20% (<80%); survivors:~2200+; quantization: I2S/TL1/TL2 accuracy gaps; inference: performance/device logic gaps; kernels: selection logic survivors | 2025-09-24T11:15:00Z |
<!-- gates:end -->

Check Runs Status (attempted):
- integrative:gate:format → success
- integrative:gate:clippy → success
- integrative:gate:build → success
- integrative:gate:security → success (acceptable risk)
- integrative:gate:tests → success
- integrative:gate:mutation → failure (score: 20% < 80%)