# ADR-005: Automatic CPU fallback when GPU unavailable

- **Status:** Accepted
- **Date:** 2025-07-14
- **Context:** bitnet-rs must work reliably on machines without a GPU — CI runners,
  cloud VMs, laptops with driver issues, and WSL environments.  The existing
  architecture already implements this pattern: `KernelManager` probes providers in
  priority order (CUDA → AVX-512 → AVX2 → NEON → scalar fallback) and
  `DeviceAwareQuantizer` transparently falls back from a GPU primary to a CPU
  secondary.

  With multiple GPU backends ([ADR-002](./ADR-002-microcrate-per-backend.md)) the
  fallback surface grows: any of OpenCL, Vulkan, CUDA, ROCm, or Metal may fail to
  initialize.  We need a consistent, predictable fallback policy.

- **Decision:** GPU availability is detected at runtime, never assumed at compile time.
  The fallback cascade is:

  1. **Probe** — At startup, `bitnet-device-probe` enumerates available GPU devices
     via each compiled backend.  Results are logged as
     `requested=<user_pref> detected=[<backends>] selected=<chosen>` (the existing
     `BackendStartupSummary` pattern).
  2. **Select** — `KernelManager::select_best()` picks the highest-priority available
     provider.  Priority: CUDA > OpenCL > Vulkan > ROCm > Metal > AVX-512 > AVX2 >
     NEON > scalar.
  3. **Fallback** — If no GPU provider initializes successfully, CPU SIMD kernels are
     used.  A `warn_once!` message notifies the user: *"No GPU detected; falling back
     to CPU (<simd_level>). Set BITNET_LOG=debug for details."*
  4. **Strict mode** — When `BITNET_STRICT_MODE=1`, GPU fallback is still allowed
     (it's a hardware limitation, not a configuration error), but `BITNET_GPU_FAKE`
     is rejected.

  The selected backend is recorded in inference receipts (`compute_path` field) for
  auditability.

- **Consequences:**
  - *Positive:* `bitnet-rs` always works — users never see "GPU not found" errors
    that block inference entirely.
  - *Positive:* Transparent to higher layers — inference, server, and CLI code never
    branch on device type.
  - *Positive:* Receipts and startup logs make the selected backend auditable.
  - *Negative:* Users may not realize they're running on CPU and report poor
    performance; mitigated by the startup log line and `--device` CLI flag.
  - *Negative:* Dispatch logic grows with each new backend; mitigated by the
    `KernelProvider` trait and priority list in `KernelManager`.

## Alternatives considered

- **Fail-fast when GPU requested but unavailable:** Simpler but hostile UX; users
  on headless servers would need to explicitly pass `--device cpu`.
- **Compile-time backend selection only (`#[cfg]`):** No runtime overhead but cannot
  adapt to the deployment environment; a single binary couldn't serve both GPU and
  CPU machines.
- **Environment-variable-only selection (`BITNET_DEVICE=cpu`):** Requires user
  knowledge; bad default experience.  Still supported as an override but not
  required.

## How to revert

Make GPU backends fail-fast (return `Err`) instead of falling through to CPU.  Add a
`--device cpu` flag and require it when no GPU is present.  Remove the automatic
probe-and-select logic from `KernelManager`.
