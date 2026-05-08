# CUDA-PROD-001 Strict Ask Receipt Default

## Summary

`CUDA-PROD-001` moves `bitnet ask` closer to a normal user path by making strict
CPU/CUDA ask runs receipt-backed even when the user does not pass
`--receipt-out`.

Before this change, strict ask validation happened only after `run` wrote a JSON
receipt to the user-supplied `--receipt-out` path. A strict command without
`--receipt-out` could still generate text, but it returned before checking the
receipt-level strict backend, fallback, and answer-quality gates.

After this change, strict ask writes a default receipt:

```text
target/bitnet/receipts/cuda-answer-readiness/strict-cuda-ask-latest.json
target/bitnet/receipts/cuda-answer-readiness/strict-cpu-ask-latest.json
```

The user may still pass `--receipt-out` to choose a path. Non-strict ask remains
unchanged and does not force a receipt.

## Claim Boundary

Allowed claim:

- Strict `bitnet ask` validates backend, fallback, and answer quality even when
  `--receipt-out` is omitted.

Not allowed:

- Broad chat quality beyond committed answer-readiness evidence.
- CUDA speedup.
- Full CUDA residency for every transformer operation.
- Completed model fetch/verify/install UX.

## Product Impact

This closes one narrow gap in the normal command path:

```powershell
bitnet ask --device nvidia-rtx-5070-ti-cuda --strict-cuda ...
```

now always has a machine-checkable answer receipt behind the strict validation
decision. The next product gap remains warm-session usability: loading once,
uploading weights once, asking multiple questions, and producing per-turn or
session receipts without broadening the claim boundary.
