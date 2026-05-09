# CUDA-UX-002: Strict Ask Proof Summary

## Summary

`CUDA-UX-002` makes strict `bitnet ask` reuse the receipt explanation layer added
by `CUDA-UX-001`. Instead of printing a bespoke one-line proof string, strict
CPU/CUDA ask now builds a normalized `ReceiptExplanation` from the answer receipt
and prints a compact proof summary with the same field extraction used by:

```powershell
bitnet receipts explain <receipt.json>
```

The shared summary covers:

- model identity;
- model-aware planner route;
- selected backend and runtime API;
- kernel IDs;
- fallback status;
- answer quality signal;
- upload-once weight status;
- measured kernel time and transfer byte fields where present;
- full CUDA residency claim status;
- speed claim status;
- receipt path.

## Claim Boundary

May claim:

- strict `bitnet ask` proof summaries reuse the same normalized receipt
  explanation object as `bitnet receipts explain`;
- strict ask output exposes route, backend, kernel, fallback, quality, measured
  timing/transfer fields, upload-once status, claim limits, and receipt path.

Must not claim:

- new inference behavior;
- tokenizer, prompt-template, loader, transformer, kernel, benchmark, or server
  behavior changes;
- dense GGUF inference;
- BitNet packed proof from dense CUDA receipts;
- speedup;
- full CUDA residency.

Schema-specific validation remains owned by the existing strict ask receipt
validators. This is a user-output reuse of the proof cockpit, not a new receipt
acceptance gate.
