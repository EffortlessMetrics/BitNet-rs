# MODEL-ARTIFACT-003 Reference-Runner Compatibility

**Date:** 2026-05-07
**Campaign:** `model-artifacts`
**Status:** diagnostic-only; no candidate is `answer_ready`

## Summary

`MODEL-ARTIFACT-003` records a narrower blocker discovered after the shared
reference-good search: the latest stock `llama.cpp` Windows CPU runner cannot
load the official Microsoft I2_S GGUF or the official-derived `tdh111`
IQ2_BN_R4 candidate. Both fail before prompt execution, so neither result can
be used as answer-readiness evidence.

This report does not change runtime behavior, tokenizer behavior, model loader
behavior, CUDA code, or answer quality gates.

## Runner

The target-local runner was downloaded from the latest `ggml-org/llama.cpp`
Windows CPU release available during this run:

```text
release = b9061
published_at = 2026-05-07T20:50:23Z
archive = llama-b9061-bin-win-cpu-x64.zip
archive_sha256 = c671e6740638441775366e384a279f9b2a3d621f9ef0150a4eb652e4b7c296d4
llama-completion = target/model-artifacts/tools/llama-b9061-win-cpu-x64/llama-completion.exe
version = 9061 (deab41ec6)
cpu_backend = ggml-cpu-zen4.dll
```

The downloaded runner and model files stayed under `target/` and are not
committed.

## Command Shape

Both compatibility checks used the same deterministic prompt shape:

```powershell
target\model-artifacts\tools\llama-b9061-win-cpu-x64\llama-completion.exe `
  -m <candidate.gguf> `
  -ngl 0 `
  -t 8 `
  -c 2048 `
  -b 1 `
  -n 8 `
  --temp 0 `
  --seed 42 `
  --no-display-prompt `
  --no-perf `
  -p "What is 2+2? Answer with only the number."
```

## Candidate Results

| Candidate | File | SHA256 | Result | Decision |
|---|---|---|---|---|
| `microsoft_bitnet_b158_2b_4t_gguf_i2s_current` | `ggml-model-i2_s.gguf` | `4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162` | Stock llama.cpp b9061 rejects `blk.0.ffn_down.weight` as removed `TYPE_IQ4_NL_4_4` with invalid block size. | remains rejected |
| `tdh111_bitnet_b158_2b_4t_iq2_bn_r4` | `bitnet1582b4t-iq2_bn_r4.gguf` | `a99001aaa5c1dc24acffe8035315c7d2970e82d8ccd3189383275c5d5a5287b5` | Stock llama.cpp b9061 rejects `blk.0.ffn_down.weight` with invalid GGML type `335`. | rejected for stock reference-runner compatibility |

The `tdh111` model card says the IQ2_BN files are for `ik_llama.cpp`, so this
stock runner failure is not evidence that the file is answer-bad under its
intended runner. It is evidence that the current shared answer-artifact gate
cannot promote it using stock `llama.cpp` alone.

## Not Promoted

Additional discovery found non-official or different-family candidates with
explicit tokenizer sidecars. They were not promoted into the BitNet
answer-ready path in this PR:

| Candidate | Reason |
|---|---|
| `nebuxcloud/Falcon3-3B-Instruct-1.58bit-GGUF` | Falcon3/Llama architecture, useful as a compatibility candidate only; it cannot satisfy packed BitNet answer readiness. |
| `BoscoTheDog/bitnet-mistral.0.2-330m-v0.2-grokfast-v2.9_gguf` | Mistral-derived small GGUF, useful for cheap tokenizer/format smoke only; it cannot satisfy the official BitNet packed-inference target. |

## Decision

No new `answer_ready` artifact was found.

The next useful unblocker is either:

1. a target-local `ik_llama.cpp` or Microsoft BitNet reference-runner attempt
   for the `tdh111` IQ2_BN_R4 candidate; or
2. a new upstream-supported BitNet GGUF/tokenizer artifact that stock
   reference runners can load and that passes the deterministic prompt suite.

Until then, backend answer-readiness lanes remain blocked or diagnostic-only.
