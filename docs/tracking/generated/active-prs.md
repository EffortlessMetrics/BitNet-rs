<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-015 | #4911 | `codex/slm-cpu-015-thread-thermal-envelope` | Add bounded i5-8250U Qwen3-0.6B Q8_0 thread and timing envelope evidence after SLM-CPU-014 by running or documenting strict warm-session measurements for 1, 2, 4, and 8 CPU threads where available. The slice must preserve generated IDs, strict GGUF tokenizer authority, selected CPU backend, and fallback=false, record warm/cold, prompt/corpus, thread count, timing, and any thermal/power fields as measured or explicitly unavailable, and keep claims bounded to the recorded host/model/corpus/backend. It must not claim sustained 8250U throughput beyond recorded thermal boundaries, Q4/Q5 quant expansion, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
