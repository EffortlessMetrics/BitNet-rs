# Intel NPU Goal

Make Intel Lunar Lake NPU validation receipt-backed through OpenVINO static-shape detection, smoke, parity, and proof artifacts without conflating NPU, GPU, or CPU work.

Device-node visibility is not inference, OpenVINO NPU smoke is not full BitNet inference, and CPU fallback cannot count as NPU execution. Do not assume WSL can see the NPU unless OpenVINO reports NPU inside that environment.
