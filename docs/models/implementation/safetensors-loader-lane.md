# safetensors loader lane

This lane records future native tensor loading for non-BitNet model families.

A loader scaffold must include tensor-name mapping, dtype/quantization expectations, sharding policy, tokenizer/prompt location, architecture module target, unsupported tensor policy, and receipt requirements. It must not claim inference until runtime execution is smoke-tested and receipt-backed.
