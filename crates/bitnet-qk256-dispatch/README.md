# bitnet-qk256-dispatch

Single-responsibility microcrate for running I2_S QK256 matrix-vector dispatch on Candle tensors.

This crate extracts the QK256 tensor-shape validation, byte flattening, GEMV invocation, and output reshape logic used by transformer projection paths.
