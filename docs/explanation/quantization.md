# Explanation: Quantization

This document explains the different quantization techniques supported by BitNet.rs.

## What is Quantization?

Quantization is the process of reducing the precision of a model's weights. For example, a model's weights are typically stored as 32-bit floating-point numbers (FP32). Quantization can reduce the precision to 16-bit floating-point (FP16), 8-bit integers (INT8), or even lower.

The main benefits of quantization are:

- **Reduced memory usage**: Lower-precision weights take up less memory, which means you can run larger models on the same hardware.
- **Faster inference**: Lower-precision arithmetic is often faster, especially on hardware that has specialized support for it (e.g., GPUs with Tensor Cores).

The main drawback of quantization is that it can lead to a loss of accuracy. However, by using sophisticated quantization techniques, it is often possible to achieve significant performance improvements with minimal loss of accuracy.

## BitNet Quantization

BitNet.rs is designed to work with "1-bit" models, which are a type of quantized model where the weights are represented by a very small number of bits. The original BitNet paper proposed a 1.58-bit quantization scheme, where the weights are ternary, meaning they can take one of three values: -1, 0, or 1.

BitNet.rs supports several different quantization schemes, each with its own trade-offs between performance and accuracy.

### I2S Quantization

I2S (2-bit signed) is a universal quantization scheme that is supported on all hardware. It quantizes the weights to 2-bit signed integers, which means each weight can take one of four values. This provides a good balance between performance and accuracy, and is a good choice for general-purpose use.

### TL1 Quantization

TL1 (Table Lookup 1) is a table-lookup-based quantization scheme that is optimized for ARM NEON processors. It uses a lookup table to perform the quantization and dequantization operations, which can be very fast on ARM-based devices such as mobile phones and the Apple M1/M2 chips.

### TL2 Quantization

TL2 (Table Lookup 2) is a table-lookup-based quantization scheme that is optimized for x86 processors with AVX2 or AVX-512 support. Like TL1, it uses a lookup table to perform the quantization and dequantization operations, but the table is optimized for the specific instruction set of x86 processors.

### Dynamic Quantization

Dynamic quantization is a technique where the quantization is performed at runtime, on a per-input basis. This can provide better accuracy than static quantization, but it can also be slower. Dynamic quantization is a good choice for use cases where accuracy is more important than performance.

## Choosing a Quantization Scheme

The best quantization scheme for your use case will depend on your specific requirements. Here are some general guidelines:

- **For general-purpose use on a variety of hardware**, use I2S.
- **For maximum performance on ARM-based devices**, use TL1.
- **For maximum performance on x86-based devices**, use TL2.
- **For maximum accuracy**, use dynamic quantization.

You can also use the `QuantizerFactory::best_for_arch()` method to automatically select the best quantization scheme for the current hardware.
