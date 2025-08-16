# Explanation: Memory Management

This document explains the different memory management techniques supported by BitNet.rs.

## Overview

Memory management is a critical aspect of running large language models, as they can consume a large amount of memory. BitNet.rs includes several features for optimizing memory usage, which can help you to run larger models on the same hardware, and to improve performance by reducing memory-related bottlenecks.

## KV Cache

The KV cache is a key-value cache that is used to store the intermediate results of the attention mechanism. In a transformer model, the attention mechanism is used to compute the relationships between the different tokens in the input sequence. The results of the attention mechanism are stored in a key-value cache, which can be reused for subsequent tokens.

By caching the attention results, the KV cache can significantly improve performance by avoiding the need to recompute the attention scores for each token. This is especially important for long sequences, where the attention computation can be a major bottleneck.

The size of the KV cache can be configured when creating the inference engine. A larger cache can improve performance, but it will also consume more memory.

## Memory-Mapped Files

Memory-mapped files are a feature of the operating system that allows you to map a file into memory, and to access it as if it were a regular array in memory. BitNet.rs can use memory-mapped files to load large models without having to load the entire model into memory at once.

When you use a memory-mapped file, the operating system will only load the parts of the file that are actually needed into memory. This can significantly reduce memory usage when loading large models, and can also improve performance by reducing the time it takes to load the model.

## Memory Pooling

Memory pooling is a technique where you pre-allocate a large pool of memory, and then allocate smaller blocks of memory from the pool as needed. This can be more efficient than allocating memory from the operating system for each request, as it can reduce the overhead of memory allocation.

BitNet.rs uses memory pooling for a variety of tasks, including storing the KV cache and the model weights.

## Threading

BitNet.rs uses the Rayon library for parallel processing on the CPU. The number of threads used by Rayon can be configured using the `RAYON_NUM_THREADS` environment variable. By default, Rayon will use one thread for each CPU core.

For memory-sensitive applications, you may want to reduce the number of threads to reduce memory usage. This is because each thread will have its own stack, which can consume a significant amount of memory.
