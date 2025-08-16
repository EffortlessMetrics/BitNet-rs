# Explanation: Inference Engine

This document explains how the BitNet.rs inference engine works.

## Overview

The inference engine is the core of the BitNet.rs project. It is responsible for taking a model and a tokenizer, and running inference to generate text. The `InferenceEngine` struct is the main entry point for all inference operations.

The inference engine is designed to be:

- **High-performance**: The engine is highly optimized for both CPU and GPU inference, and includes features such as a KV cache and streaming support to maximize performance.
- **Flexible**: The engine can be used with any model that implements the `Model` trait, and any tokenizer that implements the `Tokenizer` trait.
- **Easy to use**: The engine provides a simple, high-level API for text generation, while also providing low-level access to the underlying components for advanced use cases.

## Components

The inference engine is made up of the following components:

- **Model**: The `Model` trait defines the interface for all BitNet models. The inference engine uses the model to perform the forward pass and generate logits.
- **Tokenizer**: The `Tokenizer` trait defines the interface for all tokenizers. The inference engine uses the tokenizer to encode the input prompt and decode the generated tokens.
- **Backend**: The `Backend` trait defines the interface for the CPU and GPU backends. The inference engine uses the backend to perform the actual computation.
- **KV Cache**: The KV cache is a key-value cache that is used to store the intermediate results of the attention mechanism. This can significantly improve performance by avoiding the need to recompute the attention scores for each token.

## The Inference Process

The inference process consists of the following steps:

1.  **Tokenization**: The input prompt is encoded into a sequence of token IDs by the tokenizer.
2.  **Embedding**: The token IDs are embedded into a tensor by the model.
3.  **Forward Pass**: The embedded tensor is passed through the model to generate a tensor of logits. This is done by the backend.
4.  **Sampling**: A new token is sampled from the logits using a sampling strategy (e.g., greedy, top-k, top-p).
5.  **Decoding**: The new token is decoded into a string by the tokenizer.
6.  **Streaming**: The decoded token is yielded to the user.
7.  **Loop**: The new token is appended to the input sequence, and the process is repeated until a stop condition is met (e.g., the end-of-sentence token is generated, or the maximum number of tokens has been generated).

## Streaming

The inference engine supports streaming, which means that it can generate text one token at a time, rather than generating the entire sequence at once. This can significantly improve the user experience, as it allows the user to see the generated text as it is being produced.

Streaming is implemented using the `GenerationStream` struct, which is an asynchronous stream that yields a new token on each iteration.

## Backends

The inference engine supports both CPU and GPU backends.

- **CPU Backend**: The CPU backend is optimized for running inference on the CPU. It uses SIMD instructions (e.g., AVX2, NEON) to maximize performance.
- **GPU Backend**: The GPU backend is optimized for running inference on the GPU. It uses CUDA for NVIDIA GPUs and Metal for Apple GPUs.

The backend is selected automatically based on the available hardware, or can be specified manually when creating the inference engine.
