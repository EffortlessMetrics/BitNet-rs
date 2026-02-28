/* tslint:disable */
/* eslint-disable */

/**
 * WebAssembly bindings for BitNet 1-bit LLM inference
 *
 * This package provides WebAssembly bindings for running BitNet 1-bit Large Language Models
 * directly in the browser with optimized performance and memory usage.
 */

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
}

/**
 * Node/browser-friendly module initialization input.
 *
 * In Node.js this supports file paths and Buffer/typed array payloads.
 * In browsers this supports Request/Response/URL-based loading.
 */
export type InitInput =
  | string
  | URL
  | BufferSource
  | WebAssembly.Module
  | Response
  | Request;

/**
 * Runtime-neutral readable stream type for browser and Node.js.
 */
export type BitNetReadableStream = ReadableStream<Uint8Array>;

/**
 * Initialize the WebAssembly module
 * @param module_or_path - WebAssembly module or path to .wasm file
 * @returns Promise that resolves to the initialized module
 */
export default function init(module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;

/**
 * Configuration for WASM model loading
 */
export class WasmModelConfig {
  constructor();

  /** Maximum memory usage in bytes */
  max_memory_bytes: number | undefined;

  /** Enable progressive loading for large models */
  progressive_loading: boolean;

  /** Chunk size for progressive loading in bytes */
  chunk_size_bytes: number;

  /** Model format hint ("gguf", "safetensors", "auto") */
  format_hint: string;

  /** Tokenizer type ("gpt2", "sentencepiece", "auto") */
  tokenizer_type: string;
}

/**
 * WebAssembly-compatible BitNet model wrapper
 */
export class WasmBitNetModel {
  constructor(config?: WasmModelConfig);

  /**
   * Load model from byte array
   * @param model_bytes - Model data as Uint8Array
   * @param tokenizer_bytes - Optional tokenizer data
   * @returns Promise that resolves when model is loaded
   */
  load_from_bytes(model_bytes: Uint8Array, tokenizer_bytes?: Uint8Array): Promise<any>;

  /**
   * Get model information
   * @returns Model metadata and statistics
   */
  get_model_info(): any;

  /**
   * Check if model is loaded
   * @returns True if model is ready for inference
   */
  is_loaded(): boolean;

  /**
   * Get current memory usage in bytes
   * @returns Current memory usage
   */
  get_memory_usage(): number;

  /**
   * Get maximum allowed memory in bytes
   * @returns Maximum memory limit
   */
  get_max_memory(): number | undefined;

  /**
   * Set memory limit
   * @param bytes - New memory limit in bytes
   */
  set_memory_limit(bytes?: number): void;

  /**
   * Force garbage collection
   * @returns Number of bytes freed
   */
  gc(): number;

  /**
   * Unload the model to free memory
   */
  unload(): void;
}

/**
 * Configuration for text generation
 */
export class WasmGenerationConfig {
  constructor();

  /** Maximum number of new tokens to generate */
  max_new_tokens: number;

  /** Temperature for sampling (0.0 = greedy, higher = more random) */
  temperature: number;

  /** Top-k sampling parameter */
  top_k: number | undefined;

  /** Top-p (nucleus) sampling parameter */
  top_p: number | undefined;

  /** Repetition penalty */
  repetition_penalty: number;

  /** Random seed for deterministic generation */
  seed: number | undefined;

  /** Enable streaming output */
  streaming: boolean;

  /**
   * Add a stop token
   * @param token - Token that will halt generation
   */
  add_stop_token(token: string): void;

  /**
   * Remove a stop token
   * @param token - Token to remove from stop list
   */
  remove_stop_token(token: string): void;

  /**
   * Get stop tokens as array
   * @returns Array of stop tokens
   */
  get_stop_tokens(): string[];

  /**
   * Set stop tokens from array
   * @param tokens - Array of stop tokens
   */
  set_stop_tokens(tokens: string[]): void;
}

/**
 * WebAssembly inference wrapper
 */
export class WasmInference {
  constructor(model: WasmBitNetModel);

  /**
   * Generate text synchronously
   * @param prompt - Input text prompt
   * @param config - Generation configuration
   * @returns Generated text
   */
  generate(prompt: string, config?: WasmGenerationConfig): string;

  /**
   * Generate text asynchronously
   * @param prompt - Input text prompt
   * @param config - Generation configuration
   * @returns Promise that resolves to generated text
   */
  generate_async(prompt: string, config?: WasmGenerationConfig): Promise<string>;

  /**
   * Create a streaming generation iterator
   * @param prompt - Input text prompt
   * @param config - Generation configuration
   * @returns Streaming generation object
   */
  generate_stream(prompt: string, config?: WasmGenerationConfig): WasmGenerationStream;

  /**
   * Get generation statistics
   * @returns Statistics object
   */
  get_stats(): any;

  /**
   * Reset generation statistics
   */
  reset_stats(): void;

  /**
   * Check if the model is ready for inference
   * @returns True if ready
   */
  is_ready(): boolean;

  /**
   * Get current memory usage
   * @returns Memory usage in bytes
   */
  get_memory_usage(): number;

  /**
   * Force garbage collection
   * @returns Number of bytes freed
   */
  gc(): number;
}

/**
 * WebAssembly-compatible streaming generation
 */
export class WasmGenerationStream {
  /**
   * Get the next token
   * @returns Promise that resolves to iterator result
   */
  next(): Promise<IteratorResult<string>>;

  /**
   * Check if the stream is finished
   * @returns True if stream is complete
   */
  is_finished(): boolean;

  /**
   * Get current position in the stream
   * @returns Current token position
   */
  get_position(): number;

  /**
   * Get total number of tokens that will be generated
   * @returns Total token count
   */
  get_total_tokens(): number;

  /**
   * Cancel the stream
   */
  cancel(): void;

  /**
   * Convert to JavaScript async iterator
   * @returns Async iterator
   */
  to_async_iterator(): AsyncIterator<string>;

  /**
   * Convert to ReadableStream
   * @returns ReadableStream for use with Streams API
   */
  to_readable_stream(): BitNetReadableStream;
}

/**
 * Memory statistics
 */
export class MemoryStats {
  /** Current memory usage in bytes */
  readonly current_bytes: number;

  /** Maximum memory limit in bytes */
  readonly max_bytes: number | undefined;

  /** Memory usage as percentage */
  readonly usage_percent: number;

  /** Number of active allocations */
  readonly allocation_count: number;

  /** Whether garbage collection should be triggered */
  readonly should_gc: boolean;
}

/**
 * Memory-efficient buffer for progressive loading
 */
export class WasmBuffer {
  constructor(initial_capacity: number, max_capacity: number);

  /**
   * Append data to the buffer
   * @param data - Data to append
   */
  append(data: Uint8Array): void;

  /**
   * Get current size
   * @returns Buffer size in bytes
   */
  size(): number;

  /**
   * Get capacity
   * @returns Buffer capacity in bytes
   */
  capacity(): number;

  /**
   * Get maximum capacity
   * @returns Maximum buffer capacity in bytes
   */
  max_capacity(): number;

  /**
   * Clear the buffer
   */
  clear(): void;

  /**
   * Get data as Uint8Array
   * @returns Buffer contents
   */
  to_uint8_array(): Uint8Array;

  /**
   * Shrink buffer to fit current data
   */
  shrink_to_fit(): void;
}

/**
 * Progressive loader for large models
 */
export class ProgressiveLoader {
  constructor(chunk_size: number, max_size: number);

  /**
   * Set total expected size
   * @param size - Total size in bytes
   */
  set_total_size(size: number): void;

  /**
   * Load a chunk of data
   * @param chunk - Chunk data
   * @returns Loading progress (0.0 to 1.0)
   */
  load_chunk(chunk: Uint8Array): number;

  /**
   * Check if loading is complete
   * @returns True if complete
   */
  is_complete(): boolean;

  /**
   * Get loaded data
   * @returns All loaded data
   */
  get_data(): Uint8Array;

  /**
   * Get loading progress
   * @returns Progress from 0.0 to 1.0
   */
  get_progress(): number;

  /**
   * Get loaded size in bytes
   * @returns Loaded size
   */
  get_loaded_size(): number;

  /**
   * Get total size in bytes
   * @returns Total expected size
   */
  get_total_size(): number | undefined;
}

/**
 * Comprehensive benchmark suite for WASM BitNet
 */
export class WasmBenchmarkSuite {
  constructor();

  /**
   * Run all benchmarks
   * @returns Promise that resolves to comprehensive results
   */
  run_all_benchmarks(): Promise<any>;

  /**
   * Benchmark kernel performance
   * @returns Promise that resolves to kernel benchmark results
   */
  benchmark_kernels(): Promise<any>;

  /**
   * Benchmark memory management
   * @returns Promise that resolves to memory benchmark results
   */
  benchmark_memory(): Promise<any>;

  /**
   * Benchmark progressive loading
   * @returns Promise that resolves to loading benchmark results
   */
  benchmark_loading(): Promise<any>;

  /**
   * Benchmark inference performance
   * @returns Promise that resolves to inference benchmark results
   */
  benchmark_inference(): Promise<any>;
}

/**
 * Logging utilities
 */
export class Logger {
  /**
   * Log info message
   * @param message - Message to log
   */
  static info(message: string): void;

  /**
   * Log warning message
   * @param message - Message to log
   */
  static warn(message: string): void;

  /**
   * Log error message
   * @param message - Message to log
   */
  static error(message: string): void;

  /**
   * Log debug message
   * @param message - Message to log
   */
  static debug(message: string): void;
}

/**
 * Memory utilities
 */
export class MemoryUtils {
  /**
   * Format bytes as human-readable string
   * @param bytes - Number of bytes
   * @returns Formatted string (e.g., "1.5 MB")
   */
  static format_bytes(bytes: number): string;

  /**
   * Parse human-readable byte string to number
   * @param input - Input string (e.g., "1.5 MB")
   * @returns Number of bytes
   */
  static parse_bytes(input: string): number;
}

/**
 * Feature detection utilities
 */
export class FeatureDetection {
  /**
   * Check if WebAssembly SIMD is supported
   * @returns True if SIMD is supported
   */
  static supports_wasm_simd(): boolean;

  /**
   * Check if WebAssembly threads are supported
   * @returns True if threads are supported
   */
  static supports_wasm_threads(): boolean;

  /**
   * Check if WebAssembly bulk memory operations are supported
   * @returns True if bulk memory is supported
   */
  static supports_wasm_bulk_memory(): boolean;

  /**
   * Get WebAssembly feature support summary
   * @returns Object with feature support flags
   */
  static get_feature_support(): {
    simd: boolean;
    threads: boolean;
    bulkMemory: boolean;
  };
}

/**
 * Performance monitoring utilities
 */
export class PerformanceMonitor {
  constructor();

  /**
   * Add a performance mark
   * @param name - Mark name
   */
  mark(name: string): void;

  /**
   * Get elapsed time since start
   * @returns Elapsed time in milliseconds
   */
  elapsed(): number;

  /**
   * Get all marks
   * @returns Object with mark names and times
   */
  get_marks(): Record<string, number>;

  /**
   * Reset the monitor
   */
  reset(): void;
}

/**
 * JavaScript-friendly error type
 */
export class JsError extends Error {
  constructor(message: string);

  /** Error name */
  readonly name: string;

  /** Error message */
  readonly message: string;

  /** Stack trace */
  readonly stack?: string;

  /**
   * Convert to JavaScript Error object
   * @returns JavaScript Error
   */
  to_js_error(): Error;
}
