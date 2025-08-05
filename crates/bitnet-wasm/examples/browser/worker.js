// BitNet WASM Web Worker
// This worker runs BitNet inference in a separate thread to keep the main thread responsive

import init, { 
    WasmBitNetModel, 
    WasmModelConfig, 
    WasmInference, 
    WasmGenerationConfig,
    Logger 
} from './pkg/bitnet_wasm.js';

let wasmModule = null;
let model = null;
let inference = null;

// Initialize the worker
async function initializeWorker() {
    try {
        Logger.info('Initializing BitNet WASM worker...');
        
        // Initialize WASM module
        wasmModule = await init();
        
        // Create a default model configuration for the worker
        const config = new WasmModelConfig();
        config.set_max_memory_bytes(256 * 1024 * 1024); // 256MB limit for worker
        config.set_progressive_loading(true);
        
        // For demo purposes, we'll create a mock model
        // In a real implementation, you'd load the actual model here
        model = new WasmBitNetModel(config);
        
        // Note: In a real implementation, you'd need to transfer the model data
        // from the main thread or load it directly in the worker
        
        Logger.info('BitNet WASM worker initialized successfully');
        
        // Notify main thread that worker is ready
        self.postMessage({ 
            type: 'initialized',
            data: 'Worker ready for inference'
        });
        
    } catch (error) {
        Logger.error(`Worker initialization failed: ${error}`);
        self.postMessage({ 
            type: 'error', 
            error: error.message 
        });
    }
}

// Generate text in the worker
async function generateText(prompt, config) {
    try {
        Logger.info(`Worker generating text for prompt: "${prompt.substring(0, 50)}..."`);
        
        if (!model) {
            throw new Error('Model not loaded in worker');
        }
        
        // Create inference engine if not already created
        if (!inference) {
            inference = new WasmInference(model);
        }
        
        // Create generation config
        const genConfig = new WasmGenerationConfig();
        genConfig.set_max_new_tokens(config.maxTokens || 50);
        genConfig.set_temperature(config.temperature || 0.7);
        genConfig.set_top_k(config.topK || 50);
        genConfig.set_top_p(config.topP || 0.9);
        
        // Generate text (this is a mock implementation)
        // In a real implementation, this would call the actual inference engine
        const result = await simulateGeneration(prompt, genConfig);
        
        Logger.info(`Worker generation completed: ${result.length} characters`);
        
        // Send result back to main thread
        self.postMessage({
            type: 'generation_complete',
            data: result
        });
        
    } catch (error) {
        Logger.error(`Worker generation failed: ${error}`);
        self.postMessage({
            type: 'error',
            error: error.message
        });
    }
}

// Simulate text generation (for demo purposes)
async function simulateGeneration(prompt, config) {
    const maxTokens = config.max_new_tokens || 50;
    const temperature = config.temperature || 0.7;
    
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
    
    // Generate mock response
    const responses = [
        `This is a response generated in a Web Worker for the prompt: "${prompt}". `,
        `The worker processed this request with temperature ${temperature} and max tokens ${maxTokens}. `,
        `Web Workers allow us to run computationally intensive tasks like LLM inference without blocking the main thread. `,
        `This keeps the user interface responsive while the model generates text in the background. `,
        `The BitNet WASM implementation is optimized for browser environments with memory constraints.`
    ];
    
    let result = '';
    const numSentences = Math.min(maxTokens / 10, responses.length);
    
    for (let i = 0; i < numSentences; i++) {
        result += responses[i];
        
        // Simulate progressive generation
        if (i < numSentences - 1) {
            await new Promise(resolve => setTimeout(resolve, 200));
        }
    }
    
    return result.trim();
}

// Handle streaming generation in worker
async function generateStream(prompt, config) {
    try {
        Logger.info(`Worker starting streaming generation for: "${prompt.substring(0, 50)}..."`);
        
        const maxTokens = config.maxTokens || 50;
        const words = [
            'This', 'is', 'a', 'streaming', 'response', 'generated', 'in', 'a', 'Web', 'Worker.',
            'Each', 'token', 'is', 'sent', 'back', 'to', 'the', 'main', 'thread', 'as', 'it', 'is', 'generated.',
            'This', 'allows', 'for', 'real-time', 'text', 'generation', 'without', 'blocking', 'the', 'UI.',
            'The', 'BitNet', 'model', 'runs', 'efficiently', 'in', 'WebAssembly', 'with', 'optimized', 'kernels.'
        ];
        
        for (let i = 0; i < Math.min(maxTokens, words.length); i++) {
            // Simulate generation delay
            await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200));
            
            // Send token back to main thread
            self.postMessage({
                type: 'stream_token',
                data: {
                    token: words[i] + ' ',
                    position: i,
                    isComplete: i === Math.min(maxTokens, words.length) - 1
                }
            });
        }
        
        Logger.info('Worker streaming generation completed');
        
    } catch (error) {
        Logger.error(`Worker streaming failed: ${error}`);
        self.postMessage({
            type: 'error',
            error: error.message
        });
    }
}

// Load model data in worker
async function loadModel(modelData, tokenizerData) {
    try {
        Logger.info(`Loading model in worker: ${modelData.length} bytes`);
        
        const config = new WasmModelConfig();
        config.set_max_memory_bytes(256 * 1024 * 1024); // 256MB for worker
        
        model = new WasmBitNetModel(config);
        
        // Convert ArrayBuffer to Uint8Array
        const modelBytes = new Uint8Array(modelData);
        const tokenizerBytes = tokenizerData ? new Uint8Array(tokenizerData) : null;
        
        // Load model
        await model.load_from_bytes(modelBytes, tokenizerBytes);
        
        // Create inference engine
        inference = new WasmInference(model);
        
        Logger.info('Model loaded successfully in worker');
        
        self.postMessage({
            type: 'model_loaded',
            data: {
                modelSize: modelData.length,
                memoryUsage: model.get_memory_usage()
            }
        });
        
    } catch (error) {
        Logger.error(`Worker model loading failed: ${error}`);
        self.postMessage({
            type: 'error',
            error: error.message
        });
    }
}

// Get worker statistics
function getWorkerStats() {
    const stats = {
        modelLoaded: model !== null,
        inferenceReady: inference !== null,
        memoryUsage: model ? model.get_memory_usage() : 0,
        maxMemory: model ? model.get_max_memory() : 0
    };
    
    self.postMessage({
        type: 'worker_stats',
        data: stats
    });
}

// Cleanup worker resources
function cleanup() {
    try {
        if (model) {
            model.unload();
            model = null;
        }
        
        inference = null;
        
        Logger.info('Worker resources cleaned up');
        
        self.postMessage({
            type: 'cleanup_complete'
        });
        
    } catch (error) {
        Logger.error(`Worker cleanup failed: ${error}`);
        self.postMessage({
            type: 'error',
            error: error.message
        });
    }
}

// Message handler
self.onmessage = async function(e) {
    const { type, prompt, config, modelData, tokenizerData } = e.data;
    
    try {
        switch (type) {
            case 'init':
                await initializeWorker();
                break;
                
            case 'generate':
                await generateText(prompt, config || {});
                break;
                
            case 'generate_stream':
                await generateStream(prompt, config || {});
                break;
                
            case 'load_model':
                await loadModel(modelData, tokenizerData);
                break;
                
            case 'get_stats':
                getWorkerStats();
                break;
                
            case 'cleanup':
                cleanup();
                break;
                
            default:
                Logger.warn(`Unknown message type: ${type}`);
                self.postMessage({
                    type: 'error',
                    error: `Unknown message type: ${type}`
                });
        }
        
    } catch (error) {
        Logger.error(`Worker message handling failed: ${error}`);
        self.postMessage({
            type: 'error',
            error: error.message
        });
    }
};

// Handle worker errors
self.onerror = function(error) {
    Logger.error(`Worker error: ${error.message}`);
    self.postMessage({
        type: 'error',
        error: error.message
    });
};

// Handle unhandled promise rejections
self.onunhandledrejection = function(event) {
    Logger.error(`Worker unhandled rejection: ${event.reason}`);
    self.postMessage({
        type: 'error',
        error: `Unhandled rejection: ${event.reason}`
    });
};

Logger.info('BitNet WASM Worker script loaded');