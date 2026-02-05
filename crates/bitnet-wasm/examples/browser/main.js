// BitNet WASM Browser Integration Example
import init, {
    WasmBitNetModel,
    WasmModelConfig,
    WasmInference,
    WasmGenerationConfig,
    WasmBenchmarkSuite,
    Logger,
    MemoryUtils,
    FeatureDetection
} from './pkg/bitnet_wasm.js';

// Global state
let wasmModule = null;
let model = null;
let inference = null;
let worker = null;
let streamingActive = false;
let benchmarkSuite = null;

// Initialize the application
async function initApp() {
    try {
        updateStatus('Initializing WebAssembly module...', 'loading');
        updateProgress(10);

        // Initialize WASM module
        wasmModule = await init();
        updateProgress(30);

        // Initialize benchmark suite
        benchmarkSuite = new WasmBenchmarkSuite();
        updateProgress(50);

        // Detect platform features
        await detectPlatformFeatures();
        updateProgress(70);

        // Load settings
        loadSettings();
        updateProgress(90);

        updateStatus('WebAssembly module initialized successfully!', 'success');
        updateProgress(100);

        Logger.info('BitNet WASM application initialized');

        // Hide progress bar after a delay
        setTimeout(() => {
            document.getElementById('progress').parentElement.style.display = 'none';
        }, 1000);

    } catch (error) {
        updateStatus(`Initialization failed: ${error.message}`, 'error');
        Logger.error(`Initialization error: ${error}`);
    }
}

// Update status display
function updateStatus(message, type = 'loading') {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
}

// Update progress bar
function updateProgress(percent) {
    const progressEl = document.getElementById('progress');
    progressEl.style.width = `${percent}%`;
}

// Detect platform features
async function detectPlatformFeatures() {
    const features = FeatureDetection.get_feature_support();

    document.getElementById('platform-simd').textContent =
        features.simd ? 'Supported' : 'Not Supported';

    // Get device info
    const memory = navigator.deviceMemory || 'Unknown';
    const cores = navigator.hardwareConcurrency || 'Unknown';

    document.getElementById('platform-memory').textContent = memory;
    document.getElementById('platform-cores').textContent = cores;

    Logger.info(`Platform features detected - SIMD: ${features.simd}, Memory: ${memory}GB, Cores: ${cores}`);
}

// Load model from file
async function loadModel() {
    const fileInput = document.getElementById('model-file');
    const file = fileInput.files[0];

    if (!file) {
        updateStatus('Please select a model file', 'error');
        return;
    }

    try {
        updateStatus('Loading model...', 'loading');
        Logger.info(`Loading model: ${file.name} (${MemoryUtils.format_bytes(file.size)})`);

        // Create model configuration
        const config = new WasmModelConfig();
        config.set_max_memory_bytes(parseInt(document.getElementById('memory-limit').value) * 1024 * 1024);
        config.set_chunk_size_bytes(parseInt(document.getElementById('chunk-size').value) * 1024 * 1024);

        // Create model instance
        model = new WasmBitNetModel(config);

        // Read file as array buffer
        const arrayBuffer = await file.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);

        // Load model
        await model.load_from_bytes(uint8Array);

        // Create inference wrapper
        inference = new WasmInference(model);

        // Update UI
        document.getElementById('model-size').textContent = MemoryUtils.format_bytes(file.size);
        document.getElementById('memory-usage').textContent = MemoryUtils.format_bytes(model.get_memory_usage());
        document.getElementById('generate').disabled = false;
        document.getElementById('start-streaming').disabled = false;

        updateStatus('Model loaded successfully!', 'success');
        Logger.info('Model loaded and inference engine initialized');

    } catch (error) {
        updateStatus(`Model loading failed: ${error.message}`, 'error');
        Logger.error(`Model loading error: ${error}`);
    }
}

// Generate text
async function generateText() {
    if (!inference) {
        updateStatus('Please load a model first', 'error');
        return;
    }

    try {
        const prompt = document.getElementById('prompt').value;
        if (!prompt.trim()) {
            updateStatus('Please enter a prompt', 'error');
            return;
        }

        updateStatus('Generating text...', 'loading');
        const startTime = performance.now();

        // Create generation config
        const config = createGenerationConfig();

        // Generate text
        const result = await inference.generate_async(prompt, config);

        const endTime = performance.now();
        const generationTime = endTime - startTime;

        // Update output
        document.getElementById('output').textContent = result;

        // Update stats
        document.getElementById('generation-time').textContent = `${generationTime.toFixed(0)}ms`;
        document.getElementById('memory-usage').textContent = MemoryUtils.format_bytes(model.get_memory_usage());

        // Estimate tokens per second (rough approximation)
        const estimatedTokens = result.split(' ').length;
        const tokensPerSec = (estimatedTokens / (generationTime / 1000)).toFixed(1);
        document.getElementById('tokens-per-sec').textContent = tokensPerSec;

        updateStatus('Text generated successfully!', 'success');
        Logger.info(`Generated ${estimatedTokens} tokens in ${generationTime.toFixed(0)}ms`);

    } catch (error) {
        updateStatus(`Generation failed: ${error.message}`, 'error');
        Logger.error(`Generation error: ${error}`);
    }
}

// Start streaming generation
async function startStreaming() {
    if (!inference) {
        updateStatus('Please load a model first', 'error');
        return;
    }

    try {
        const prompt = document.getElementById('streaming-prompt').value;
        if (!prompt.trim()) {
            updateStatus('Please enter a prompt', 'error');
            return;
        }

        streamingActive = true;
        document.getElementById('start-streaming').disabled = true;
        document.getElementById('stop-streaming').disabled = false;

        const outputEl = document.getElementById('streaming-output');
        outputEl.textContent = '';

        // Create streaming config
        const config = createGenerationConfig();
        config.set_streaming(true);

        // Create stream
        const stream = inference.generate_stream(prompt, config);
        const iterator = stream.to_async_iterator();

        let tokenCount = 0;
        const startTime = performance.now();

        // Process stream
        while (streamingActive) {
            const result = await iterator.next();

            if (result.done) {
                break;
            }

            // Append token to output
            outputEl.textContent += result.value;
            outputEl.scrollTop = outputEl.scrollHeight;

            // Update stats
            tokenCount++;
            const elapsedTime = (performance.now() - startTime) / 1000;
            const tokensPerSec = (tokenCount / elapsedTime).toFixed(1);

            document.getElementById('stream-tokens').textContent = tokenCount;
            document.getElementById('stream-speed').textContent = tokensPerSec;
            document.getElementById('stream-time').textContent = elapsedTime.toFixed(1);

            // Small delay to make streaming visible
            await new Promise(resolve => setTimeout(resolve, 50));
        }

        streamingActive = false;
        document.getElementById('start-streaming').disabled = false;
        document.getElementById('stop-streaming').disabled = true;

        updateStatus('Streaming completed!', 'success');
        Logger.info(`Streaming completed: ${tokenCount} tokens generated`);

    } catch (error) {
        streamingActive = false;
        document.getElementById('start-streaming').disabled = false;
        document.getElementById('stop-streaming').disabled = true;
        updateStatus(`Streaming failed: ${error.message}`, 'error');
        Logger.error(`Streaming error: ${error}`);
    }
}

// Stop streaming
function stopStreaming() {
    streamingActive = false;
    document.getElementById('start-streaming').disabled = false;
    document.getElementById('stop-streaming').disabled = true;
    updateStatus('Streaming stopped', 'success');
}

// Clear streaming output
function clearStreaming() {
    document.getElementById('streaming-output').textContent = 'Streaming output will appear here...';
    document.getElementById('stream-tokens').textContent = '0';
    document.getElementById('stream-speed').textContent = '0';
    document.getElementById('stream-time').textContent = '0';
}

// Initialize Web Worker
function initWorker() {
    try {
        worker = new Worker('worker.js');

        worker.onmessage = function(e) {
            const { type, data, error } = e.data;

            switch (type) {
                case 'initialized':
                    document.getElementById('worker-indicator').classList.add('active');
                    document.getElementById('worker-status-text').textContent = 'Worker initialized';
                    document.getElementById('worker-generate').disabled = false;
                    document.getElementById('terminate-worker').disabled = false;
                    Logger.info('Web Worker initialized successfully');
                    break;

                case 'generation_complete':
                    document.getElementById('worker-output').textContent = data;
                    updateStatus('Worker generation completed!', 'success');
                    break;

                case 'error':
                    updateStatus(`Worker error: ${error}`, 'error');
                    Logger.error(`Worker error: ${error}`);
                    break;
            }
        };

        worker.onerror = function(error) {
            updateStatus(`Worker initialization failed: ${error.message}`, 'error');
            Logger.error(`Worker error: ${error}`);
        };

        // Initialize worker
        worker.postMessage({ type: 'init' });

    } catch (error) {
        updateStatus(`Worker creation failed: ${error.message}`, 'error');
        Logger.error(`Worker creation error: ${error}`);
    }
}

// Generate text in worker
function workerGenerate() {
    if (!worker) {
        updateStatus('Please initialize worker first', 'error');
        return;
    }

    const prompt = document.getElementById('worker-prompt').value;
    if (!prompt.trim()) {
        updateStatus('Please enter a prompt', 'error');
        return;
    }

    updateStatus('Generating in worker...', 'loading');
    worker.postMessage({
        type: 'generate',
        prompt: prompt,
        config: createGenerationConfig()
    });
}

// Terminate worker
function terminateWorker() {
    if (worker) {
        worker.terminate();
        worker = null;

        document.getElementById('worker-indicator').classList.remove('active');
        document.getElementById('worker-status-text').textContent = 'Worker terminated';
        document.getElementById('worker-generate').disabled = true;
        document.getElementById('terminate-worker').disabled = true;

        Logger.info('Web Worker terminated');
    }
}

// Run all benchmarks
async function runBenchmarks() {
    if (!benchmarkSuite) {
        updateStatus('Benchmark suite not initialized', 'error');
        return;
    }

    try {
        updateStatus('Running comprehensive benchmarks...', 'loading');
        showBenchmarkProgress(true);

        const results = await benchmarkSuite.run_all_benchmarks();

        // Display results
        document.getElementById('benchmark-output').textContent = JSON.stringify(results, null, 2);

        // Update platform stats
        if (results.platform_info) {
            document.getElementById('platform-gflops').textContent =
                results.platform_info.estimated_gflops.toFixed(2);
        }

        showBenchmarkProgress(false);
        updateStatus('Benchmarks completed successfully!', 'success');
        Logger.info('Comprehensive benchmarks completed');

    } catch (error) {
        showBenchmarkProgress(false);
        updateStatus(`Benchmark failed: ${error.message}`, 'error');
        Logger.error(`Benchmark error: ${error}`);
    }
}

// Run individual benchmark categories
async function runKernelBenchmark() {
    try {
        updateStatus('Running kernel benchmarks...', 'loading');
        const results = await benchmarkSuite.benchmark_kernels();
        document.getElementById('benchmark-output').textContent = JSON.stringify(results, null, 2);
        updateStatus('Kernel benchmarks completed!', 'success');
    } catch (error) {
        updateStatus(`Kernel benchmark failed: ${error.message}`, 'error');
    }
}

async function runMemoryBenchmark() {
    try {
        updateStatus('Running memory benchmarks...', 'loading');
        const results = await benchmarkSuite.benchmark_memory();
        document.getElementById('benchmark-output').textContent = JSON.stringify(results, null, 2);
        updateStatus('Memory benchmarks completed!', 'success');
    } catch (error) {
        updateStatus(`Memory benchmark failed: ${error.message}`, 'error');
    }
}

async function runLoadingBenchmark() {
    try {
        updateStatus('Running loading benchmarks...', 'loading');
        const results = await benchmarkSuite.benchmark_loading();
        document.getElementById('benchmark-output').textContent = JSON.stringify(results, null, 2);
        updateStatus('Loading benchmarks completed!', 'success');
    } catch (error) {
        updateStatus(`Loading benchmark failed: ${error.message}`, 'error');
    }
}

// Show/hide benchmark progress
function showBenchmarkProgress(show) {
    const progressEl = document.getElementById('benchmark-progress');
    if (show) {
        progressEl.style.display = 'block';
        // Animate progress bar
        let progress = 0;
        const interval = setInterval(() => {
            progress += 2;
            document.getElementById('benchmark-progress-fill').style.width = `${progress}%`;
            if (progress >= 100) {
                clearInterval(interval);
            }
        }, 100);
    } else {
        progressEl.style.display = 'none';
    }
}

// Create generation configuration from UI settings
function createGenerationConfig() {
    const config = new WasmGenerationConfig();
    config.set_max_new_tokens(parseInt(document.getElementById('max-tokens').value));
    config.set_temperature(parseFloat(document.getElementById('temperature').value));
    config.set_top_k(parseInt(document.getElementById('top-k').value));
    config.set_top_p(parseFloat(document.getElementById('top-p').value));
    return config;
}

// Tab switching
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
        tab.setAttribute('aria-selected', 'false');
        tab.setAttribute('tabindex', '-1');
    });

    // Show selected tab content
    document.getElementById(`${tabName}-tab`).classList.add('active');

    // Add active class and A11y attributes to selected tab
    const selectedTab = document.getElementById(`tab-${tabName}`);
    if (selectedTab) {
        selectedTab.classList.add('active');
        selectedTab.setAttribute('aria-selected', 'true');
        selectedTab.setAttribute('tabindex', '0');
        // Ensure focus follows if this wasn't a mouse click
        if (document.activeElement !== selectedTab) {
            selectedTab.focus();
        }
    }
}

// Handle keyboard navigation for tabs
function handleTabKeydown(e) {
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    const index = tabs.indexOf(e.target);

    let nextTab = null;

    switch (e.key) {
        case 'ArrowRight':
            nextTab = tabs[(index + 1) % tabs.length];
            break;
        case 'ArrowLeft':
            nextTab = tabs[(index - 1 + tabs.length) % tabs.length];
            break;
        case 'Home':
            nextTab = tabs[0];
            break;
        case 'End':
            nextTab = tabs[tabs.length - 1];
            break;
    }

    if (nextTab) {
        e.preventDefault();
        // Trigger the click to switch tabs
        nextTab.click();
        nextTab.focus();
    }
}

// Settings management
function loadSettings() {
    const settings = JSON.parse(localStorage.getItem('bitnet-wasm-settings') || '{}');

    if (settings.maxTokens) document.getElementById('max-tokens').value = settings.maxTokens;
    if (settings.temperature) {
        document.getElementById('temperature').value = settings.temperature;
        document.getElementById('temperature-value').textContent = settings.temperature;
    }
    if (settings.topK) document.getElementById('top-k').value = settings.topK;
    if (settings.topP) {
        document.getElementById('top-p').value = settings.topP;
        document.getElementById('top-p-value').textContent = settings.topP;
    }
    if (settings.memoryLimit) document.getElementById('memory-limit').value = settings.memoryLimit;
    if (settings.chunkSize) document.getElementById('chunk-size').value = settings.chunkSize;
}

function saveSettings() {
    const settings = {
        maxTokens: document.getElementById('max-tokens').value,
        temperature: document.getElementById('temperature').value,
        topK: document.getElementById('top-k').value,
        topP: document.getElementById('top-p').value,
        memoryLimit: document.getElementById('memory-limit').value,
        chunkSize: document.getElementById('chunk-size').value
    };

    localStorage.setItem('bitnet-wasm-settings', JSON.stringify(settings));
    updateStatus('Settings saved!', 'success');
}

function resetSettings() {
    localStorage.removeItem('bitnet-wasm-settings');
    location.reload();
}

function exportSettings() {
    const settings = localStorage.getItem('bitnet-wasm-settings') || '{}';
    const blob = new Blob([settings], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bitnet-wasm-settings.json';
    a.click();
    URL.revokeObjectURL(url);
}

function importSettings() {
    const file = document.getElementById('import-settings').files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const settings = JSON.parse(e.target.result);
            localStorage.setItem('bitnet-wasm-settings', JSON.stringify(settings));
            location.reload();
        } catch (error) {
            updateStatus('Invalid settings file', 'error');
        }
    };
    reader.readAsText(file);
}

// Utility functions
function clearOutput() {
    document.getElementById('output').textContent = 'Generated text will appear here...';
    document.getElementById('generation-time').textContent = '-';
    document.getElementById('tokens-per-sec').textContent = '-';
}

// Event listeners for range inputs
document.getElementById('temperature').addEventListener('input', function() {
    document.getElementById('temperature-value').textContent = this.value;
});

document.getElementById('top-p').addEventListener('input', function() {
    document.getElementById('top-p-value').textContent = this.value;
});

// Make functions globally available
window.loadModel = loadModel;
window.generateText = generateText;
window.startStreaming = startStreaming;
window.stopStreaming = stopStreaming;
window.clearStreaming = clearStreaming;
window.clearOutput = clearOutput;
window.initWorker = initWorker;
window.workerGenerate = workerGenerate;
window.terminateWorker = terminateWorker;
window.runBenchmarks = runBenchmarks;
window.runKernelBenchmark = runKernelBenchmark;
window.runMemoryBenchmark = runMemoryBenchmark;
window.runLoadingBenchmark = runLoadingBenchmark;
window.switchTab = switchTab;
window.saveSettings = saveSettings;
window.resetSettings = resetSettings;
window.exportSettings = exportSettings;
window.importSettings = importSettings;

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    // Initialize tab keyboard accessibility
    document.querySelectorAll('[role="tab"]').forEach(tab => {
        tab.addEventListener('keydown', handleTabKeydown);
    });
});
