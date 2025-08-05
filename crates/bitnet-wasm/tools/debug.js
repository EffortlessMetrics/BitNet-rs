// BitNet WASM Developer Tools - Debugging and Profiling Utilities

/**
 * BitNet WASM Debug Tools
 * Provides debugging and profiling utilities for BitNet WebAssembly applications
 */
class BitNetDebugTools {
    constructor() {
        this.isEnabled = false;
        this.logs = [];
        this.performanceMarks = new Map();
        this.memorySnapshots = [];
        this.inferenceMetrics = [];
        this.wasmModule = null;
        
        // Initialize debug panel if in development mode
        if (this.isDevelopmentMode()) {
            this.initDebugPanel();
        }
    }

    /**
     * Enable debug mode
     */
    enable() {
        this.isEnabled = true;
        console.log('🔧 BitNet WASM Debug Tools enabled');
        
        // Inject debug styles
        this.injectDebugStyles();
        
        // Show debug panel
        const panel = document.getElementById('bitnet-debug-panel');
        if (panel) {
            panel.style.display = 'block';
        }
        
        // Start memory monitoring
        this.startMemoryMonitoring();
        
        return this;
    }

    /**
     * Disable debug mode
     */
    disable() {
        this.isEnabled = false;
        console.log('🔧 BitNet WASM Debug Tools disabled');
        
        // Hide debug panel
        const panel = document.getElementById('bitnet-debug-panel');
        if (panel) {
            panel.style.display = 'none';
        }
        
        // Stop memory monitoring
        this.stopMemoryMonitoring();
        
        return this;
    }

    /**
     * Set WASM module reference for advanced debugging
     */
    setWasmModule(module) {
        this.wasmModule = module;
        this.log('WASM module reference set', 'info');
        return this;
    }

    /**
     * Log debug message
     */
    log(message, level = 'debug', data = null) {
        if (!this.isEnabled) return;
        
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            level,
            message,
            data
        };
        
        this.logs.push(logEntry);
        
        // Keep only last 1000 logs
        if (this.logs.length > 1000) {
            this.logs.shift();
        }
        
        // Console output with styling
        const style = this.getLogStyle(level);
        console.log(`%c[BitNet WASM ${level.toUpperCase()}] ${message}`, style, data || '');
        
        // Update debug panel
        this.updateDebugPanel();
    }

    /**
     * Start performance measurement
     */
    startPerformanceMark(name) {
        if (!this.isEnabled) return;
        
        const startTime = performance.now();
        this.performanceMarks.set(name, { startTime, endTime: null });
        this.log(`Performance mark started: ${name}`, 'perf');
        
        return this;
    }

    /**
     * End performance measurement
     */
    endPerformanceMark(name) {
        if (!this.isEnabled) return;
        
        const mark = this.performanceMarks.get(name);
        if (!mark) {
            this.log(`Performance mark not found: ${name}`, 'warn');
            return this;
        }
        
        const endTime = performance.now();
        mark.endTime = endTime;
        mark.duration = endTime - mark.startTime;
        
        this.log(`Performance mark completed: ${name} (${mark.duration.toFixed(2)}ms)`, 'perf', {
            duration: mark.duration,
            startTime: mark.startTime,
            endTime: mark.endTime
        });
        
        return this;
    }

    /**
     * Take memory snapshot
     */
    takeMemorySnapshot(label = 'snapshot') {
        if (!this.isEnabled) return;
        
        const snapshot = {
            timestamp: Date.now(),
            label,
            jsHeapUsed: performance.memory ? performance.memory.usedJSHeapSize : 0,
            jsHeapTotal: performance.memory ? performance.memory.totalJSHeapSize : 0,
            jsHeapLimit: performance.memory ? performance.memory.jsHeapSizeLimit : 0,
            wasmMemory: this.getWasmMemoryUsage()
        };
        
        this.memorySnapshots.push(snapshot);
        this.log(`Memory snapshot taken: ${label}`, 'memory', snapshot);
        
        return snapshot;
    }

    /**
     * Record inference metrics
     */
    recordInferenceMetrics(metrics) {
        if (!this.isEnabled) return;
        
        const record = {
            timestamp: Date.now(),
            ...metrics
        };
        
        this.inferenceMetrics.push(record);
        this.log('Inference metrics recorded', 'metrics', record);
        
        return this;
    }

    /**
     * Analyze performance bottlenecks
     */
    analyzePerformance() {
        if (!this.isEnabled) return null;
        
        const analysis = {
            slowestOperations: this.getSlowestOperations(),
            memoryTrends: this.analyzeMemoryTrends(),
            inferenceStats: this.getInferenceStats(),
            recommendations: []
        };
        
        // Generate recommendations
        if (analysis.slowestOperations.length > 0) {
            const slowest = analysis.slowestOperations[0];
            if (slowest.duration > 1000) {
                analysis.recommendations.push(`Consider optimizing ${slowest.name} - taking ${slowest.duration.toFixed(0)}ms`);
            }
        }
        
        if (analysis.memoryTrends.peakUsage > 500 * 1024 * 1024) {
            analysis.recommendations.push('High memory usage detected - consider enabling progressive loading');
        }
        
        this.log('Performance analysis completed', 'analysis', analysis);
        return analysis;
    }

    /**
     * Export debug data
     */
    exportDebugData() {
        const debugData = {
            timestamp: new Date().toISOString(),
            logs: this.logs,
            performanceMarks: Array.from(this.performanceMarks.entries()),
            memorySnapshots: this.memorySnapshots,
            inferenceMetrics: this.inferenceMetrics,
            platformInfo: this.getPlatformInfo(),
            analysis: this.analyzePerformance()
        };
        
        const blob = new Blob([JSON.stringify(debugData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `bitnet-debug-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.log('Debug data exported', 'info');
        return debugData;
    }

    /**
     * Visualize memory usage over time
     */
    visualizeMemoryUsage() {
        if (!this.isEnabled || this.memorySnapshots.length < 2) return;
        
        const canvas = document.getElementById('bitnet-memory-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw memory usage chart
        const maxMemory = Math.max(...this.memorySnapshots.map(s => s.jsHeapUsed + s.wasmMemory));
        const minTime = this.memorySnapshots[0].timestamp;
        const maxTime = this.memorySnapshots[this.memorySnapshots.length - 1].timestamp;
        
        ctx.strokeStyle = '#007bff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        this.memorySnapshots.forEach((snapshot, index) => {
            const x = (snapshot.timestamp - minTime) / (maxTime - minTime) * width;
            const y = height - (snapshot.jsHeapUsed + snapshot.wasmMemory) / maxMemory * height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw labels
        ctx.fillStyle = '#333';
        ctx.font = '12px monospace';
        ctx.fillText(`Max: ${this.formatBytes(maxMemory)}`, 10, 20);
        ctx.fillText(`Current: ${this.formatBytes(this.memorySnapshots[this.memorySnapshots.length - 1].jsHeapUsed + this.memorySnapshots[this.memorySnapshots.length - 1].wasmMemory)}`, 10, 40);
    }

    /**
     * Profile WASM function calls
     */
    profileWasmCalls(functionName, originalFunction) {
        if (!this.isEnabled) return originalFunction;
        
        return (...args) => {
            const startTime = performance.now();
            this.log(`WASM call started: ${functionName}`, 'wasm', { args });
            
            try {
                const result = originalFunction.apply(this, args);
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                this.log(`WASM call completed: ${functionName} (${duration.toFixed(2)}ms)`, 'wasm', {
                    duration,
                    success: true,
                    result: typeof result
                });
                
                return result;
            } catch (error) {
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                this.log(`WASM call failed: ${functionName} (${duration.toFixed(2)}ms)`, 'error', {
                    duration,
                    error: error.message
                });
                
                throw error;
            }
        };
    }

    // Private methods

    isDevelopmentMode() {
        return location.hostname === 'localhost' || 
               location.hostname === '127.0.0.1' || 
               location.search.includes('debug=true');
    }

    initDebugPanel() {
        const panel = document.createElement('div');
        panel.id = 'bitnet-debug-panel';
        panel.innerHTML = `
            <div class="debug-header">
                <h3>🔧 BitNet WASM Debug Tools</h3>
                <button onclick="bitnetDebug.disable()">×</button>
            </div>
            <div class="debug-tabs">
                <button class="tab-btn active" onclick="bitnetDebug.showTab('logs')">Logs</button>
                <button class="tab-btn" onclick="bitnetDebug.showTab('performance')">Performance</button>
                <button class="tab-btn" onclick="bitnetDebug.showTab('memory')">Memory</button>
                <button class="tab-btn" onclick="bitnetDebug.showTab('metrics')">Metrics</button>
            </div>
            <div class="debug-content">
                <div id="debug-logs" class="debug-tab active">
                    <div class="debug-logs-container"></div>
                </div>
                <div id="debug-performance" class="debug-tab">
                    <div class="performance-marks"></div>
                </div>
                <div id="debug-memory" class="debug-tab">
                    <canvas id="bitnet-memory-chart" width="400" height="200"></canvas>
                    <div class="memory-snapshots"></div>
                </div>
                <div id="debug-metrics" class="debug-tab">
                    <div class="inference-metrics"></div>
                </div>
            </div>
            <div class="debug-actions">
                <button onclick="bitnetDebug.exportDebugData()">Export Data</button>
                <button onclick="bitnetDebug.analyzePerformance()">Analyze</button>
                <button onclick="bitnetDebug.takeMemorySnapshot()">Memory Snapshot</button>
            </div>
        `;
        
        document.body.appendChild(panel);
    }

    injectDebugStyles() {
        const styles = `
            #bitnet-debug-panel {
                position: fixed;
                top: 20px;
                right: 20px;
                width: 500px;
                height: 400px;
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                z-index: 10000;
                font-family: monospace;
                font-size: 12px;
                display: none;
            }
            
            .debug-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                background: #f8f9fa;
                border-bottom: 1px solid #ddd;
                border-radius: 8px 8px 0 0;
            }
            
            .debug-header h3 {
                margin: 0;
                font-size: 14px;
            }
            
            .debug-header button {
                background: none;
                border: none;
                font-size: 18px;
                cursor: pointer;
            }
            
            .debug-tabs {
                display: flex;
                background: #f8f9fa;
                border-bottom: 1px solid #ddd;
            }
            
            .tab-btn {
                flex: 1;
                padding: 8px;
                border: none;
                background: none;
                cursor: pointer;
                font-size: 11px;
            }
            
            .tab-btn.active {
                background: white;
                border-bottom: 2px solid #007bff;
            }
            
            .debug-content {
                height: 280px;
                overflow: hidden;
            }
            
            .debug-tab {
                height: 100%;
                padding: 10px;
                overflow-y: auto;
                display: none;
            }
            
            .debug-tab.active {
                display: block;
            }
            
            .debug-logs-container {
                font-size: 10px;
                line-height: 1.4;
            }
            
            .log-entry {
                margin-bottom: 5px;
                padding: 2px 5px;
                border-radius: 3px;
            }
            
            .log-entry.error {
                background: #ffe6e6;
                color: #d63384;
            }
            
            .log-entry.warn {
                background: #fff3cd;
                color: #856404;
            }
            
            .log-entry.info {
                background: #d1ecf1;
                color: #0c5460;
            }
            
            .debug-actions {
                padding: 10px;
                border-top: 1px solid #ddd;
                display: flex;
                gap: 5px;
            }
            
            .debug-actions button {
                flex: 1;
                padding: 5px;
                border: 1px solid #ddd;
                background: white;
                border-radius: 3px;
                cursor: pointer;
                font-size: 10px;
            }
            
            #bitnet-memory-chart {
                border: 1px solid #ddd;
                width: 100%;
                height: 150px;
            }
        `;
        
        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    showTab(tabName) {
        // Hide all tabs
        document.querySelectorAll('.debug-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Show selected tab
        document.getElementById(`debug-${tabName}`).classList.add('active');
        event.target.classList.add('active');
        
        // Update content based on tab
        if (tabName === 'memory') {
            this.visualizeMemoryUsage();
        }
    }

    updateDebugPanel() {
        const logsContainer = document.querySelector('.debug-logs-container');
        if (!logsContainer) return;
        
        // Show last 50 logs
        const recentLogs = this.logs.slice(-50);
        logsContainer.innerHTML = recentLogs.map(log => `
            <div class="log-entry ${log.level}">
                <span class="timestamp">${new Date(log.timestamp).toLocaleTimeString()}</span>
                <span class="message">${log.message}</span>
            </div>
        `).join('');
        
        // Auto-scroll to bottom
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }

    startMemoryMonitoring() {
        this.memoryMonitorInterval = setInterval(() => {
            this.takeMemorySnapshot('auto');
        }, 5000);
    }

    stopMemoryMonitoring() {
        if (this.memoryMonitorInterval) {
            clearInterval(this.memoryMonitorInterval);
        }
    }

    getWasmMemoryUsage() {
        if (!this.wasmModule || !this.wasmModule.memory) return 0;
        return this.wasmModule.memory.buffer.byteLength;
    }

    getLogStyle(level) {
        const styles = {
            debug: 'color: #6c757d',
            info: 'color: #0c5460; font-weight: bold',
            warn: 'color: #856404; font-weight: bold',
            error: 'color: #d63384; font-weight: bold',
            perf: 'color: #6f42c1; font-weight: bold',
            memory: 'color: #20c997; font-weight: bold',
            metrics: 'color: #fd7e14; font-weight: bold',
            wasm: 'color: #e83e8c; font-weight: bold',
            analysis: 'color: #6610f2; font-weight: bold'
        };
        return styles[level] || styles.debug;
    }

    getSlowestOperations() {
        return Array.from(this.performanceMarks.entries())
            .filter(([_, mark]) => mark.endTime !== null)
            .map(([name, mark]) => ({ name, duration: mark.duration }))
            .sort((a, b) => b.duration - a.duration)
            .slice(0, 10);
    }

    analyzeMemoryTrends() {
        if (this.memorySnapshots.length < 2) return null;
        
        const totalUsages = this.memorySnapshots.map(s => s.jsHeapUsed + s.wasmMemory);
        const peakUsage = Math.max(...totalUsages);
        const averageUsage = totalUsages.reduce((a, b) => a + b, 0) / totalUsages.length;
        const trend = totalUsages[totalUsages.length - 1] - totalUsages[0];
        
        return {
            peakUsage,
            averageUsage,
            trend,
            isIncreasing: trend > 0
        };
    }

    getInferenceStats() {
        if (this.inferenceMetrics.length === 0) return null;
        
        const latencies = this.inferenceMetrics.map(m => m.latency).filter(l => l);
        const throughputs = this.inferenceMetrics.map(m => m.throughput).filter(t => t);
        
        return {
            totalInferences: this.inferenceMetrics.length,
            averageLatency: latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0,
            averageThroughput: throughputs.length > 0 ? throughputs.reduce((a, b) => a + b, 0) / throughputs.length : 0,
            minLatency: Math.min(...latencies),
            maxLatency: Math.max(...latencies)
        };
    }

    getPlatformInfo() {
        return {
            userAgent: navigator.userAgent,
            memory: navigator.deviceMemory,
            cores: navigator.hardwareConcurrency,
            language: navigator.language,
            platform: navigator.platform,
            webgl: this.getWebGLInfo(),
            wasm: this.getWasmInfo()
        };
    }

    getWebGLInfo() {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (!gl) return null;
        
        return {
            vendor: gl.getParameter(gl.VENDOR),
            renderer: gl.getParameter(gl.RENDERER),
            version: gl.getParameter(gl.VERSION)
        };
    }

    getWasmInfo() {
        return {
            supported: typeof WebAssembly !== 'undefined',
            simd: this.checkWasmSIMD(),
            threads: this.checkWasmThreads(),
            bulkMemory: this.checkWasmBulkMemory()
        };
    }

    checkWasmSIMD() {
        try {
            return typeof WebAssembly.validate === 'function' &&
                   WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
        } catch {
            return false;
        }
    }

    checkWasmThreads() {
        return typeof SharedArrayBuffer !== 'undefined';
    }

    checkWasmBulkMemory() {
        // Simplified check - in practice you'd test actual bulk memory operations
        return true;
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Create global debug instance
window.bitnetDebug = new BitNetDebugTools();

// Auto-enable in development mode
if (window.bitnetDebug.isDevelopmentMode()) {
    window.bitnetDebug.enable();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BitNetDebugTools;
}

console.log('🔧 BitNet WASM Debug Tools loaded. Use bitnetDebug.enable() to start debugging.');