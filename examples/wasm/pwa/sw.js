// BitNet WASM Service Worker
// Provides offline caching and background model loading

const CACHE_NAME = 'bitnet-wasm-v1.0.0';
const STATIC_CACHE_NAME = 'bitnet-wasm-static-v1.0.0';
const MODEL_CACHE_NAME = 'bitnet-wasm-models-v1.0.0';

// Files to cache for offline use
const STATIC_FILES = [
    '/',
    '/index.html',
    '/main.js',
    '/worker.js',
    '/manifest.json',
    '/pkg/bitnet_wasm.js',
    '/pkg/bitnet_wasm_bg.wasm',
    '/icons/icon-192x192.png',
    '/icons/icon-512x512.png',
    // Add other static assets
];

// Model files that can be cached (user-uploaded models)
const MODEL_PATTERNS = [
    /\.gguf$/,
    /\.safetensors$/,
    /\.bin$/,
    /\.bitnet$/
];

// Install event - cache static files
self.addEventListener('install', event => {
    console.log('BitNet WASM Service Worker installing...');

    event.waitUntil(
        Promise.all([
            // Cache static files
            caches.open(STATIC_CACHE_NAME).then(cache => {
                console.log('Caching static files...');
                return cache.addAll(STATIC_FILES);
            }),

            // Initialize model cache
            caches.open(MODEL_CACHE_NAME).then(cache => {
                console.log('Model cache initialized');
                return Promise.resolve();
            })
        ]).then(() => {
            console.log('BitNet WASM Service Worker installed successfully');
            // Force activation of new service worker
            return self.skipWaiting();
        }).catch(error => {
            console.error('Service Worker installation failed:', error);
        })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
    console.log('BitNet WASM Service Worker activating...');

    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    // Delete old caches
                    if (cacheName !== STATIC_CACHE_NAME &&
                        cacheName !== MODEL_CACHE_NAME &&
                        cacheName.startsWith('bitnet-wasm-')) {
                        console.log('Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => {
            console.log('BitNet WASM Service Worker activated');
            // Take control of all clients immediately
            return self.clients.claim();
        })
    );
});

// Fetch event - serve cached content and cache new content
self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);

    // Handle different types of requests
    if (isStaticFile(url.pathname)) {
        event.respondWith(handleStaticFile(event.request));
    } else if (isModelFile(url.pathname)) {
        event.respondWith(handleModelFile(event.request));
    } else if (isAPIRequest(url.pathname)) {
        event.respondWith(handleAPIRequest(event.request));
    } else {
        // Default network-first strategy
        event.respondWith(handleDefault(event.request));
    }
});

// Handle static files (cache-first strategy)
async function handleStaticFile(request) {
    try {
        const cache = await caches.open(STATIC_CACHE_NAME);
        const cachedResponse = await cache.match(request);

        if (cachedResponse) {
            console.log('Serving static file from cache:', request.url);
            return cachedResponse;
        }

        // Fetch from network and cache
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            cache.put(request, networkResponse.clone());
        }

        return networkResponse;

    } catch (error) {
        console.error('Error handling static file:', error);

        // Return offline fallback if available
        if (request.url.endsWith('.html')) {
            return caches.match('/offline.html') || new Response('Offline', { status: 503 });
        }

        return new Response('Network Error', { status: 503 });
    }
}

// Handle model files (cache-first with size limits)
async function handleModelFile(request) {
    try {
        const cache = await caches.open(MODEL_CACHE_NAME);
        const cachedResponse = await cache.match(request);

        if (cachedResponse) {
            console.log('Serving model from cache:', request.url);
            return cachedResponse;
        }

        // Fetch from network
        const networkResponse = await fetch(request);

        if (networkResponse.ok) {
            // Check model size before caching
            const contentLength = networkResponse.headers.get('content-length');
            const modelSize = contentLength ? parseInt(contentLength) : 0;

            // Only cache models smaller than 500MB
            if (modelSize > 0 && modelSize < 500 * 1024 * 1024) {
                console.log(`Caching model (${formatBytes(modelSize)}):`, request.url);

                // Ensure we don't exceed storage quota
                await manageModelCacheSize(cache, modelSize);
                cache.put(request, networkResponse.clone());
            } else {
                console.log('Model too large to cache:', request.url);
            }
        }

        return networkResponse;

    } catch (error) {
        console.error('Error handling model file:', error);
        return new Response('Model Loading Error', { status: 503 });
    }
}

// Handle API requests (network-first with offline fallback)
async function handleAPIRequest(request) {
    try {
        // Try network first
        const networkResponse = await fetch(request);

        if (networkResponse.ok) {
            // Cache successful API responses for offline use
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, networkResponse.clone());
        }

        return networkResponse;

    } catch (error) {
        console.log('Network failed, trying cache for API request:', request.url);

        // Fallback to cache
        const cache = await caches.open(CACHE_NAME);
        const cachedResponse = await cache.match(request);

        if (cachedResponse) {
            return cachedResponse;
        }

        // Return offline response
        return new Response(JSON.stringify({
            error: 'Offline',
            message: 'This request requires an internet connection'
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

// Default handler (network-first)
async function handleDefault(request) {
    try {
        const networkResponse = await fetch(request);
        return networkResponse;
    } catch (error) {
        // Try cache as fallback
        const cache = await caches.open(CACHE_NAME);
        const cachedResponse = await cache.match(request);

        if (cachedResponse) {
            return cachedResponse;
        }

        return new Response('Offline', { status: 503 });
    }
}

// Manage model cache size to prevent storage quota issues
async function manageModelCacheSize(cache, newModelSize) {
    try {
        // Get current cache size
        const cacheKeys = await cache.keys();
        let totalSize = 0;

        for (const request of cacheKeys) {
            const response = await cache.match(request);
            if (response) {
                const size = response.headers.get('content-length');
                if (size) {
                    totalSize += parseInt(size);
                }
            }
        }

        // Maximum cache size: 1GB
        const maxCacheSize = 1024 * 1024 * 1024;

        // If adding new model would exceed limit, remove oldest models
        if (totalSize + newModelSize > maxCacheSize) {
            console.log('Model cache size limit reached, cleaning up...');

            // Sort by last accessed time (simplified - in practice you'd track this)
            const sortedKeys = cacheKeys.sort((a, b) => {
                // Simple heuristic: sort by URL (older models likely have different naming)
                return a.url.localeCompare(b.url);
            });

            // Remove oldest models until we have enough space
            for (const request of sortedKeys) {
                if (totalSize + newModelSize <= maxCacheSize) {
                    break;
                }

                const response = await cache.match(request);
                if (response) {
                    const size = response.headers.get('content-length');
                    if (size) {
                        totalSize -= parseInt(size);
                        await cache.delete(request);
                        console.log('Removed cached model:', request.url);
                    }
                }
            }
        }

    } catch (error) {
        console.error('Error managing model cache size:', error);
    }
}

// Utility functions
function isStaticFile(pathname) {
    return STATIC_FILES.some(file => pathname === file || pathname.endsWith(file)) ||
           pathname.endsWith('.js') ||
           pathname.endsWith('.wasm') ||
           pathname.endsWith('.css') ||
           pathname.endsWith('.html') ||
           pathname.endsWith('.png') ||
           pathname.endsWith('.ico');
}

function isModelFile(pathname) {
    return MODEL_PATTERNS.some(pattern => pattern.test(pathname));
}

function isAPIRequest(pathname) {
    return pathname.startsWith('/api/') ||
           pathname.includes('inference') ||
           pathname.includes('generate');
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Background sync for model preloading
self.addEventListener('sync', event => {
    if (event.tag === 'preload-model') {
        event.waitUntil(preloadPopularModels());
    }
});

// Preload popular models in the background
async function preloadPopularModels() {
    try {
        console.log('Background sync: Preloading popular models...');

        // List of popular small models to preload
        const popularModels = [
            '/models/bitnet-small-1b.gguf',
            '/models/bitnet-chat-3b.gguf'
        ];

        const cache = await caches.open(MODEL_CACHE_NAME);

        for (const modelUrl of popularModels) {
            try {
                const cachedResponse = await cache.match(modelUrl);
                if (!cachedResponse) {
                    console.log('Preloading model:', modelUrl);
                    const response = await fetch(modelUrl);
                    if (response.ok) {
                        await cache.put(modelUrl, response);
                        console.log('Preloaded model:', modelUrl);
                    }
                }
            } catch (error) {
                console.log('Failed to preload model:', modelUrl, error);
            }
        }

    } catch (error) {
        console.error('Background model preloading failed:', error);
    }
}

// Handle push notifications for model updates
self.addEventListener('push', event => {
    if (event.data) {
        const data = event.data.json();

        if (data.type === 'model-update') {
            event.waitUntil(handleModelUpdate(data));
        }
    }
});

// Handle model update notifications
async function handleModelUpdate(data) {
    try {
        console.log('Received model update notification:', data);

        // Show notification to user
        await self.registration.showNotification('BitNet Model Update', {
            body: `New model available: ${data.modelName}`,
            icon: '/icons/icon-192x192.png',
            badge: '/icons/badge-72x72.png',
            actions: [
                {
                    action: 'download',
                    title: 'Download Now'
                },
                {
                    action: 'later',
                    title: 'Later'
                }
            ],
            data: data
        });

    } catch (error) {
        console.error('Error handling model update:', error);
    }
}

// Handle notification clicks
self.addEventListener('notificationclick', event => {
    event.notification.close();

    if (event.action === 'download') {
        // Open app and start model download
        event.waitUntil(
            clients.openWindow(`/?download=${encodeURIComponent(event.notification.data.modelUrl)}`)
        );
    } else if (event.action === 'later') {
        // Schedule background sync for later
        self.registration.sync.register('preload-model');
    } else {
        // Default action - open app
        event.waitUntil(clients.openWindow('/'));
    }
});

// Handle messages from main thread
self.addEventListener('message', event => {
    const { type, data } = event.data;

    switch (type) {
        case 'CACHE_MODEL':
            event.waitUntil(cacheModel(data.url, data.data));
            break;

        case 'GET_CACHE_SIZE':
            event.waitUntil(getCacheSize().then(size => {
                event.ports[0].postMessage({ type: 'CACHE_SIZE', size });
            }));
            break;

        case 'CLEAR_MODEL_CACHE':
            event.waitUntil(clearModelCache().then(() => {
                event.ports[0].postMessage({ type: 'CACHE_CLEARED' });
            }));
            break;

        default:
            console.log('Unknown message type:', type);
    }
});

// Cache a model manually
async function cacheModel(url, modelData) {
    try {
        const cache = await caches.open(MODEL_CACHE_NAME);
        const response = new Response(modelData, {
            headers: {
                'Content-Type': 'application/octet-stream',
                'Content-Length': modelData.byteLength.toString()
            }
        });

        await cache.put(url, response);
        console.log('Model cached manually:', url);

    } catch (error) {
        console.error('Error caching model:', error);
    }
}

// Get total cache size
async function getCacheSize() {
    try {
        const cacheNames = await caches.keys();
        let totalSize = 0;

        for (const cacheName of cacheNames) {
            if (cacheName.startsWith('bitnet-wasm-')) {
                const cache = await caches.open(cacheName);
                const keys = await cache.keys();

                for (const request of keys) {
                    const response = await cache.match(request);
                    if (response) {
                        const size = response.headers.get('content-length');
                        if (size) {
                            totalSize += parseInt(size);
                        }
                    }
                }
            }
        }

        return totalSize;

    } catch (error) {
        console.error('Error calculating cache size:', error);
        return 0;
    }
}

// Clear model cache
async function clearModelCache() {
    try {
        const cache = await caches.open(MODEL_CACHE_NAME);
        const keys = await cache.keys();

        for (const request of keys) {
            await cache.delete(request);
        }

        console.log('Model cache cleared');

    } catch (error) {
        console.error('Error clearing model cache:', error);
    }
}

console.log('BitNet WASM Service Worker loaded');
