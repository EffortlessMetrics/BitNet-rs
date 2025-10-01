# Issue #251: Production Inference Server API Contracts

## Overview

This document defines the comprehensive API contracts for the BitNet.rs production inference server, including REST endpoints, request/response schemas, error handling, and streaming protocols. All APIs are designed with quantization awareness, device optimization, and production reliability requirements.

## API Version and Compatibility

**API Version**: `v1`
**Content Type**: `application/json`
**Character Encoding**: `UTF-8`
**Authentication**: JWT Bearer Token (for model management endpoints)

## Core Inference APIs

### Synchronous Inference

**Endpoint**: `POST /v1/inference`

**Request Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["prompt"],
  "properties": {
    "prompt": {
      "type": "string",
      "minLength": 1,
      "maxLength": 8192,
      "description": "Input text prompt for inference"
    },
    "max_tokens": {
      "type": "integer",
      "minimum": 1,
      "maximum": 2048,
      "default": 100,
      "description": "Maximum number of tokens to generate"
    },
    "model": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_-]+$",
      "description": "Model identifier (optional, uses default if not specified)"
    },
    "temperature": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 2.0,
      "default": 0.7,
      "description": "Sampling temperature for randomness control"
    },
    "top_p": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 0.9,
      "description": "Nucleus sampling probability threshold"
    },
    "top_k": {
      "type": "integer",
      "minimum": 1,
      "maximum": 1000,
      "default": 50,
      "description": "Top-k sampling parameter"
    },
    "repetition_penalty": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 2.0,
      "default": 1.0,
      "description": "Repetition penalty for reducing repeated tokens"
    },
    "stop_sequences": {
      "type": "array",
      "items": {
        "type": "string",
        "maxLength": 64
      },
      "maxItems": 10,
      "description": "Stop sequences to end generation"
    },
    "seed": {
      "type": "integer",
      "minimum": 0,
      "maximum": 4294967295,
      "description": "Random seed for deterministic generation"
    },
    "quantization_preference": {
      "type": "string",
      "enum": ["auto", "i2s", "tl1", "tl2"],
      "default": "auto",
      "description": "Preferred quantization format"
    },
    "device_preference": {
      "type": "string",
      "enum": ["auto", "cpu", "gpu"],
      "default": "auto",
      "description": "Preferred execution device"
    },
    "priority": {
      "type": "string",
      "enum": ["low", "normal", "high"],
      "default": "normal",
      "description": "Request processing priority"
    }
  },
  "additionalProperties": false
}
```

**Response Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": [
    "text", "tokens_generated", "inference_time_ms",
    "tokens_per_second", "model_id", "request_id"
  ],
  "properties": {
    "text": {
      "type": "string",
      "description": "Generated text response"
    },
    "tokens_generated": {
      "type": "integer",
      "minimum": 0,
      "description": "Number of tokens generated"
    },
    "inference_time_ms": {
      "type": "integer",
      "minimum": 0,
      "description": "Total inference time in milliseconds"
    },
    "tokens_per_second": {
      "type": "number",
      "minimum": 0,
      "description": "Token generation throughput"
    },
    "model_id": {
      "type": "string",
      "description": "Model identifier used for inference"
    },
    "request_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique request identifier"
    },
    "quantization_used": {
      "type": "string",
      "enum": ["i2s", "tl1", "tl2"],
      "description": "Quantization format actually used"
    },
    "device_used": {
      "type": "string",
      "pattern": "^(cpu|cuda:[0-9]+)$",
      "description": "Device used for inference"
    },
    "accuracy_metrics": {
      "type": "object",
      "properties": {
        "quantization_accuracy": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Quantization accuracy vs reference"
        },
        "cross_validation_score": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Cross-validation accuracy score"
        }
      },
      "required": ["quantization_accuracy"]
    },
    "performance_metrics": {
      "type": "object",
      "properties": {
        "batch_size": {
          "type": "integer",
          "minimum": 1,
          "description": "Batch size used for inference"
        },
        "queue_time_ms": {
          "type": "integer",
          "minimum": 0,
          "description": "Time spent in request queue"
        },
        "processing_time_ms": {
          "type": "integer",
          "minimum": 0,
          "description": "Actual processing time"
        },
        "memory_usage_mb": {
          "type": "number",
          "minimum": 0,
          "description": "Peak memory usage during inference"
        }
      }
    }
  },
  "additionalProperties": false
}
```

**Example Request**:
```json
{
  "prompt": "Explain the concept of neural network quantization in simple terms.",
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9,
  "quantization_preference": "i2s",
  "device_preference": "auto"
}
```

**Example Response**:
```json
{
  "text": "Neural network quantization is a technique that reduces the precision of numbers used in neural networks...",
  "tokens_generated": 45,
  "inference_time_ms": 1247,
  "tokens_per_second": 36.1,
  "model_id": "bitnet-2b-i2s",
  "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "quantization_used": "i2s",
  "device_used": "cuda:0",
  "accuracy_metrics": {
    "quantization_accuracy": 0.995,
    "cross_validation_score": 0.992
  },
  "performance_metrics": {
    "batch_size": 4,
    "queue_time_ms": 23,
    "processing_time_ms": 1224,
    "memory_usage_mb": 2847.5
  }
}
```

### Streaming Inference

**Endpoint**: `POST /v1/inference/stream`

**Request Schema**: Same as synchronous inference

**Response Format**: Server-Sent Events (SSE)
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

**Event Types**:

#### Token Event
```
event: token
data: {"type": "token", "text": "Hello", "position": 0, "timestamp": "2023-12-01T10:30:00.123Z"}
```

#### Progress Event
```
event: progress
data: {"type": "progress", "tokens_generated": 15, "tokens_per_second": 32.5, "estimated_remaining_ms": 2340}
```

#### Metrics Event
```
event: metrics
data: {"type": "metrics", "device": "cuda:0", "memory_usage_mb": 2847.5, "quantization_accuracy": 0.995}
```

#### Error Event
```
event: error
data: {"type": "error", "code": "INFERENCE_FAILED", "message": "Model execution failed", "details": {"device": "cuda:0", "error_type": "memory_exhaustion"}}
```

#### Complete Event
```
event: complete
data: {"type": "complete", "total_tokens": 25, "inference_time_ms": 780, "final_accuracy": 0.995, "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"}
```

## Model Management APIs

### Load Model

**Endpoint**: `POST /v1/models/load`
**Authentication**: Required (JWT Bearer Token)

**Request Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["model_path"],
  "properties": {
    "model_path": {
      "type": "string",
      "pattern": "^[/].*\\.gguf$",
      "description": "Absolute path to GGUF model file"
    },
    "model_id": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_-]+$",
      "maxLength": 64,
      "description": "Custom model identifier"
    },
    "validation_config": {
      "type": "object",
      "properties": {
        "enable_cross_validation": {
          "type": "boolean",
          "default": true,
          "description": "Enable cross-validation against C++ reference"
        },
        "min_accuracy": {
          "type": "number",
          "minimum": 0.9,
          "maximum": 1.0,
          "default": 0.99,
          "description": "Minimum required accuracy threshold"
        },
        "validation_samples": {
          "type": "integer",
          "minimum": 10,
          "maximum": 1000,
          "default": 100,
          "description": "Number of validation samples"
        },
        "timeout_seconds": {
          "type": "integer",
          "minimum": 30,
          "maximum": 3600,
          "default": 300,
          "description": "Validation timeout in seconds"
        }
      }
    },
    "device_preference": {
      "type": "string",
      "enum": ["auto", "cpu", "gpu"],
      "default": "auto",
      "description": "Preferred device for model execution"
    },
    "quantization_format": {
      "type": "string",
      "enum": ["auto", "i2s", "tl1", "tl2"],
      "default": "auto",
      "description": "Expected quantization format"
    },
    "cache_config": {
      "type": "object",
      "properties": {
        "enable_kv_cache": {
          "type": "boolean",
          "default": true,
          "description": "Enable key-value cache for performance"
        },
        "cache_size_mb": {
          "type": "integer",
          "minimum": 100,
          "maximum": 8192,
          "default": 1024,
          "description": "KV cache size in MB"
        }
      }
    }
  },
  "additionalProperties": false
}
```

**Response Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["model_id", "status", "load_time_ms"],
  "properties": {
    "model_id": {
      "type": "string",
      "description": "Unique model identifier"
    },
    "status": {
      "type": "string",
      "enum": ["loaded", "failed"],
      "description": "Model loading status"
    },
    "load_time_ms": {
      "type": "integer",
      "minimum": 0,
      "description": "Total loading time in milliseconds"
    },
    "model_metadata": {
      "type": "object",
      "properties": {
        "file_size_bytes": {
          "type": "integer",
          "minimum": 0,
          "description": "Model file size in bytes"
        },
        "parameter_count": {
          "type": "integer",
          "minimum": 0,
          "description": "Total model parameters"
        },
        "quantization_format": {
          "type": "string",
          "enum": ["i2s", "tl1", "tl2"],
          "description": "Detected quantization format"
        },
        "architecture": {
          "type": "string",
          "description": "Model architecture (e.g., bitnet-transformer)"
        },
        "vocab_size": {
          "type": "integer",
          "minimum": 0,
          "description": "Vocabulary size"
        }
      },
      "required": ["quantization_format", "parameter_count"]
    },
    "validation_results": {
      "type": "object",
      "properties": {
        "gguf_validation": {
          "type": "object",
          "properties": {
            "format_valid": {
              "type": "boolean",
              "description": "GGUF format validation result"
            },
            "version": {
              "type": "string",
              "description": "GGUF format version"
            },
            "tensor_count": {
              "type": "integer",
              "minimum": 0,
              "description": "Number of tensors in model"
            }
          },
          "required": ["format_valid"]
        },
        "quantization_validation": {
          "type": "object",
          "properties": {
            "accuracy_score": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "description": "Quantization accuracy vs reference"
            },
            "cross_validation_score": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "description": "Cross-validation accuracy"
            },
            "statistical_significance": {
              "type": "number",
              "description": "P-value for statistical significance"
            }
          },
          "required": ["accuracy_score"]
        },
        "performance_validation": {
          "type": "object",
          "properties": {
            "tokens_per_second": {
              "type": "number",
              "minimum": 0,
              "description": "Benchmark inference throughput"
            },
            "memory_usage_mb": {
              "type": "number",
              "minimum": 0,
              "description": "Peak memory usage during validation"
            },
            "device_compatibility": {
              "type": "array",
              "items": {
                "type": "string",
                "pattern": "^(cpu|cuda:[0-9]+)$"
              },
              "description": "Compatible execution devices"
            }
          }
        }
      }
    },
    "error_details": {
      "type": "object",
      "properties": {
        "error_code": {
          "type": "string",
          "description": "Error code for failed loading"
        },
        "error_message": {
          "type": "string",
          "description": "Human-readable error message"
        },
        "validation_failures": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "validation_type": {
                "type": "string",
                "description": "Type of validation that failed"
              },
              "expected": {
                "type": "string",
                "description": "Expected value or condition"
              },
              "actual": {
                "type": "string",
                "description": "Actual value found"
              }
            }
          }
        }
      }
    }
  },
  "additionalProperties": false
}
```

### Model Hot-Swap

**Endpoint**: `POST /v1/models/swap`
**Authentication**: Required (JWT Bearer Token)

**Request Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["new_model_path", "target_model_id"],
  "properties": {
    "new_model_path": {
      "type": "string",
      "pattern": "^[/].*\\.gguf$",
      "description": "Path to new model file"
    },
    "target_model_id": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_-]+$",
      "description": "ID of model to replace"
    },
    "swap_strategy": {
      "type": "string",
      "enum": ["atomic", "blue_green", "rolling"],
      "default": "atomic",
      "description": "Hot-swap strategy"
    },
    "rollback_on_failure": {
      "type": "boolean",
      "default": true,
      "description": "Automatically rollback on validation failure"
    },
    "validation_timeout_seconds": {
      "type": "integer",
      "minimum": 10,
      "maximum": 300,
      "default": 30,
      "description": "Post-swap validation timeout"
    },
    "health_check_config": {
      "type": "object",
      "properties": {
        "inference_test_prompts": {
          "type": "array",
          "items": {
            "type": "string",
            "maxLength": 256
          },
          "maxItems": 5,
          "description": "Test prompts for health validation"
        },
        "accuracy_threshold": {
          "type": "number",
          "minimum": 0.9,
          "maximum": 1.0,
          "default": 0.99,
          "description": "Minimum accuracy for health check"
        }
      }
    }
  },
  "additionalProperties": false
}
```

**Response Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["swap_id", "status", "total_duration_ms"],
  "properties": {
    "swap_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique swap operation identifier"
    },
    "status": {
      "type": "string",
      "enum": ["completed", "failed", "rolled_back"],
      "description": "Hot-swap operation status"
    },
    "total_duration_ms": {
      "type": "integer",
      "minimum": 0,
      "description": "Total swap operation duration"
    },
    "previous_model_metadata": {
      "type": "object",
      "properties": {
        "model_id": {
          "type": "string",
          "description": "Previous model identifier"
        },
        "quantization_format": {
          "type": "string",
          "description": "Previous model quantization format"
        },
        "performance_baseline": {
          "type": "object",
          "properties": {
            "tokens_per_second": {
              "type": "number",
              "description": "Previous model throughput"
            },
            "accuracy_score": {
              "type": "number",
              "description": "Previous model accuracy"
            }
          }
        }
      }
    },
    "new_model_metadata": {
      "type": "object",
      "description": "New model metadata (same schema as previous_model_metadata)"
    },
    "swap_phases": {
      "type": "object",
      "properties": {
        "validation_duration_ms": {
          "type": "integer",
          "minimum": 0,
          "description": "Model validation phase duration"
        },
        "swap_duration_ms": {
          "type": "integer",
          "minimum": 0,
          "description": "Atomic swap phase duration"
        },
        "health_check_duration_ms": {
          "type": "integer",
          "minimum": 0,
          "description": "Post-swap health check duration"
        }
      }
    },
    "quantization_comparison": {
      "type": "object",
      "properties": {
        "format_changed": {
          "type": "boolean",
          "description": "Whether quantization format changed"
        },
        "accuracy_delta": {
          "type": "number",
          "description": "Accuracy difference between models"
        },
        "performance_delta": {
          "type": "number",
          "description": "Performance difference (positive = improvement)"
        }
      }
    },
    "health_check_results": {
      "type": "object",
      "properties": {
        "overall_healthy": {
          "type": "boolean",
          "description": "Overall health check result"
        },
        "individual_checks": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "check_name": {
                "type": "string",
                "description": "Name of health check"
              },
              "passed": {
                "type": "boolean",
                "description": "Check result"
              },
              "details": {
                "type": "string",
                "description": "Check details or error message"
              }
            }
          }
        }
      }
    },
    "rollback_info": {
      "type": "object",
      "properties": {
        "rollback_performed": {
          "type": "boolean",
          "description": "Whether rollback was performed"
        },
        "rollback_reason": {
          "type": "string",
          "description": "Reason for rollback"
        },
        "rollback_duration_ms": {
          "type": "integer",
          "minimum": 0,
          "description": "Rollback operation duration"
        }
      }
    }
  },
  "additionalProperties": false
}
```

### List Models

**Endpoint**: `GET /v1/models`

**Query Parameters**:
- `status` (optional): Filter by status (`active`, `loading`, `failed`)
- `quantization_format` (optional): Filter by quantization format
- `device` (optional): Filter by execution device

**Response Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["models", "total_count"],
  "properties": {
    "models": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "status", "quantization_format"],
        "properties": {
          "id": {
            "type": "string",
            "description": "Model identifier"
          },
          "status": {
            "type": "string",
            "enum": ["active", "loading", "failed", "unloading"],
            "description": "Current model status"
          },
          "path": {
            "type": "string",
            "description": "Model file path"
          },
          "quantization_format": {
            "type": "string",
            "enum": ["i2s", "tl1", "tl2"],
            "description": "Model quantization format"
          },
          "device": {
            "type": "string",
            "pattern": "^(cpu|cuda:[0-9]+)$",
            "description": "Execution device"
          },
          "load_time": {
            "type": "string",
            "format": "date-time",
            "description": "Model load timestamp"
          },
          "performance_metrics": {
            "type": "object",
            "properties": {
              "avg_tokens_per_second": {
                "type": "number",
                "minimum": 0,
                "description": "Average inference throughput"
              },
              "avg_inference_time_ms": {
                "type": "number",
                "minimum": 0,
                "description": "Average inference time"
              },
              "accuracy_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Model accuracy score"
              },
              "total_requests": {
                "type": "integer",
                "minimum": 0,
                "description": "Total requests processed"
              }
            }
          },
          "resource_usage": {
            "type": "object",
            "properties": {
              "memory_usage_mb": {
                "type": "number",
                "minimum": 0,
                "description": "Current memory usage"
              },
              "gpu_memory_usage_mb": {
                "type": "number",
                "minimum": 0,
                "description": "GPU memory usage (if applicable)"
              }
            }
          }
        }
      }
    },
    "total_count": {
      "type": "integer",
      "minimum": 0,
      "description": "Total number of models"
    },
    "active_count": {
      "type": "integer",
      "minimum": 0,
      "description": "Number of active models"
    }
  },
  "additionalProperties": false
}
```

## Monitoring and Health APIs

### Health Check

**Endpoint**: `GET /health`

**Response Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["status", "timestamp"],
  "properties": {
    "status": {
      "type": "string",
      "enum": ["healthy", "degraded", "unhealthy"],
      "description": "Overall system health status"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "Health check timestamp"
    },
    "components": {
      "type": "object",
      "properties": {
        "model_manager": {
          "type": "string",
          "enum": ["healthy", "degraded", "unhealthy"],
          "description": "Model manager component status"
        },
        "execution_router": {
          "type": "string",
          "enum": ["healthy", "degraded", "unhealthy"],
          "description": "Execution router component status"
        },
        "batch_engine": {
          "type": "string",
          "enum": ["healthy", "degraded", "unhealthy"],
          "description": "Batch engine component status"
        },
        "device_monitor": {
          "type": "string",
          "enum": ["healthy", "degraded", "unhealthy"],
          "description": "Device monitor component status"
        },
        "quantization_engine": {
          "type": "string",
          "enum": ["healthy", "degraded", "unhealthy"],
          "description": "Quantization engine component status"
        }
      },
      "required": [
        "model_manager", "execution_router",
        "batch_engine", "device_monitor", "quantization_engine"
      ]
    },
    "system_metrics": {
      "type": "object",
      "properties": {
        "cpu_utilization": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "CPU utilization ratio"
        },
        "gpu_utilization": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "GPU utilization ratio"
        },
        "memory_usage_bytes": {
          "type": "integer",
          "minimum": 0,
          "description": "System memory usage in bytes"
        },
        "gpu_memory_usage_bytes": {
          "type": "integer",
          "minimum": 0,
          "description": "GPU memory usage in bytes"
        },
        "active_requests": {
          "type": "integer",
          "minimum": 0,
          "description": "Currently active inference requests"
        },
        "queue_depth": {
          "type": "integer",
          "minimum": 0,
          "description": "Request queue depth"
        }
      }
    },
    "performance_indicators": {
      "type": "object",
      "properties": {
        "avg_response_time_ms": {
          "type": "number",
          "minimum": 0,
          "description": "Average response time (last 5 minutes)"
        },
        "requests_per_second": {
          "type": "number",
          "minimum": 0,
          "description": "Request rate (last 5 minutes)"
        },
        "error_rate": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Error rate (last 5 minutes)"
        },
        "sla_compliance": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "SLA compliance ratio"
        }
      }
    }
  },
  "additionalProperties": false
}
```

### Kubernetes Liveness Probe

**Endpoint**: `GET /health/live`

**Response**: HTTP 200 (healthy) or HTTP 503 (unhealthy)
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:30:00Z"
}
```

### Kubernetes Readiness Probe

**Endpoint**: `GET /health/ready`

**Response**: HTTP 200 (ready) or HTTP 503 (not ready)
```json
{
  "status": "ready",
  "timestamp": "2023-12-01T10:30:00Z",
  "checks": {
    "model_loaded": true,
    "inference_engine_ready": true,
    "device_available": true,
    "resources_available": true
  }
}
```

### Server Statistics

**Endpoint**: `GET /v1/stats`

**Query Parameters**:
- `time_range` (optional): Time range for metrics (`1h`, `24h`, `7d`, `30d`)
- `include_details` (optional): Include detailed breakdown (`true`/`false`)

**Response Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["server_stats", "inference_stats"],
  "properties": {
    "server_stats": {
      "type": "object",
      "properties": {
        "uptime_seconds": {
          "type": "integer",
          "minimum": 0,
          "description": "Server uptime in seconds"
        },
        "total_requests": {
          "type": "integer",
          "minimum": 0,
          "description": "Total requests processed"
        },
        "successful_requests": {
          "type": "integer",
          "minimum": 0,
          "description": "Successfully processed requests"
        },
        "failed_requests": {
          "type": "integer",
          "minimum": 0,
          "description": "Failed requests"
        },
        "error_rate": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Overall error rate"
        },
        "avg_response_time_ms": {
          "type": "number",
          "minimum": 0,
          "description": "Average response time"
        }
      },
      "required": [
        "uptime_seconds", "total_requests",
        "successful_requests", "failed_requests"
      ]
    },
    "inference_stats": {
      "type": "object",
      "properties": {
        "total_tokens_generated": {
          "type": "integer",
          "minimum": 0,
          "description": "Total tokens generated"
        },
        "avg_tokens_per_second": {
          "type": "number",
          "minimum": 0,
          "description": "Average token generation rate"
        },
        "quantization_distribution": {
          "type": "object",
          "properties": {
            "i2s": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "description": "Proportion of I2S inference"
            },
            "tl1": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "description": "Proportion of TL1 inference"
            },
            "tl2": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "description": "Proportion of TL2 inference"
            }
          },
          "required": ["i2s", "tl1", "tl2"]
        },
        "accuracy_metrics": {
          "type": "object",
          "properties": {
            "avg_quantization_accuracy": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "description": "Average quantization accuracy"
            },
            "min_accuracy_seen": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "description": "Minimum accuracy observed"
            },
            "accuracy_violations": {
              "type": "integer",
              "minimum": 0,
              "description": "Count of accuracy threshold violations"
            }
          }
        }
      },
      "required": ["total_tokens_generated", "quantization_distribution"]
    },
    "device_stats": {
      "type": "object",
      "properties": {
        "cpu_inference_count": {
          "type": "integer",
          "minimum": 0,
          "description": "Number of CPU inferences"
        },
        "gpu_inference_count": {
          "type": "integer",
          "minimum": 0,
          "description": "Number of GPU inferences"
        },
        "fallback_events": {
          "type": "integer",
          "minimum": 0,
          "description": "Device fallback events"
        },
        "device_utilization": {
          "type": "object",
          "properties": {
            "avg_cpu_utilization": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "description": "Average CPU utilization"
            },
            "avg_gpu_utilization": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "description": "Average GPU utilization"
            }
          }
        }
      },
      "required": ["cpu_inference_count", "gpu_inference_count"]
    },
    "batch_stats": {
      "type": "object",
      "properties": {
        "total_batches_processed": {
          "type": "integer",
          "minimum": 0,
          "description": "Total batches processed"
        },
        "avg_batch_size": {
          "type": "number",
          "minimum": 0,
          "description": "Average batch size"
        },
        "batch_formation_time_ms": {
          "type": "number",
          "minimum": 0,
          "description": "Average batch formation time"
        },
        "batch_efficiency": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0,
          "description": "Batch formation efficiency"
        }
      }
    }
  },
  "additionalProperties": false
}
```

## Error Response Schema

All API endpoints use a standardized error response format:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["error"],
  "properties": {
    "error": {
      "type": "object",
      "required": ["code", "message"],
      "properties": {
        "code": {
          "type": "string",
          "description": "Machine-readable error code"
        },
        "message": {
          "type": "string",
          "description": "Human-readable error message"
        },
        "details": {
          "type": "object",
          "description": "Additional error context"
        },
        "request_id": {
          "type": "string",
          "format": "uuid",
          "description": "Request identifier for debugging"
        },
        "timestamp": {
          "type": "string",
          "format": "date-time",
          "description": "Error timestamp"
        },
        "recovery_suggestions": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Suggested recovery actions"
        }
      }
    }
  },
  "additionalProperties": false
}
```

### Error Codes

**Client Errors (4xx)**:
- `INVALID_REQUEST`: Malformed request or invalid parameters
- `VALIDATION_FAILED`: Request validation failed
- `MODEL_NOT_FOUND`: Specified model not found
- `AUTHENTICATION_REQUIRED`: Authentication token required
- `AUTHORIZATION_FAILED`: Insufficient permissions
- `RATE_LIMIT_EXCEEDED`: Request rate limit exceeded
- `QUOTA_EXCEEDED`: Usage quota exceeded

**Server Errors (5xx)**:
- `INFERENCE_FAILED`: Inference execution failed
- `MODEL_LOAD_FAILED`: Model loading failed
- `DEVICE_UNAVAILABLE`: Required device not available
- `MEMORY_EXHAUSTED`: Insufficient memory for operation
- `INTERNAL_ERROR`: Unexpected internal error
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable
- `QUANTIZATION_ERROR`: Quantization operation failed
- `CROSS_VALIDATION_FAILED`: Cross-validation failed

## Rate Limiting

API endpoints implement rate limiting with the following headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1638360000
X-RateLimit-Retry-After: 60
```

**Rate Limits**:
- Inference endpoints: 100 requests/minute per client
- Model management: 10 requests/minute per client
- Health checks: 600 requests/minute per client
- Statistics: 60 requests/minute per client

## Authentication and Authorization

**Model Management APIs** require JWT Bearer Token authentication:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Token Claims**:
```json
{
  "sub": "user_id",
  "iat": 1638360000,
  "exp": 1638446400,
  "scopes": ["model:read", "model:write", "model:manage"],
  "client_id": "production_client"
}
```

**Required Scopes**:
- `model:read`: List and view model information
- `model:write`: Load new models
- `model:manage`: Hot-swap and manage existing models

## Validation and Testing

All API contracts are validated using:

1. **JSON Schema Validation**: Request/response schema enforcement
2. **OpenAPI Specification**: Complete API documentation and validation
3. **Contract Testing**: Automated API contract verification
4. **Integration Testing**: End-to-end API workflow testing
5. **Performance Testing**: API performance and scalability validation

**Validation Commands**:
```bash
# API contract validation
cargo test --no-default-features --features cpu -p bitnet-server --test api_contracts -- test_api_schema_validation

# End-to-end API testing
cargo test --no-default-features --features cpu -p bitnet-server --test integration_tests -- test_api_workflows

# Performance testing
cargo run -p bitnet-server-bench -- --test api-performance --duration 300s
```

This comprehensive API contract specification ensures consistent, reliable, and production-ready interfaces for the BitNet.rs inference server with full quantization awareness and enterprise-grade capabilities.