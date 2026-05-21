use crate::{ErrorResponse, security};
use anyhow::Result;
use axum::{
    Json,
    http::{HeaderMap, StatusCode},
};
use bitnet_common::Device;
use std::{net::IpAddr, time::Duration};

use crate::batch_engine::RequestPriority;

pub(crate) fn calculate_tokens_per_second(tokens: u64, duration: Duration) -> f64 {
    let duration_ms = duration.as_millis();
    if duration_ms > 0 && tokens > 0 { (tokens as f64 * 1000.0) / duration_ms as f64 } else { 0.0 }
}

pub(crate) fn create_error_response(
    error: &str,
    error_code: &str,
    request_id: Option<String>,
    details: Option<serde_json::Value>,
) -> Json<ErrorResponse> {
    Json(ErrorResponse {
        error: error.to_string(),
        error_code: error_code.to_string(),
        request_id,
        details,
    })
}

pub(crate) fn handle_validation_error(
    error: &security::ValidationError,
    request_id: Option<String>,
) -> (StatusCode, Json<ErrorResponse>) {
    let (status, error_code) = match error {
        security::ValidationError::PromptTooLong(_, _) => {
            (StatusCode::BAD_REQUEST, "PROMPT_TOO_LONG")
        }
        security::ValidationError::TooManyTokens(_, _) => {
            (StatusCode::BAD_REQUEST, "TOO_MANY_TOKENS")
        }
        security::ValidationError::InvalidCharacters => {
            (StatusCode::BAD_REQUEST, "INVALID_CHARACTERS")
        }
        security::ValidationError::BlockedContent(_) => {
            (StatusCode::BAD_REQUEST, "BLOCKED_CONTENT")
        }
        security::ValidationError::MissingField(_) => (StatusCode::BAD_REQUEST, "MISSING_FIELD"),
        security::ValidationError::InvalidFieldValue(_) => {
            (StatusCode::BAD_REQUEST, "INVALID_FIELD_VALUE")
        }
    };

    let response = create_error_response(&error.to_string(), error_code, request_id, None);
    (status, response)
}

pub(crate) fn parse_priority(priority: Option<&str>) -> RequestPriority {
    match priority {
        Some("low") => RequestPriority::Low,
        Some("normal") => RequestPriority::Normal,
        Some("high") => RequestPriority::High,
        Some("critical") => RequestPriority::Critical,
        _ => RequestPriority::Normal,
    }
}

pub(crate) fn parse_device(device: &str) -> Result<Device> {
    let normalized = device.to_lowercase();
    match normalized.as_str() {
        "cpu" => Ok(Device::Cpu),
        "gpu" | "cuda" | "vulkan" | "opencl" | "ocl" => Ok(Device::Cuda(0)),
        _ if normalized.starts_with("cuda:") => Ok(Device::Cuda(normalized[5..].parse::<usize>()?)),
        _ if normalized.starts_with("vulkan:") => {
            Ok(Device::Cuda(normalized[7..].parse::<usize>()?))
        }
        _ if normalized.starts_with("opencl:") => {
            Ok(Device::Cuda(normalized[7..].parse::<usize>()?))
        }
        _ if normalized.starts_with("ocl:") => Ok(Device::Cuda(normalized[4..].parse::<usize>()?)),
        _ => anyhow::bail!("Unknown device: {}", device),
    }
}

pub(crate) fn extract_client_ip_from_headers(headers: &HeaderMap) -> Option<IpAddr> {
    security::extract_client_ip_from_headers(headers)
}
