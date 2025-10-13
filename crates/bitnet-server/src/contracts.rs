//! API Contract Validation and Versioning
//!
//! This module ensures API backward compatibility and prevents breaking changes.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

/// API versioning information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre_release: Option<String>,
}

impl ApiVersion {
    pub const CURRENT: Self = Self {
        major: 1,
        minor: 0,
        patch: 0,
        pre_release: None,
    };

    pub fn to_string(&self) -> String {
        if let Some(pre) = &self.pre_release {
            format!("{}.{}.{}-{}", self.major, self.minor, self.patch, pre)
        } else {
            format!("{}.{}.{}", self.major, self.minor, self.patch)
        }
    }

    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        // Major version must match for compatibility
        self.major == other.major
    }
}

/// Contract validation errors
#[derive(Debug, Error)]
pub enum ContractError {
    #[error("Breaking change detected: {0}")]
    BreakingChange(String),

    #[error("Schema validation failed: {0}")]
    SchemaValidation(String),

    #[error("Version incompatible: expected {expected}, got {actual}")]
    VersionMismatch { expected: String, actual: String },

    #[error("Required field missing: {0}")]
    MissingField(String),

    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
}

/// API contract validator
pub struct ContractValidator {
    schemas: HashMap<String, Value>,
    version: ApiVersion,
}

impl ContractValidator {
    pub fn new() -> Self {
        Self {
            schemas: Self::load_schemas(),
            version: ApiVersion::CURRENT,
        }
    }

    fn load_schemas() -> HashMap<String, Value> {
        let mut schemas = HashMap::new();

        // Load inference contract
        let inference_schema = include_str!("../../../api-contracts/v1/inference.json");
        schemas.insert(
            "inference".to_string(),
            serde_json::from_str(inference_schema).expect("Invalid inference schema"),
        );

        schemas
    }

    /// Validate a request against its contract
    pub fn validate_request(
        &self,
        endpoint: &str,
        request: &Value,
    ) -> Result<(), ContractError> {
        // Get the appropriate schema
        let schema = self.schemas.get("inference")
            .ok_or_else(|| ContractError::SchemaValidation("Schema not found".into()))?;

        // Extract the endpoint definition
        let endpoints = schema.get("endpoints")
            .ok_or_else(|| ContractError::SchemaValidation("No endpoints defined".into()))?;

        let endpoint_def = endpoints.get(endpoint)
            .ok_or_else(|| ContractError::SchemaValidation(format!("Unknown endpoint: {}", endpoint)))?;

        // Validate against the request schema
        if let Some(post_def) = endpoint_def.get("POST") {
            if let Some(request_schema) = post_def.get("request") {
                self.validate_against_schema(request, request_schema)?;
            }
        }

        Ok(())
    }

    /// Validate a response against its contract
    pub fn validate_response(
        &self,
        endpoint: &str,
        response: &Value,
    ) -> Result<(), ContractError> {
        let schema = self.schemas.get("inference")
            .ok_or_else(|| ContractError::SchemaValidation("Schema not found".into()))?;

        let endpoints = schema.get("endpoints")
            .ok_or_else(|| ContractError::SchemaValidation("No endpoints defined".into()))?;

        let endpoint_def = endpoints.get(endpoint)
            .ok_or_else(|| ContractError::SchemaValidation(format!("Unknown endpoint: {}", endpoint)))?;

        // Validate against the response schema
        if let Some(method_def) = endpoint_def.get("POST").or(endpoint_def.get("GET")) {
            if let Some(response_schema) = method_def.get("response") {
                self.validate_against_schema(response, response_schema)?;
            }
        }

        Ok(())
    }

    fn validate_against_schema(
        &self,
        data: &Value,
        schema: &Value,
    ) -> Result<(), ContractError> {
        // Check if it's a reference
        if let Some(ref_path) = schema.get("$ref").and_then(|v| v.as_str()) {
            let resolved = self.resolve_ref(ref_path)?;
            return self.validate_against_schema(data, &resolved);
        }

        // Validate type
        if let Some(type_str) = schema.get("type").and_then(|v| v.as_str()) {
            self.validate_type(data, type_str)?;
        }

        // Validate object properties
        if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
            let data_obj = data.as_object()
                .ok_or_else(|| ContractError::TypeMismatch {
                    expected: "object".to_string(),
                    actual: format!("{:?}", data),
                })?;

            // Check required fields
            if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
                for req_field in required {
                    let field_name = req_field.as_str()
                        .ok_or_else(|| ContractError::SchemaValidation("Invalid required field".into()))?;

                    if !data_obj.contains_key(field_name) {
                        return Err(ContractError::MissingField(field_name.to_string()));
                    }
                }
            }

            // Validate each property
            for (key, prop_schema) in properties {
                if let Some(value) = data_obj.get(key) {
                    self.validate_against_schema(value, prop_schema)?;
                }
            }
        }

        // Validate array items
        if let Some(items_schema) = schema.get("items") {
            let data_array = data.as_array()
                .ok_or_else(|| ContractError::TypeMismatch {
                    expected: "array".to_string(),
                    actual: format!("{:?}", data),
                })?;

            for item in data_array {
                self.validate_against_schema(item, items_schema)?;
            }
        }

        // Validate enum values
        if let Some(enum_values) = schema.get("enum").and_then(|v| v.as_array()) {
            let mut valid = false;
            for enum_val in enum_values {
                if data == enum_val {
                    valid = true;
                    break;
                }
            }
            if !valid {
                return Err(ContractError::SchemaValidation(
                    format!("Value {:?} not in enum {:?}", data, enum_values)
                ));
            }
        }

        Ok(())
    }

    fn validate_type(&self, data: &Value, expected_type: &str) -> Result<(), ContractError> {
        let valid = match expected_type {
            "string" => data.is_string(),
            "number" => data.is_number(),
            "integer" => data.is_i64() || data.is_u64(),
            "boolean" => data.is_boolean(),
            "object" => data.is_object(),
            "array" => data.is_array(),
            "null" => data.is_null(),
            _ => return Err(ContractError::SchemaValidation(format!("Unknown type: {}", expected_type))),
        };

        if !valid {
            return Err(ContractError::TypeMismatch {
                expected: expected_type.to_string(),
                actual: format!("{:?}", data),
            });
        }

        Ok(())
    }

    fn resolve_ref(&self, ref_path: &str) -> Result<Value, ContractError> {
        // Parse reference like "#/definitions/ModelConfig"
        let parts: Vec<&str> = ref_path.split('/').collect();
        if parts.len() < 3 || parts[0] != "#" {
            return Err(ContractError::SchemaValidation(format!("Invalid reference: {}", ref_path)));
        }

        let schema = self.schemas.get("inference")
            .ok_or_else(|| ContractError::SchemaValidation("Schema not found".into()))?;

        let mut current = schema;
        for part in &parts[1..] {
            current = current.get(part)
                .ok_or_else(|| ContractError::SchemaValidation(format!("Reference not found: {}", ref_path)))?;
        }

        Ok(current.clone())
    }
}

/// Middleware for contract validation
pub fn validate_contract_middleware() -> impl Fn(&Value, &Value) -> Result<(), ContractError> {
    let validator = ContractValidator::new();

    move |request: &Value, response: &Value| {
        // Validate request
        if let Some(endpoint) = request.get("endpoint").and_then(|v| v.as_str()) {
            validator.validate_request(endpoint, request)?;
            validator.validate_response(endpoint, response)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_api_version_compatibility() {
        let v1 = ApiVersion { major: 1, minor: 0, patch: 0, pre_release: None };
        let v1_1 = ApiVersion { major: 1, minor: 1, patch: 0, pre_release: None };
        let v2 = ApiVersion { major: 2, minor: 0, patch: 0, pre_release: None };

        assert!(v1.is_compatible_with(&v1_1));
        assert!(!v1.is_compatible_with(&v2));
    }

    #[test]
    fn test_validate_inference_request() {
        let validator = ContractValidator::new();

        let valid_request = json!({
            "prompt": "Hello, world!",
            "max_tokens": 100,
            "temperature": 0.7
        });

        // This should validate successfully
        validator.validate_against_schema(
            &valid_request,
            &json!({
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": { "type": "string" },
                    "max_tokens": { "type": "integer" },
                    "temperature": { "type": "number" }
                }
            })
        ).unwrap();
    }

    #[test]
    fn test_missing_required_field() {
        let validator = ContractValidator::new();

        let invalid_request = json!({
            "max_tokens": 100
        });

        let result = validator.validate_against_schema(
            &invalid_request,
            &json!({
                "type": "object",
                "required": ["prompt"],
                "properties": {
                    "prompt": { "type": "string" },
                    "max_tokens": { "type": "integer" }
                }
            })
        );

        assert!(matches!(result, Err(ContractError::MissingField(_))));
    }
}
