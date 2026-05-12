//! JSON schema definitions and validation for structured output.
//!
//! This submodule owns schema data types plus recursive validation.  Keeping
//! validation here leaves the top-level structured-output module focused on
//! parsing, grammar forcing, repair, and engine orchestration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── JSON schema types ────────────────────────────────────────────────────

/// Primitive and compound types used in JSON schema definitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
    Null,
}

impl fmt::Display for SchemaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::String => "string",
            Self::Number => "number",
            Self::Integer => "integer",
            Self::Boolean => "boolean",
            Self::Array => "array",
            Self::Object => "object",
            Self::Null => "null",
        };
        write!(f, "{s}")
    }
}

/// A simplified JSON schema definition.
///
/// Supports type constraints, required fields, enumeration values, and
/// nested object/array schemas—enough for typical structured-output use
/// cases without pulling in a full JSON-Schema library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Expected top-level type.
    pub schema_type: SchemaType,
    /// Property schemas (only meaningful when `schema_type == Object`).
    pub properties: HashMap<String, Self>,
    /// Required property names (only meaningful for objects).
    pub required: Vec<String>,
    /// Allowed literal values (acts as an enum constraint).
    pub enum_values: Vec<serde_json::Value>,
    /// Schema for array items (only meaningful when `schema_type == Array`).
    pub items: Option<Box<Self>>,
    /// Optional human-readable description.
    pub description: Option<String>,
}

impl Default for JsonSchema {
    fn default() -> Self {
        Self {
            schema_type: SchemaType::Object,
            properties: HashMap::new(),
            required: Vec::new(),
            enum_values: Vec::new(),
            items: None,
            description: None,
        }
    }
}

impl JsonSchema {
    /// Create a schema that expects a specific primitive type.
    pub fn new(schema_type: SchemaType) -> Self {
        Self { schema_type, ..Default::default() }
    }

    /// Create an object schema with the given property definitions.
    pub fn object(properties: HashMap<String, Self>) -> Self {
        Self { schema_type: SchemaType::Object, properties, ..Default::default() }
    }

    /// Create an array schema with a given item schema.
    pub fn array(items: Self) -> Self {
        Self { schema_type: SchemaType::Array, items: Some(Box::new(items)), ..Default::default() }
    }

    /// Mark certain property names as required.
    #[must_use]
    pub fn with_required(mut self, required: Vec<String>) -> Self {
        self.required = required;
        self
    }

    /// Restrict values to an explicit set.
    #[must_use]
    pub fn with_enum(mut self, values: Vec<serde_json::Value>) -> Self {
        self.enum_values = values;
        self
    }
}

// ── Schema validation ────────────────────────────────────────────────────

/// Validation errors produced by [`SchemaValidator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// The value's type does not match the schema.
    TypeMismatch { expected: SchemaType, got: String },
    /// A required property is missing from an object.
    MissingRequired(String),
    /// The value is not among the allowed enum values.
    InvalidEnumValue(String),
    /// A nested property failed validation.
    PropertyError { property: String, error: Box<Self> },
    /// An array item failed validation.
    ItemError { index: usize, error: Box<Self> },
    /// The input could not be parsed as JSON.
    ParseError(String),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {expected}, got {got}")
            }
            Self::MissingRequired(field) => {
                write!(f, "missing required field: {field}")
            }
            Self::InvalidEnumValue(val) => {
                write!(f, "value not in enum: {val}")
            }
            Self::PropertyError { property, error } => {
                write!(f, "property '{property}': {error}")
            }
            Self::ItemError { index, error } => {
                write!(f, "item[{index}]: {error}")
            }
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

/// Validates a [`serde_json::Value`] against a [`JsonSchema`].
pub struct SchemaValidator;

impl SchemaValidator {
    /// Validate `value` against `schema`, returning all errors found.
    pub fn validate(schema: &JsonSchema, value: &serde_json::Value) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        Self::validate_inner(schema, value, &mut errors);
        errors
    }

    fn validate_inner(
        schema: &JsonSchema,
        value: &serde_json::Value,
        errors: &mut Vec<ValidationError>,
    ) {
        Self::validate_enum_constraint(schema, value, errors);

        if !Self::validate_type_constraint(schema, value, errors) {
            return; // No point diving deeper if the top-level type is wrong.
        }

        match schema.schema_type {
            SchemaType::Object => Self::validate_object(schema, value, errors),
            SchemaType::Array => Self::validate_array(schema, value, errors),
            _ => {} // Primitive types already type-checked above.
        }
    }

    fn validate_enum_constraint(
        schema: &JsonSchema,
        value: &serde_json::Value,
        errors: &mut Vec<ValidationError>,
    ) {
        if !schema.enum_values.is_empty() && !schema.enum_values.contains(value) {
            errors.push(ValidationError::InvalidEnumValue(value.to_string()));
        }
    }

    fn validate_type_constraint(
        schema: &JsonSchema,
        value: &serde_json::Value,
        errors: &mut Vec<ValidationError>,
    ) -> bool {
        if Self::type_matches(&schema.schema_type, value) {
            return true;
        }

        errors.push(ValidationError::TypeMismatch {
            expected: schema.schema_type.clone(),
            got: Self::json_type_name(value).to_string(),
        });
        false
    }

    fn validate_object(
        schema: &JsonSchema,
        value: &serde_json::Value,
        errors: &mut Vec<ValidationError>,
    ) {
        let Some(obj) = value.as_object() else {
            return;
        };

        for req in &schema.required {
            if !obj.contains_key(req) {
                errors.push(ValidationError::MissingRequired(req.clone()));
            }
        }

        for (key, prop_schema) in &schema.properties {
            if let Some(prop_val) = obj.get(key) {
                Self::validate_property(key, prop_schema, prop_val, errors);
            }
        }
    }

    fn validate_property(
        key: &str,
        prop_schema: &JsonSchema,
        prop_val: &serde_json::Value,
        errors: &mut Vec<ValidationError>,
    ) {
        let mut prop_errors = Vec::new();
        Self::validate_inner(prop_schema, prop_val, &mut prop_errors);
        errors.extend(prop_errors.into_iter().map(|error| ValidationError::PropertyError {
            property: key.to_string(),
            error: Box::new(error),
        }));
    }

    fn validate_array(
        schema: &JsonSchema,
        value: &serde_json::Value,
        errors: &mut Vec<ValidationError>,
    ) {
        let (Some(items_schema), Some(arr)) = (&schema.items, value.as_array()) else {
            return;
        };

        for (idx, item) in arr.iter().enumerate() {
            Self::validate_array_item(idx, items_schema, item, errors);
        }
    }

    fn validate_array_item(
        index: usize,
        items_schema: &JsonSchema,
        item: &serde_json::Value,
        errors: &mut Vec<ValidationError>,
    ) {
        let mut item_errors = Vec::new();
        Self::validate_inner(items_schema, item, &mut item_errors);
        errors.extend(
            item_errors
                .into_iter()
                .map(|error| ValidationError::ItemError { index, error: Box::new(error) }),
        );
    }

    fn type_matches(expected: &SchemaType, value: &serde_json::Value) -> bool {
        match (expected, value) {
            (SchemaType::Integer, serde_json::Value::Number(n)) => {
                n.as_i64().is_some() || n.as_u64().is_some()
            }
            (SchemaType::String, serde_json::Value::String(_))
            | (SchemaType::Number, serde_json::Value::Number(_))
            | (SchemaType::Boolean, serde_json::Value::Bool(_))
            | (SchemaType::Array, serde_json::Value::Array(_))
            | (SchemaType::Object, serde_json::Value::Object(_))
            | (SchemaType::Null, serde_json::Value::Null) => true,
            _ => false,
        }
    }

    const fn json_type_name(value: &serde_json::Value) -> &'static str {
        match value {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "boolean",
            serde_json::Value::Number(_) => "number",
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Object(_) => "object",
        }
    }
}
