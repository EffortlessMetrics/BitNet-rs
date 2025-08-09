//! C++ implementation wrapper for BitNet.cpp cross-validation testing
//!
//! This module provides a wrapper around the BitNet.cpp implementation that conforms
//! to the BitNetImplementation trait for cross-validation testing.

use crate::common::cross_validation::implementation::{
    BitNetImplementation, ImplementationCapabilities, ImplementationFactory, InferenceConfig,
    InferenceResult, ModelFormat, ModelInfo, PerformanceMetrics, ResourceInfo,
};
use crate::common::errors::{ImplementationError, ImplementationResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::colle