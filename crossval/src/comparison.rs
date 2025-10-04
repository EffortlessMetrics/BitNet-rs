//! High-level comparison functions for cross-validation

use crate::{
    CrossvalConfig, Result,
    cpp_bindings::CppModel,
    fixtures::TestFixture,
    utils::{compare_tokens, logging, perf},
};

/// Result of a cross-validation comparison
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComparisonResult {
    pub test_name: String,
    pub prompt: String,
    pub rust_tokens: Vec<u32>,
    pub cpp_tokens: Vec<u32>,
    pub tokens_match: bool,
    pub rust_performance: Option<f64>, // tokens per second
    pub cpp_performance: Option<f64>,  // tokens per second
    pub error: Option<String>,
}

/// Cross-validation runner
pub struct CrossValidator {
    config: CrossvalConfig,
}

impl CrossValidator {
    /// Create a new cross-validator with the given configuration
    pub fn new(config: CrossvalConfig) -> Self {
        Self { config }
    }

    /// Run cross-validation on a single test fixture
    pub fn validate_fixture(&self, fixture: &TestFixture) -> Result<Vec<ComparisonResult>> {
        // Load C++ model
        let cpp_model = CppModel::load(&fixture.model_path)?;

        let mut results = Vec::new();

        for prompt in &fixture.test_prompts {
            let result = self.compare_single_prompt(&fixture.name, prompt, &cpp_model);

            results.push(result);
        }

        Ok(results)
    }

    /// Compare a single prompt between Rust and C++ implementations
    fn compare_single_prompt(
        &self,
        test_name: &str,
        prompt: &str,
        cpp_model: &CppModel,
    ) -> ComparisonResult {
        let mut result = ComparisonResult {
            test_name: test_name.to_string(),
            prompt: prompt.to_string(),
            rust_tokens: Vec::new(),
            cpp_tokens: Vec::new(),
            tokens_match: false,
            rust_performance: None,
            cpp_performance: None,
            error: None,
        };

        // Generate with Rust implementation
        let (rust_perf, rust_tokens) = match self.generate_rust(prompt) {
            Ok(tokens) => {
                let (perf, _) = perf::measure(|| tokens.len());
                (Some(perf.tokens_per_second), tokens)
            }
            Err(e) => {
                result.error = Some(format!("Rust generation failed: {}", e));
                return result;
            }
        };

        // Generate with C++ implementation
        let (cpp_perf, cpp_tokens) = match cpp_model.generate(prompt, self.config.max_tokens) {
            Ok(tokens) => {
                let (perf, _) = perf::measure(|| tokens.len());
                (Some(perf.tokens_per_second), tokens)
            }
            Err(e) => {
                result.error = Some(format!("C++ generation failed: {}", e));
                return result;
            }
        };

        // Compare tokens
        let tokens_match = match compare_tokens(&rust_tokens, &cpp_tokens, &self.config) {
            Ok(matches) => matches,
            Err(e) => {
                result.error = Some(format!("Token comparison failed: {}", e));
                false
            }
        };

        // Log results
        logging::log_comparison(test_name, rust_tokens.len(), cpp_tokens.len(), tokens_match);

        if let (Some(rust_tps), Some(cpp_tps)) = (rust_perf, cpp_perf) {
            logging::log_performance(test_name, rust_tps, cpp_tps);
        }

        result.rust_tokens = rust_tokens;
        result.cpp_tokens = cpp_tokens;
        result.tokens_match = tokens_match;
        result.rust_performance = rust_perf;
        result.cpp_performance = cpp_perf;

        result
    }

    /// Generate tokens using the Rust implementation
    /// This is a placeholder - in real implementation, this would call into bitnet-inference
    fn generate_rust(&self, _prompt: &str) -> Result<Vec<u32>> {
        // Placeholder implementation
        // In real code, this would use the bitnet-inference crate
        Ok(vec![1, 2, 3, 4, 5]) // Dummy tokens
    }
}

/// Run cross-validation on all available fixtures
pub fn validate_all_fixtures(config: CrossvalConfig) -> Result<Vec<ComparisonResult>> {
    let validator = CrossValidator::new(config);
    let fixture_names = crate::fixtures::TestFixture::list_available()?;

    let mut all_results = Vec::new();

    for fixture_name in fixture_names {
        println!("Running cross-validation for fixture: {}", fixture_name);

        let fixture = match crate::fixtures::TestFixture::load(&fixture_name) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to load fixture '{}': {}", fixture_name, e);
                continue;
            }
        };

        match validator.validate_fixture(&fixture) {
            Ok(mut results) => all_results.append(&mut results),
            Err(e) => {
                eprintln!("Failed to validate fixture '{}': {}", fixture_name, e);
            }
        }
    }

    Ok(all_results)
}
