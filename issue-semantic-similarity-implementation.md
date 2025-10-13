# [Validation] Implement semantic similarity calculation for cross-validation accuracy

## Problem Description

The `calculate_semantic_similarity` function in `crates/bitnet-inference/src/validation.rs` is currently a placeholder that returns `average_token_accuracy * 0.9` instead of performing actual semantic similarity analysis between Rust and Python inference outputs. This undermines the accuracy validation framework and prevents meaningful cross-validation testing.

## Environment
- **File**: `crates/bitnet-inference/src/validation.rs`
- **Function**: `calculate_accuracy_metrics` → `calculate_semantic_similarity`
- **Cross-validation**: Rust vs Python BitNet implementations
- **MSRV**: Rust 1.90.0
- **Feature Flags**: `crossval` feature required

## Reproduction Steps

1. Enable cross-validation feature and run accuracy tests:
   ```bash
   cargo test --features crossval -p bitnet-inference validation
   ```

2. Examine the validation results:
   ```bash
   cargo run -p xtask -- crossval --model test-model.gguf --verbose
   ```

3. Check semantic similarity calculations in output

**Expected Results**:
- Semantic similarity should be calculated using actual text embedding comparison
- Results should reflect meaningful semantic differences between outputs
- Similarity scores should correlate with human evaluation of text quality

**Actual Results**:
- Placeholder calculation: `semantic_similarity = average_token_accuracy * 0.9`
- No actual semantic analysis performed
- Validation metrics don't reflect true output quality

## Root Cause Analysis

### Current Placeholder Implementation

```rust
fn calculate_accuracy_metrics(&self, test_results: &[TestResult]) -> AccuracyMetrics {
    // ... other calculations ...

    // Placeholder for semantic similarity (would use embeddings in practice)
    let semantic_similarity = average_token_accuracy * 0.9;

    AccuracyMetrics {
        token_accuracy: average_token_accuracy,
        perplexity: avg_perplexity,
        semantic_similarity, // ← Placeholder value
        cross_entropy_loss: avg_cross_entropy,
    }
}
```

### Why This Is Insufficient

1. **No Semantic Understanding**: Token-level accuracy doesn't capture semantic meaning
2. **Misleading Metrics**: Artificial correlation with token accuracy provides false confidence
3. **Cross-validation Gaps**: Cannot detect semantic degradation in quantization
4. **Production Risk**: May miss quality regressions that affect user experience

### Semantic Similarity Requirements

For effective cross-validation, semantic similarity should:
- Use sentence/document embeddings to capture meaning
- Handle paraphrasing and equivalent expressions
- Be robust to minor wording differences
- Correlate with human judgment of semantic equivalence

## Impact Assessment

- **Severity**: Medium-High (validation framework integrity)
- **Validation Impact**:
  - Inaccurate assessment of model quality
  - Cannot detect semantic drift in quantized models
  - False confidence in cross-validation results

- **Production Risk**:
  - May deploy models with degraded semantic quality
  - User experience could suffer without detection
  - Quality regressions in quantization may go unnoticed

- **Development Impact**:
  - Developers cannot trust validation metrics
  - Difficult to optimize quantization parameters
  - No reliable quality gates for releases

## Proposed Solution

Implement comprehensive semantic similarity calculation using modern text embedding models and multiple similarity metrics.

### Technical Implementation

#### 1. Embedding-Based Semantic Similarity

```rust
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::sentence_transformers::{SentenceTransformersModel, SentenceTransformersConfig};
use tokenizers::Tokenizer;

pub struct SemanticSimilarityCalculator {
    model: SentenceTransformersModel,
    tokenizer: Tokenizer,
    device: Device,
    config: SimilarityConfig,
}

#[derive(Debug, Clone)]
pub struct SimilarityConfig {
    pub model_name: String,
    pub max_sequence_length: usize,
    pub similarity_metrics: Vec<SimilarityMetric>,
    pub batch_size: usize,
}

#[derive(Debug, Clone)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    Manhattan,
    Semantic,  // Weighted combination
}

impl SemanticSimilarityCalculator {
    pub fn new(config: SimilarityConfig) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;

        // Load sentence transformer model
        let model_path = Self::download_model_if_needed(&config.model_name)?;
        let tokenizer = Tokenizer::from_file(format!("{}/tokenizer.json", model_path))?;

        let model_config = SentenceTransformersConfig::load(&format!("{}/config.json", model_path))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[
            format!("{}/model.safetensors", model_path)
        ], candle_core::DType::F32, &device)? };

        let model = SentenceTransformersModel::load(&vb, &model_config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    pub fn calculate_similarity(
        &self,
        rust_outputs: &[String],
        python_outputs: &[String],
    ) -> Result<SemanticSimilarityResult> {
        if rust_outputs.len() != python_outputs.len() {
            return Err(ValidationError::MismatchedOutputs(
                rust_outputs.len(),
                python_outputs.len(),
            ));
        }

        let mut results = Vec::new();

        // Process in batches for efficiency
        for chunk in rust_outputs.chunks(self.config.batch_size) {
            let python_chunk = &python_outputs[
                results.len()..results.len() + chunk.len()
            ];

            let batch_result = self.calculate_batch_similarity(chunk, python_chunk)?;
            results.extend(batch_result);
        }

        // Aggregate results
        let overall_similarity = self.aggregate_similarity_scores(&results)?;

        Ok(SemanticSimilarityResult {
            individual_scores: results,
            overall_cosine_similarity: overall_similarity.cosine,
            overall_semantic_similarity: overall_similarity.semantic,
            confidence_score: overall_similarity.confidence,
            quality_assessment: self.assess_quality(&overall_similarity),
        })
    }

    fn calculate_batch_similarity(
        &self,
        rust_batch: &[String],
        python_batch: &[String],
    ) -> Result<Vec<SimilarityScore>> {
        // Generate embeddings for both batches
        let rust_embeddings = self.generate_embeddings(rust_batch)?;
        let python_embeddings = self.generate_embeddings(python_batch)?;

        let mut scores = Vec::new();

        for (rust_emb, python_emb) in rust_embeddings.iter().zip(python_embeddings.iter()) {
            let score = self.calculate_embedding_similarity(rust_emb, python_emb)?;
            scores.push(score);
        }

        Ok(scores)
    }

    fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Tensor>> {
        let mut embeddings = Vec::new();

        for text in texts {
            // Tokenize text
            let encoding = self.tokenizer.encode(text, true)?;
            let tokens = Tensor::new(encoding.get_ids(), &self.device)?
                .unsqueeze(0)?; // Add batch dimension

            let attention_mask = Tensor::new(encoding.get_attention_mask(), &self.device)?
                .unsqueeze(0)?;

            // Generate embedding
            let embedding = self.model.forward(&tokens, &attention_mask)?;

            // Mean pooling for sentence embedding
            let pooled_embedding = self.mean_pooling(&embedding, &attention_mask)?;
            let normalized_embedding = self.normalize_embedding(&pooled_embedding)?;

            embeddings.push(normalized_embedding);
        }

        Ok(embeddings)
    }

    fn calculate_embedding_similarity(
        &self,
        emb1: &Tensor,
        emb2: &Tensor,
    ) -> Result<SimilarityScore> {
        let mut score = SimilarityScore::default();

        for metric in &self.config.similarity_metrics {
            match metric {
                SimilarityMetric::Cosine => {
                    score.cosine = self.cosine_similarity(emb1, emb2)?;
                }
                SimilarityMetric::Euclidean => {
                    score.euclidean = self.euclidean_distance(emb1, emb2)?;
                }
                SimilarityMetric::Manhattan => {
                    score.manhattan = self.manhattan_distance(emb1, emb2)?;
                }
                SimilarityMetric::Semantic => {
                    score.semantic = self.weighted_semantic_similarity(emb1, emb2)?;
                }
            }
        }

        Ok(score)
    }

    fn cosine_similarity(&self, emb1: &Tensor, emb2: &Tensor) -> Result<f64> {
        let dot_product = (emb1 * emb2)?.sum_all()?.to_scalar::<f32>()?;
        let norm1 = emb1.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let norm2 = emb2.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;

        Ok((dot_product / (norm1 * norm2)) as f64)
    }

    fn weighted_semantic_similarity(&self, emb1: &Tensor, emb2: &Tensor) -> Result<f64> {
        let cosine = self.cosine_similarity(emb1, emb2)?;
        let euclidean = self.euclidean_distance(emb1, emb2)?;

        // Weighted combination emphasizing cosine similarity
        let semantic_score = 0.8 * cosine + 0.2 * (1.0 - euclidean.min(2.0) / 2.0);

        Ok(semantic_score)
    }

    fn mean_pooling(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Expand attention mask to match embedding dimensions
        let mask_expanded = attention_mask.unsqueeze(-1)?
            .expand(embeddings.shape())?
            .to_dtype(embeddings.dtype())?;

        // Apply mask and calculate mean
        let masked_embeddings = (embeddings * &mask_expanded)?;
        let sum_embeddings = masked_embeddings.sum(1)?;
        let sum_mask = mask_expanded.sum(1)?.clamp(1e-9, f64::INFINITY)?;

        Ok((sum_embeddings / sum_mask)?)
    }

    fn normalize_embedding(&self, embedding: &Tensor) -> Result<Tensor> {
        let norm = embedding.sqr()?.sum_keepdim(1)?.sqrt()?;
        Ok((embedding / norm)?)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SimilarityScore {
    pub cosine: f64,
    pub euclidean: f64,
    pub manhattan: f64,
    pub semantic: f64,
}

#[derive(Debug, Clone)]
pub struct SemanticSimilarityResult {
    pub individual_scores: Vec<SimilarityScore>,
    pub overall_cosine_similarity: f64,
    pub overall_semantic_similarity: f64,
    pub confidence_score: f64,
    pub quality_assessment: QualityAssessment,
}

#[derive(Debug, Clone)]
pub enum QualityAssessment {
    Excellent(f64),    // > 0.95
    Good(f64),         // 0.85 - 0.95
    Acceptable(f64),   // 0.75 - 0.85
    Poor(f64),         // 0.6 - 0.75
    Unacceptable(f64), // < 0.6
}
```

#### 2. Alternative Lightweight Implementation

For environments where large embedding models aren't feasible:

```rust
pub struct LightweightSemanticCalculator {
    tfidf_vectorizer: TfIdfVectorizer,
    word_overlap_calculator: WordOverlapCalculator,
    ngram_analyzer: NgramAnalyzer,
}

impl LightweightSemanticCalculator {
    pub fn new() -> Result<Self> {
        Ok(Self {
            tfidf_vectorizer: TfIdfVectorizer::new()?,
            word_overlap_calculator: WordOverlapCalculator::new(),
            ngram_analyzer: NgramAnalyzer::new(2, 3), // bigrams and trigrams
        })
    }

    pub fn calculate_similarity(
        &mut self,
        rust_outputs: &[String],
        python_outputs: &[String],
    ) -> Result<f64> {
        let mut total_similarity = 0.0;

        for (rust_text, python_text) in rust_outputs.iter().zip(python_outputs.iter()) {
            let tfidf_sim = self.tfidf_similarity(rust_text, python_text)?;
            let word_overlap_sim = self.word_overlap_calculator.calculate(rust_text, python_text)?;
            let ngram_sim = self.ngram_analyzer.similarity(rust_text, python_text)?;

            // Weighted combination
            let combined_similarity = 0.5 * tfidf_sim + 0.3 * word_overlap_sim + 0.2 * ngram_sim;
            total_similarity += combined_similarity;
        }

        Ok(total_similarity / rust_outputs.len() as f64)
    }

    fn tfidf_similarity(&mut self, text1: &str, text2: &str) -> Result<f64> {
        let vec1 = self.tfidf_vectorizer.transform(text1)?;
        let vec2 = self.tfidf_vectorizer.transform(text2)?;

        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm1 * norm2))
    }
}
```

#### 3. Integration with Validation Framework

```rust
impl ValidationFramework {
    fn calculate_accuracy_metrics(&self, test_results: &[TestResult]) -> AccuracyMetrics {
        // Existing calculations
        let average_token_accuracy = test_results.iter()
            .map(|r| r.token_accuracy)
            .sum::<f64>() / test_results.len() as f64;

        let avg_perplexity = test_results.iter()
            .map(|r| r.perplexity)
            .sum::<f64>() / test_results.len() as f64;

        let avg_cross_entropy = test_results.iter()
            .map(|r| r.cross_entropy_loss)
            .sum::<f64>() / test_results.len() as f64;

        // Real semantic similarity calculation
        let semantic_similarity = self.calculate_semantic_similarity(test_results)
            .unwrap_or_else(|e| {
                log::warn!("Failed to calculate semantic similarity: {}", e);
                // Fallback to token accuracy estimation with penalty
                average_token_accuracy * 0.8 // Lower than placeholder to indicate degraded quality
            });

        AccuracyMetrics {
            token_accuracy: average_token_accuracy,
            perplexity: avg_perplexity,
            semantic_similarity,
            cross_entropy_loss: avg_cross_entropy,
        }
    }

    fn calculate_semantic_similarity(&self, test_results: &[TestResult]) -> Result<f64> {
        let rust_outputs: Vec<String> = test_results.iter()
            .map(|r| r.rust_output.clone())
            .collect();

        let python_outputs: Vec<String> = test_results.iter()
            .map(|r| r.python_output.clone())
            .collect();

        match &self.semantic_calculator {
            Some(calculator) => {
                let result = calculator.calculate_similarity(&rust_outputs, &python_outputs)?;

                // Log detailed results for analysis
                log::info!("Semantic similarity analysis:");
                log::info!("  Cosine similarity: {:.4}", result.overall_cosine_similarity);
                log::info!("  Semantic similarity: {:.4}", result.overall_semantic_similarity);
                log::info!("  Quality assessment: {:?}", result.quality_assessment);

                Ok(result.overall_semantic_similarity)
            }
            None => {
                // Fallback to lightweight calculation
                let mut lightweight_calc = LightweightSemanticCalculator::new()?;
                lightweight_calc.calculate_similarity(&rust_outputs, &python_outputs)
            }
        }
    }
}

// Configuration for semantic similarity
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub semantic_similarity: SemanticSimilarityConfig,
    // ... other config fields
}

#[derive(Debug, Clone)]
pub enum SemanticSimilarityConfig {
    /// Use full embedding model (requires more resources)
    Embedding {
        model_name: String,
        batch_size: usize,
    },
    /// Use lightweight TF-IDF and n-gram analysis
    Lightweight,
    /// Disable semantic similarity (use placeholder)
    Disabled,
}

impl Default for SemanticSimilarityConfig {
    fn default() -> Self {
        Self::Lightweight // Safe default that works everywhere
    }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Design semantic similarity calculator interface
- [ ] Implement lightweight TF-IDF based calculator
- [ ] Add configuration system for similarity methods
- [ ] Basic integration with validation framework

### Phase 2: Embedding Models (Week 3-4)
- [ ] Integrate sentence transformer models
- [ ] Add model downloading and caching
- [ ] Implement batch processing for efficiency
- [ ] Add GPU acceleration support

### Phase 3: Advanced Features (Week 5)
- [ ] Multiple similarity metrics implementation
- [ ] Quality assessment and thresholds
- [ ] Confidence scoring and uncertainty estimation
- [ ] Performance optimization

### Phase 4: Validation & Testing (Week 6)
- [ ] Comprehensive test suite
- [ ] Cross-validation with human evaluation
- [ ] Performance benchmarking
- [ ] Documentation and examples

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_text_similarity() {
        let mut calc = LightweightSemanticCalculator::new().unwrap();
        let text = "The quick brown fox jumps over the lazy dog.";

        let similarity = calc.calculate_similarity(&[text.to_string()], &[text.to_string()]).unwrap();
        assert!((similarity - 1.0).abs() < 0.01, "Identical text should have similarity ~1.0");
    }

    #[test]
    fn test_paraphrase_similarity() {
        let mut calc = LightweightSemanticCalculator::new().unwrap();
        let text1 = "The cat sat on the mat.";
        let text2 = "A cat was sitting on a mat.";

        let similarity = calc.calculate_similarity(&[text1.to_string()], &[text2.to_string()]).unwrap();
        assert!(similarity > 0.7, "Paraphrases should have high similarity: {}", similarity);
    }

    #[test]
    fn test_unrelated_text_similarity() {
        let mut calc = LightweightSemanticCalculator::new().unwrap();
        let text1 = "The weather is nice today.";
        let text2 = "Machine learning algorithms are complex.";

        let similarity = calc.calculate_similarity(&[text1.to_string()], &[text2.to_string()]).unwrap();
        assert!(similarity < 0.5, "Unrelated text should have low similarity: {}", similarity);
    }

    #[cfg(feature = "embedding-models")]
    #[test]
    fn test_embedding_model_similarity() {
        let config = SimilarityConfig {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            max_sequence_length: 512,
            similarity_metrics: vec![SimilarityMetric::Cosine, SimilarityMetric::Semantic],
            batch_size: 4,
        };

        let calc = SemanticSimilarityCalculator::new(config).unwrap();

        let rust_outputs = vec![
            "The model generates high-quality text.".to_string(),
            "Neural networks are powerful tools.".to_string(),
        ];

        let python_outputs = vec![
            "The model produces excellent text output.".to_string(),
            "Deep learning models are very effective.".to_string(),
        ];

        let result = calc.calculate_similarity(&rust_outputs, &python_outputs).unwrap();

        assert!(result.overall_semantic_similarity > 0.7);
        assert!(matches!(result.quality_assessment, QualityAssessment::Good(_) | QualityAssessment::Excellent(_)));
    }
}
```

### Integration Tests
```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_validation_framework_integration() {
        let config = ValidationConfig {
            semantic_similarity: SemanticSimilarityConfig::Lightweight,
            ..Default::default()
        };

        let framework = ValidationFramework::new(config).unwrap();

        let test_results = vec![
            TestResult {
                rust_output: "The quick brown fox".to_string(),
                python_output: "A fast brown fox".to_string(),
                token_accuracy: 0.8,
                perplexity: 15.0,
                cross_entropy_loss: 2.5,
            },
            TestResult {
                rust_output: "Machine learning is fascinating".to_string(),
                python_output: "AI and ML are very interesting".to_string(),
                token_accuracy: 0.75,
                perplexity: 18.0,
                cross_entropy_loss: 2.8,
            },
        ];

        let metrics = framework.calculate_accuracy_metrics(&test_results);

        // Semantic similarity should be reasonable for these paraphrases
        assert!(metrics.semantic_similarity > 0.6);
        assert!(metrics.semantic_similarity <= 1.0);

        // Should not be the placeholder calculation
        assert!((metrics.semantic_similarity - (metrics.token_accuracy * 0.9)).abs() > 0.1);
    }
}
```

## Performance Considerations

### Model Size and Speed
- **Lightweight TF-IDF**: ~1ms per comparison, minimal memory
- **Small Embedding Model**: ~10-50ms per comparison, ~100MB memory
- **Large Embedding Model**: ~50-200ms per comparison, ~500MB memory

### Optimization Strategies
- Batch processing for embedding models
- Caching of embeddings for repeated comparisons
- Progressive calculation (fast method first, detailed if needed)
- GPU acceleration for large-scale validation

## Acceptance Criteria

- [ ] Real semantic similarity calculation replaces placeholder
- [ ] Multiple similarity calculation methods available
- [ ] Integration with existing validation framework
- [ ] Configurable similarity calculation approach
- [ ] Performance suitable for cross-validation workflows
- [ ] Test suite validates accuracy against human judgment
- [ ] Documentation explains similarity metrics and interpretation
- [ ] Backward compatibility maintained for existing workflows
- [ ] Quality assessment provides actionable insights

## Dependencies

- Text processing: `tokenizers`, `unicode-segmentation`
- Embedding models: `candle-transformers`, model files
- Linear algebra: `candle-core` for tensor operations
- Optional: `sentence-transformers` models via Hugging Face

## Related Issues

- Cross-validation framework improvements
- Model quality assessment
- Quantization accuracy validation
- Production quality gates

## Labels
- `validation`
- `semantic-similarity`
- `cross-validation`
- `ml-models`
- `priority-medium`
- `enhancement`
