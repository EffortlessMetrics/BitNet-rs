use bitnet_common::{BitNetConfig, BitNetError, ConcreteTensor, Device, MockTensor};
use bitnet_inference::{GenerationConfig, InferenceEngine};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::sync::Arc;

struct MockModel {
    config: BitNetConfig,
}
impl MockModel {
    fn new() -> Self {
        Self { config: BitNetConfig::default() }
    }
}
impl Model for MockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }
    fn forward(
        &self,
        _input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor, BitNetError> {
        Ok(ConcreteTensor::Mock(MockTensor::new(vec![1, 1, 4])))
    }
    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor, BitNetError> {
        Ok(ConcreteTensor::Mock(MockTensor::new(vec![1, tokens.len(), 4])))
    }
    fn logits(&self, _input: &ConcreteTensor) -> Result<ConcreteTensor, BitNetError> {
        Ok(ConcreteTensor::Mock(MockTensor::new(vec![1, 1, 4])))
    }
}

struct MockTokenizer;
impl Tokenizer for MockTokenizer {
    fn encode(
        &self,
        _text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> Result<Vec<u32>, BitNetError> {
        Ok(vec![1, 2, 3])
    }
    fn decode(&self, _tokens: &[u32]) -> Result<String, BitNetError> {
        Ok(String::new())
    }
    fn vocab_size(&self) -> usize {
        4
    }
    fn eos_token_id(&self) -> Option<u32> {
        None
    }
    fn pad_token_id(&self) -> Option<u32> {
        None
    }
    fn token_to_piece(&self, _token: u32) -> Option<String> {
        None
    }
}

#[tokio::test]
#[ignore]
async fn prefill_allows_decode() {
    let model = Arc::new(MockModel::new());
    let tokenizer = Arc::new(MockTokenizer);
    let engine = InferenceEngine::new(model, tokenizer, Device::Cpu).unwrap();
    let prompt = vec![1, 2, 3];
    engine.prefill(&prompt).await.unwrap();
    let mut cfg = GenerationConfig::greedy();
    cfg.max_new_tokens = 2;
    let tokens = engine.generate_tokens(&[], &cfg).await.unwrap();
    assert!(!tokens.is_empty());
}
