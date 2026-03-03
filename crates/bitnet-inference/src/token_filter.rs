//! Token filtering and logit processing.
//!
//! Apply filters to logits before sampling: ban tokens, boost tokens,
//! apply frequency penalties, and enforce token budgets.

/// A logit filter that modifies logit values.
pub trait LogitFilter: std::fmt::Debug {
    fn name(&self) -> &str;
    fn apply(&self, logits: &mut [f32], context: &FilterContext);
}

/// Context available to filters.
#[derive(Debug, Clone)]
pub struct FilterContext {
    pub generated_ids: Vec<u32>,
    pub position: usize,
}

impl FilterContext {
    pub fn new() -> Self {
        Self { generated_ids: Vec::new(), position: 0 }
    }
    pub fn with_ids(ids: Vec<u32>) -> Self {
        Self { position: ids.len(), generated_ids: ids }
    }
}

impl Default for FilterContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Ban specific token IDs.
#[derive(Debug, Clone)]
pub struct BanTokenFilter {
    pub banned: Vec<u32>,
}

impl BanTokenFilter {
    pub fn new(banned: Vec<u32>) -> Self {
        Self { banned }
    }
}

impl LogitFilter for BanTokenFilter {
    fn name(&self) -> &str {
        "ban_tokens"
    }
    fn apply(&self, logits: &mut [f32], _ctx: &FilterContext) {
        for &id in &self.banned {
            if (id as usize) < logits.len() {
                logits[id as usize] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Boost specific token IDs by adding a bias.
#[derive(Debug, Clone)]
pub struct BoostTokenFilter {
    pub boosts: Vec<(u32, f32)>,
}

impl BoostTokenFilter {
    pub fn new(boosts: Vec<(u32, f32)>) -> Self {
        Self { boosts }
    }
}

impl LogitFilter for BoostTokenFilter {
    fn name(&self) -> &str {
        "boost_tokens"
    }
    fn apply(&self, logits: &mut [f32], _ctx: &FilterContext) {
        for &(id, bias) in &self.boosts {
            if (id as usize) < logits.len() {
                logits[id as usize] += bias;
            }
        }
    }
}

/// Frequency penalty: penalize tokens based on their count in context.
#[derive(Debug, Clone)]
pub struct FrequencyPenaltyFilter {
    pub penalty: f32,
}

impl FrequencyPenaltyFilter {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl LogitFilter for FrequencyPenaltyFilter {
    fn name(&self) -> &str {
        "frequency_penalty"
    }
    fn apply(&self, logits: &mut [f32], ctx: &FilterContext) {
        let mut counts = vec![0u32; logits.len()];
        for &id in &ctx.generated_ids {
            if (id as usize) < counts.len() {
                counts[id as usize] += 1;
            }
        }
        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                logits[i] -= self.penalty * *count as f32;
            }
        }
    }
}

/// Presence penalty: penalize tokens that appeared at all.
#[derive(Debug, Clone)]
pub struct PresencePenaltyFilter {
    pub penalty: f32,
}

impl PresencePenaltyFilter {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl LogitFilter for PresencePenaltyFilter {
    fn name(&self) -> &str {
        "presence_penalty"
    }
    fn apply(&self, logits: &mut [f32], ctx: &FilterContext) {
        let mut seen = vec![false; logits.len()];
        for &id in &ctx.generated_ids {
            if (id as usize) < seen.len() {
                seen[id as usize] = true;
            }
        }
        for (i, &present) in seen.iter().enumerate() {
            if present {
                logits[i] -= self.penalty;
            }
        }
    }
}

/// Filter pipeline.
#[derive(Debug)]
pub struct FilterPipeline {
    filters: Vec<Box<dyn LogitFilter>>,
}

impl FilterPipeline {
    pub fn new() -> Self {
        Self { filters: Vec::new() }
    }

    pub fn add<F: LogitFilter + 'static>(&mut self, filter: F) {
        self.filters.push(Box::new(filter));
    }

    pub fn apply_all(&self, logits: &mut [f32], ctx: &FilterContext) {
        for filter in &self.filters {
            filter.apply(logits, ctx);
        }
    }

    pub fn filter_count(&self) -> usize {
        self.filters.len()
    }

    pub fn filter_names(&self) -> Vec<&str> {
        self.filters.iter().map(|f| f.name()).collect()
    }
}

impl Default for FilterPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ban_tokens() {
        let filter = BanTokenFilter::new(vec![2, 5]);
        let mut logits = vec![1.0; 10];
        filter.apply(&mut logits, &FilterContext::new());
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[5], f32::NEG_INFINITY);
        assert_eq!(logits[0], 1.0);
    }

    #[test]
    fn test_boost_tokens() {
        let filter = BoostTokenFilter::new(vec![(3, 2.0)]);
        let mut logits = vec![0.0; 10];
        filter.apply(&mut logits, &FilterContext::new());
        assert_eq!(logits[3], 2.0);
        assert_eq!(logits[0], 0.0);
    }

    #[test]
    fn test_frequency_penalty() {
        let filter = FrequencyPenaltyFilter::new(0.5);
        let ctx = FilterContext::with_ids(vec![1, 1, 3]);
        let mut logits = vec![1.0; 10];
        filter.apply(&mut logits, &ctx);
        assert_eq!(logits[1], 0.0); // 1.0 - 0.5*2
        assert_eq!(logits[3], 0.5); // 1.0 - 0.5*1
        assert_eq!(logits[0], 1.0); // unchanged
    }

    #[test]
    fn test_presence_penalty() {
        let filter = PresencePenaltyFilter::new(1.0);
        let ctx = FilterContext::with_ids(vec![2, 2, 4]);
        let mut logits = vec![5.0; 10];
        filter.apply(&mut logits, &ctx);
        assert_eq!(logits[2], 4.0); // 5.0 - 1.0
        assert_eq!(logits[4], 4.0); // 5.0 - 1.0
        assert_eq!(logits[0], 5.0);
    }

    #[test]
    fn test_pipeline() {
        let mut pipeline = FilterPipeline::new();
        pipeline.add(BanTokenFilter::new(vec![0]));
        pipeline.add(BoostTokenFilter::new(vec![(1, 10.0)]));
        let mut logits = vec![1.0; 5];
        pipeline.apply_all(&mut logits, &FilterContext::new());
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 11.0);
    }

    #[test]
    fn test_pipeline_names() {
        let mut pipeline = FilterPipeline::new();
        pipeline.add(BanTokenFilter::new(vec![]));
        pipeline.add(FrequencyPenaltyFilter::new(1.0));
        assert_eq!(pipeline.filter_names(), vec!["ban_tokens", "frequency_penalty"]);
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = FilterPipeline::new();
        let mut logits = vec![1.0; 5];
        pipeline.apply_all(&mut logits, &FilterContext::new());
        assert_eq!(logits, vec![1.0; 5]);
    }

    #[test]
    fn test_ban_oob() {
        let filter = BanTokenFilter::new(vec![999]);
        let mut logits = vec![1.0; 5];
        filter.apply(&mut logits, &FilterContext::new());
        assert_eq!(logits, vec![1.0; 5]);
    }

    #[test]
    fn test_context_default() {
        let ctx = FilterContext::default();
        assert_eq!(ctx.position, 0);
        assert!(ctx.generated_ids.is_empty());
    }

    #[test]
    fn test_context_with_ids() {
        let ctx = FilterContext::with_ids(vec![1, 2, 3]);
        assert_eq!(ctx.position, 3);
    }

    #[test]
    fn test_filter_count() {
        let mut pipeline = FilterPipeline::new();
        assert_eq!(pipeline.filter_count(), 0);
        pipeline.add(BanTokenFilter::new(vec![]));
        assert_eq!(pipeline.filter_count(), 1);
    }

    #[test]
    fn test_frequency_no_repeats() {
        let filter = FrequencyPenaltyFilter::new(1.0);
        let ctx = FilterContext::new();
        let mut logits = vec![1.0; 5];
        filter.apply(&mut logits, &ctx);
        assert_eq!(logits, vec![1.0; 5]);
    }
}
