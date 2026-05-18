/// Context allocation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Fixed: prompt gets up to max_prompt, rest for generation.
    Fixed { max_prompt: usize },
    /// Dynamic: generation gets at least min_gen tokens.
    Dynamic { min_generation: usize },
    /// Even split between prompt and generation.
    EvenSplit,
}

/// Compute prompt and generation budgets.
pub fn compute_budgets(
    max_context: usize,
    prompt_len: usize,
    strategy: AllocationStrategy,
) -> (usize, usize) {
    match strategy {
        AllocationStrategy::Fixed { max_prompt } => {
            let prompt = prompt_len.min(max_prompt);
            let generation = max_context.saturating_sub(prompt);
            (prompt, generation)
        }
        AllocationStrategy::Dynamic { min_generation } => {
            let generation = min_generation.max(max_context.saturating_sub(prompt_len));
            let prompt = max_context.saturating_sub(generation);
            (prompt.min(prompt_len), generation)
        }
        AllocationStrategy::EvenSplit => {
            let half = max_context / 2;
            (prompt_len.min(half), half)
        }
    }
}
