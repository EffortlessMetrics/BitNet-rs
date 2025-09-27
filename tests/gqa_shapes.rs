//! GQA shape validation tests

#[cfg(test)]
mod tests {
    /// Test GQA shape calculations match Microsoft BitNet 2B
    #[test]
    fn gqa_shapes_match_microsoft_2b() {
        let hidden = 2560usize;
        let n_heads = 40usize;
        let head_dim = hidden / n_heads; // 64
        let n_kv_heads = 10usize;

        assert_eq!(head_dim, 64);
        assert_eq!(n_heads % n_kv_heads, 0);

        let group_size = n_heads / n_kv_heads; // 4
        let kv_out = n_kv_heads * head_dim; // 640

        assert_eq!(group_size, 4);
        assert_eq!(kv_out, 640);

        // This should match the k_proj.weight shape [2560, 640] from Microsoft 2B GGUF
        println!("Microsoft 2B GQA validation:");
        println!("  hidden_size: {}", hidden);
        println!("  n_heads: {}", n_heads);
        println!("  n_kv_heads: {}", n_kv_heads);
        println!("  head_dim: {}", head_dim);
        println!("  group_size: {}", group_size);
        println!("  kv_out (k_proj/v_proj out_dim): {}", kv_out);
    }

    /// Test GQA validation logic
    #[test]
    fn gqa_validation_logic() {
        // Valid GQA configurations
        assert_valid_gqa(32, 32); // MHA
        assert_valid_gqa(32, 8); // GQA with 4 groups
        assert_valid_gqa(32, 1); // MQA

        // Invalid configurations
        assert_invalid_gqa(32, 0); // zero KV heads
        assert_invalid_gqa(32, 33); // more KV heads than Q heads
        assert_invalid_gqa(32, 7); // not divisible
    }

    fn assert_valid_gqa(n_heads: usize, n_kv_heads: usize) {
        assert!(n_kv_heads > 0, "n_kv_heads must be > 0");
        assert!(n_kv_heads <= n_heads, "n_kv_heads must be <= n_heads");
        assert_eq!(n_heads % n_kv_heads, 0, "n_heads must be divisible by n_kv_heads");
    }

    fn assert_invalid_gqa(n_heads: usize, n_kv_heads: usize) {
        let valid = n_kv_heads > 0 && n_kv_heads <= n_heads && n_heads.is_multiple_of(n_kv_heads);
        assert!(
            !valid,
            "Configuration should be invalid: n_heads={}, n_kv_heads={}",
            n_heads, n_kv_heads
        );
    }

    /// Test common GQA configurations used in practice
    #[test]
    fn common_gqa_configurations() {
        // LLaMA 2 configurations
        test_gqa_config("LLaMA2-7B", 4096, 32, 32);
        test_gqa_config("LLaMA2-13B", 5120, 40, 40);
        test_gqa_config("LLaMA2-70B", 8192, 64, 8); // GQA

        // Mistral configurations
        test_gqa_config("Mistral-7B", 4096, 32, 8); // GQA

        // Microsoft BitNet configurations
        test_gqa_config("BitNet-2B", 2560, 40, 10); // GQA

        // MQA example
        test_gqa_config("MQA-Example", 2048, 16, 1); // MQA
    }

    fn test_gqa_config(name: &str, hidden_size: usize, n_heads: usize, n_kv_heads: usize) {
        println!(
            "Testing {}: hidden={}, heads={}, kv_heads={}",
            name, hidden_size, n_heads, n_kv_heads
        );

        let head_dim = hidden_size / n_heads;
        let group_size = n_heads / n_kv_heads;
        let kv_out = n_kv_heads * head_dim;

        // Validate basic constraints
        assert_eq!(hidden_size % n_heads, 0, "{}: hidden_size not divisible by n_heads", name);
        assert!(n_kv_heads > 0, "{}: n_kv_heads must be > 0", name);
        assert!(n_kv_heads <= n_heads, "{}: n_kv_heads must be <= n_heads", name);
        assert_eq!(n_heads % n_kv_heads, 0, "{}: n_heads not divisible by n_kv_heads", name);

        // Validate derived values
        assert_eq!(group_size * n_kv_heads, n_heads, "{}: group_size calculation incorrect", name);
        assert_eq!(kv_out, n_kv_heads * head_dim, "{}: kv_out calculation incorrect", name);

        println!("  ✓ head_dim: {}, group_size: {}, kv_out: {}", head_dim, group_size, kv_out);
    }
}
