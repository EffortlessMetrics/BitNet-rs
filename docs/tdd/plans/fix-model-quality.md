# Fix Plan: Model Quality Issues (Finding 6)

**Status:** 📋 Planning
**Priority:** CRITICAL
**Blocking:** User-facing intelligibility and product viability

**Related:**
- [Investigation Report](../receipts/inference_quality_investigation.md#finding-6)
- [CLAUDE.md Known Issues](../../../CLAUDE.md#known-issues)

---

## Problem Statement

**Symptom:** Models produce garbled/nonsensical output despite mathematically correct inference

**Current explanation (CLAUDE.md):**
> **Model Quality: microsoft-bitnet-b1.58-2B-4T-gguf**
>
> **Status:** Known limitation
> **Symptom:** Non-sensical output in some configurations
>
> - This is a **model quality issue**, not an inference engine bug

**⚠️ Critical question:** Is this ACTUALLY a model weights issue, or are we missing bugs in:
- Tensor layout/stride interpretation
- Attention mechanism implementation
- ROPE (positional encoding) computation
- Layer normalization
- Quantization/dequantization correctness
- Token embedding lookup
- Softmax numerical stability

**We should NOT accept "model quality" as root cause without deep investigation.**

---

## Investigation Plan

### Phase 1: Verify Against C++ Reference 🔍 CRITICAL

**Goal:** Prove whether output differs from bitnet.cpp with same model

**Tasks:**

1. **Install and build bitnet.cpp:**
   ```bash
   # Clone Microsoft's reference implementation
   git clone https://github.com/microsoft/BitNet.git /tmp/bitnet-cpp
   cd /tmp/bitnet-cpp

   # Build (follow their instructions)
   # Note their build requirements
   ```

2. **Run identical prompt on both engines:**

   **BitNet-rs:**
   ```bash
   RUST_LOG=debug cargo run --release -p bitnet-cli --features cpu,full-cli -- run \
     --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
     --tokenizer models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
     --prompt "What is 2+2?" \
     --max-tokens 32 \
     --temperature 0.0 --greedy --seed 42 \
     2>&1 | tee bitnet-rs-output.txt
   ```

   **bitnet.cpp:**
   ```bash
   # Use their CLI with same params
   ./bitnet-cpp-cli \
     --model models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
     --prompt "What is 2+2?" \
     --n-predict 32 \
     --temp 0.0 --seed 42 \
     2>&1 | tee bitnet-cpp-output.txt
   ```

3. **Compare outputs:**
   ```bash
   # Token-by-token comparison
   diff bitnet-rs-output.txt bitnet-cpp-output.txt
   ```

**Expected outcomes:**

**Case A: Outputs match (both garbled)**
- ✅ Confirms model weights issue (not our bug)
- ✅ Can document as known limitation
- ⚠️ Still investigate why this model is broken

**Case B: Outputs differ (C++ is coherent, ours is garbled)**
- 🚨 **CRITICAL BUG IN BITNET.RS**
- Must find where we diverge
- Proceed to Phase 2 immediately

**Case C: Outputs differ (both garbled but differently)**
- 🚨 **BUG IN BOTH IMPLEMENTATIONS**
- Or tokenizer/config mismatch
- Investigate configuration parity

**Acceptance criteria:**
- ✅ We have concrete evidence whether outputs match C++ reference
- ✅ We know if our engine produces different results than reference
- ✅ We can route investigation based on data, not assumptions

---

### Phase 2: Layer-by-Layer Parity (If Case B) 🔬 DEEP DIVE

**Goal:** Find exact layer where outputs diverge from C++ reference

**Prerequisites:** bitnet.cpp installed and confirmed to produce better output

**Tasks:**

1. **Instrument both codebases for intermediate outputs:**

   **In BitNet-rs:**
   ```rust
   // Add to forward pass
   tracing::debug!("Layer {}: output norm = {:.6}", layer_idx, output.norm());
   tracing::debug!("Layer {}: first 10 values = {:?}", layer_idx, &output[..10]);
   ```

   **In bitnet.cpp:**
   ```cpp
   // Add similar logging
   printf("Layer %d: output norm = %.6f\n", layer_idx, compute_norm(output));
   ```

2. **Run both with same input and compare per-layer:**
   ```bash
   # BitNet-rs with debug output
   RUST_LOG=debug cargo run --release -p bitnet-cli -- run \
     --model model.gguf --prompt "test" --max-tokens 1 \
     2>&1 | grep "Layer" > bitnet-rs-layers.txt

   # bitnet.cpp with debug output
   ./bitnet-cpp-cli --model model.gguf --prompt "test" --n-predict 1 \
     2>&1 | grep "Layer" > bitnet-cpp-layers.txt

   # Compare
   diff bitnet-rs-layers.txt bitnet-cpp-layers.txt
   ```

3. **Binary search for divergence:**
   - If layer N matches but layer N+1 diverges → bug is in layer N+1 implementation
   - Focus investigation on that specific layer

4. **Check specific operations in divergent layer:**

   **Attention:**
   ```rust
   // Verify Q, K, V computation
   // Verify attention scores
   // Verify attention weights (post-softmax)
   // Verify attention output
   ```

   **Layer normalization:**
   ```rust
   // Verify mean, variance computation
   // Verify normalization formula
   // Verify gamma, beta application
   ```

   **FFN (feed-forward):**
   ```rust
   // Verify gate computation
   // Verify up/down projection
   // Verify activation function (SiLU, GELU, etc.)
   ```

**Acceptance criteria:**
- ✅ We identify exact layer where divergence occurs
- ✅ We narrow down to specific operation (attention, FFN, norm)
- ✅ We have concrete numerical difference to debug

---

### Phase 3: Check Tensor Interpretation 📐 CORRECTNESS

**Goal:** Verify we're reading GGUF tensors correctly (layout, strides, quantization)

**Potential issues:**

1. **Tensor shape misinterpretation:**
   ```rust
   // Are we reading [hidden_dim, vocab_size] as [vocab_size, hidden_dim]?
   // Are we transposing when we shouldn't (or vice versa)?
   ```

   **Test:**
   ```rust
   #[test]
   fn test_tensor_shapes_match_gguf_metadata() {
       let model = load_model("test.gguf")?;

       // Check embedding table
       assert_eq!(model.token_embeddings.shape(), &[vocab_size, hidden_dim]);

       // Check output projection
       assert_eq!(model.output.shape(), &[hidden_dim, vocab_size]);

       // Check attention weights
       assert_eq!(model.layers[0].q_proj.shape(), &[hidden_dim, hidden_dim]);
   }
   ```

2. **Quantization format mismatch:**
   ```rust
   // Are we dequantizing I2_S correctly?
   // Are scales applied in the right order?
   // Are we handling signed vs unsigned correctly?
   ```

   **Test against known values:**
   ```rust
   #[test]
   fn test_i2s_dequant_matches_reference() {
       // Quantize known FP32 values
       let fp32 = vec![1.0, -1.0, 0.5, -0.5];
       let quantized = quantize_i2s(&fp32);

       // Dequantize
       let dequantized = dequantize_i2s(&quantized);

       // Should match within tolerance
       assert_approx_eq(&fp32, &dequantized, 0.01);
   }
   ```

3. **Stride/padding issues:**
   ```rust
   // Are we accounting for alignment padding?
   // Are we skipping padding bytes correctly?
   ```

4. **Endianness:**
   ```rust
   // GGUF is little-endian by spec
   // Are we reading multi-byte values correctly?
   ```

**Acceptance criteria:**
- ✅ Tensor shapes match GGUF metadata exactly
- ✅ Quantization round-trip tests pass
- ✅ Known-value tests pass (e.g., identity matrix → identity output)

---

### Phase 4: Validate Attention Implementation 🎯 HIGH RISK AREA

**Goal:** Verify attention mechanism is correct (common source of subtle bugs)

**Areas to check:**

1. **ROPE (Rotary Position Embedding):**
   ```rust
   // Verify theta parameter (default 10000.0, but can vary)
   // Verify frequency computation
   // Verify sin/cos application to Q, K
   ```

   **Test:**
   ```rust
   #[test]
   fn test_rope_matches_reference() {
       let theta = 10000.0;
       let pos = 5;
       let dim = 64;

       let (sin, cos) = compute_rope(pos, dim, theta);

       // Compare against reference implementation
       let (ref_sin, ref_cos) = reference_rope(pos, dim, theta);
       assert_approx_eq(&sin, &ref_sin, 1e-6);
       assert_approx_eq(&cos, &ref_cos, 1e-6);
   }
   ```

2. **Attention score computation:**
   ```rust
   // scores = Q @ K^T / sqrt(head_dim)
   // Verify scaling factor
   // Verify matrix multiplication correctness
   ```

3. **Causal masking:**
   ```rust
   // Are we masking future positions correctly?
   // Are we using -inf or a large negative number?
   ```

   **Test:**
   ```rust
   #[test]
   fn test_causal_mask_prevents_future_attention() {
       let scores = compute_attention_scores(q, k);
       let masked = apply_causal_mask(scores, seq_len);

       // Future positions should have -inf or very negative scores
       for i in 0..seq_len {
           for j in (i+1)..seq_len {
               assert!(masked[[i, j]] < -1e9);
           }
       }
   }
   ```

4. **Softmax numerical stability:**
   ```rust
   // Are we subtracting max before exp?
   // Are we handling -inf correctly?
   ```

5. **Multi-head attention:**
   ```rust
   // Are we splitting heads correctly?
   // Are we concatenating outputs correctly?
   // Are we handling GQA (grouped query attention) if present?
   ```

**Acceptance criteria:**
- ✅ ROPE computation matches reference
- ✅ Attention scores are numerically stable
- ✅ Causal masking works correctly
- ✅ Multi-head mechanics are correct

---

### Phase 5: Check Tokenizer Parity (Again, Deeper) 🔤 SUBTLE BUGS

**Goal:** Verify tokenizer produces EXACT same token IDs as C++ for all inputs

**Tasks:**

1. **Cross-validate on diverse inputs:**
   ```bash
   # Generate test cases
   echo "Hello world" > test-inputs.txt
   echo "What is 2+2?" >> test-inputs.txt
   echo "The capital of France is Paris." >> test-inputs.txt
   echo "   Leading spaces" >> test-inputs.txt
   echo "Special tokens: <s> </s> <|endoftext|>" >> test-inputs.txt
   ```

2. **Tokenize with both implementations:**

   **BitNet-rs:**
   ```rust
   for line in test_inputs {
       let tokens = tokenizer.encode(line, true, false)?;
       println!("Rust: {:?}", tokens);
   }
   ```

   **bitnet.cpp:**
   ```cpp
   for line in test_inputs {
       auto tokens = tokenizer.encode(line, true, false);
       printf("C++: [");
       for (auto t : tokens) printf("%d, ", t);
       printf("]\n");
   }
   ```

3. **Compare token IDs exactly:**
   ```bash
   diff rust-tokens.txt cpp-tokens.txt
   ```

4. **Check special token handling:**
   ```rust
   #[test]
   fn test_special_tokens_match_cpp() {
       let tokenizer = load_tokenizer(...)?;

       // BOS
       assert_eq!(tokenizer.bos_token_id(), Some(1)); // Or whatever C++ returns

       // EOS
       assert_eq!(tokenizer.eos_token_id(), Some(2));

       // Token-to-ID lookup
       assert_eq!(tokenizer.token_to_id("</s>"), Some(2));
   }
   ```

5. **Check decoding:**
   ```rust
   #[test]
   fn test_decode_matches_cpp() {
       let token_ids = vec![1, 15043, 995, 2]; // "Hello world" + BOS/EOS

       let rust_text = tokenizer.decode(&token_ids)?;
       let cpp_text = reference_tokenizer.decode(&token_ids)?;

       assert_eq!(rust_text, cpp_text);
   }
   ```

**Acceptance criteria:**
- ✅ Token IDs match C++ reference for 100+ diverse inputs
- ✅ Special tokens (BOS/EOS) match exactly
- ✅ Decoding produces identical text

---

### Phase 6: Check Numerical Precision 🔢 SUBTLE

**Goal:** Ensure FP32/FP16 precision handling doesn't cause divergence

**Tasks:**

1. **Check mixed precision:**
   ```rust
   // Are we using FP16 where we should use FP32?
   // Are we losing precision in intermediate computations?
   ```

2. **Verify accumulator types:**
   ```rust
   // For matmul: are we using FP32 accumulators even with FP16 inputs?
   fn matmul_f16(a: &[f16], b: &[f16]) -> Vec<f32> {
       let mut result = vec![0.0f32; ...]; // FP32 accumulator
       // ...
   }
   ```

3. **Check for catastrophic cancellation:**
   ```rust
   // Layer norm: variance = E[x²] - E[x]²
   // Can lose precision if E[x]² ≈ E[x²]

   // Use numerically stable formula:
   // variance = E[(x - E[x])²]
   ```

4. **Test with known pathological cases:**
   ```rust
   #[test]
   fn test_numerical_stability() {
       // Very large values
       let large = vec![1e6; 1000];
       let output = layer_norm(&large);
       assert!(output.iter().all(|&x| x.is_finite()));

       // Very small values
       let small = vec![1e-6; 1000];
       let output = layer_norm(&small);
       assert!(output.iter().all(|&x| x.is_finite()));
   }
   ```

**Acceptance criteria:**
- ✅ No NaN or Inf in outputs
- ✅ Precision loss is acceptable (<1e-4 relative error)
- ✅ Numerically stable on pathological inputs

---

## Fix Strategies (Based on Investigation Results)

### Strategy A: Fix Attention Bug

**If Phase 4 reveals attention divergence:**

1. **Match ROPE implementation exactly to reference:**
   ```rust
   // Copy reference formula exactly
   // Verify theta value from GGUF metadata
   ```

2. **Fix causal masking:**
   ```rust
   // Ensure masking happens before softmax
   // Use -inf for masked positions (not just large negative)
   ```

3. **Verify multi-head split/concat:**
   ```rust
   // heads = reshape(x, [batch, seq, n_heads, head_dim])
   // NOT: heads = split(x, n_heads) # Wrong!
   ```

**Estimated effort:** 1-2 days

---

### Strategy B: Fix Tensor Layout Bug

**If Phase 3 reveals shape/stride issues:**

1. **Add shape validation:**
   ```rust
   fn validate_tensor_shapes(model: &Model) -> Result<()> {
       // Check every tensor shape against GGUF metadata
       // Fail fast if mismatch
   }
   ```

2. **Fix transpose logic:**
   ```rust
   // Document expected layouts clearly
   // W @ x vs x @ W^T
   ```

3. **Add integration test:**
   ```rust
   #[test]
   fn test_forward_pass_with_known_weights() {
       // Load model with identity matrices as weights
       // Verify output equals input (modulo activations)
   }
   ```

**Estimated effort:** 1-3 days

---

### Strategy C: Fix Quantization Bug

**If Phase 3 reveals quantization issues:**

1. **Match dequantization formula exactly:**
   ```rust
   // Get reference formula from GGUF spec or bitnet.cpp
   // Implement bit-for-bit identical
   ```

2. **Add numerical validation:**
   ```rust
   #[test]
   fn test_quantization_accuracy() {
       let fp32 = generate_random_tensor();
       let quantized = quantize(&fp32);
       let dequantized = dequantize(&quantized);

       let mse = mean_squared_error(&fp32, &dequantized);
       assert!(mse < 0.01); // Threshold from spec
   }
   ```

**Estimated effort:** 1-2 days

---

### Strategy D: Fix Tokenizer Parity

**If Phase 5 reveals tokenizer differences:**

1. **Match HuggingFace tokenizer exactly:**
   ```rust
   // Use same BPE merges order
   // Use same pre-tokenizer (ByteLevel with add_prefix_space)
   // Use same special token handling
   ```

2. **Add cross-validation test:**
   ```rust
   #[test]
   fn test_tokenizer_parity_with_hf() {
       let hf = HuggingFaceTokenizer::from_pretrained("...");
       let ours = BitNetTokenizer::from_file("...");

       for text in test_cases {
           assert_eq!(hf.encode(text), ours.encode(text));
       }
   }
   ```

**Estimated effort:** 1 day

---

### Strategy E: Document Model Limitation (If Case A)

**If C++ reference ALSO produces garbled output:**

1. **Document in CLAUDE.md:**
   ```markdown
   ## Known Model Issues

   ### microsoft-bitnet-b1.58-2B-4T-gguf

   **Confirmed:** Produces garbled output in both BitNet-rs and bitnet.cpp

   **Root cause:** Model weights quality (upstream issue)

   **Workaround:** Use alternative BitNet models or expect poor quality
   ```

2. **Add warning to CLI:**
   ```rust
   if model_name.contains("microsoft-bitnet-b1.58") {
       eprintln!("⚠️  Warning: This model is known to produce poor quality output.");
       eprintln!("   This is a model weights issue, not a BitNet-rs bug.");
   }
   ```

3. **Seek alternative models:**
   - Test other BitNet checkpoints
   - Try community-converted models
   - Document which models work well

**Estimated effort:** Half day

---

## Acceptance Criteria for "Fixed"

**Minimum (confirm not our bug):**
- ✅ Proven that BitNet-rs matches C++ reference output
- ✅ Documented which models work vs don't work
- ✅ Clear user guidance on model selection

**Good (working Q&A):**
- ✅ At least one BitNet model produces coherent Q&A output
- ✅ Greedy decoding is deterministic and sensible
- ✅ Template selection produces expected formatting

**Excellent (production quality):**
- ✅ Multiple BitNet models produce high-quality output
- ✅ Output quality competitive with standard LLMs
- ✅ User satisfaction with intelligibility

---

## Open Questions

1. **Does bitnet.cpp produce coherent output with microsoft-bitnet model?**
   - Answer via: Phase 1 testing

2. **If outputs differ, where exactly do they diverge?**
   - Answer via: Phase 2 layer-by-layer parity

3. **Are there other BitNet models that work better?**
   - Answer via: Test alternative models

4. **Is the model itself actually broken, or just poorly suited for Q&A?**
   - Answer via: Test with completion tasks instead of Q&A

5. **Could this be a template/prompt engineering issue rather than model quality?**
   - Answer via: Extensive prompt template experimentation

---

**Next Action:** Execute Phase 1 (Verify Against C++ Reference) - this is CRITICAL to route investigation

**Owner:** TBD
**Due Date:** TBD
**Tracking Issue:** Create GitHub issue for this investigation
