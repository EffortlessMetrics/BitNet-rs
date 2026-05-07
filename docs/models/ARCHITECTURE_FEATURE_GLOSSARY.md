# Architecture feature glossary

Features in this glossary are shared vocabulary for family docs. Anything not source-backed must be marked `TBD`, not inferred.

- `dense_decoder`: autoregressive decoder with dense feed-forward layers.
- `moe_decoder`: autoregressive decoder with routed expert feed-forward layers.
- `effective_parameters`: parameters materially used for a task or configuration; must be source-backed or receipt-backed.
- `active_parameters`: per-token or per-forward routed parameters in MoE models; not the same as total parameters.
- `hybrid_thinking`: model family supports both reasoning/thinking and direct-answer modes.
- `thinking_mode`: prompt/runtime mode that exposes or controls model reasoning behavior.
- `reasoning_effort`: request-level control over amount of reasoning, such as `none` or `high`.
- `developer_role`: chat role distinct from system/user/assistant for coding or agentic instructions.
- `tool_calling`: structured tool invocation support.
- `structured_outputs`: constrained JSON/schema output behavior.
- `multimodal_text_image`: accepts text and image input.
- `multimodal_audio`: accepts audio input.
- `video_as_frames`: represents video as extracted image frames; preprocessing details are TBD unless source-backed.
- `vision_encoder`: image encoder or projector path for multimodal models.
- `audio_encoder`: audio encoder path for multimodal models.
- `per_layer_embeddings`: layer-specific embedding behavior such as PLE.
- `shared_kv_cache`: shared or unified K/V behavior across attention layers.
- `sliding_window_attention`: local attention window rather than full attention at every layer.
- `global_attention`: full-context/global attention layer.
- `p_rope`: positional RoPE variant named by source notes.
- `yarn`: YaRN context-extension mechanism; runtime support is TBD until proven.
- `long_context`: context lengths above ordinary local smoke sizes; runtime support requires receipt.
- `compressed_sparse_attention`: CSA-style sparse attention; implementation details are TBD until source and parity proof exist.
- `heavily_compressed_attention`: HCA-style compressed attention; implementation details are TBD until source and parity proof exist.
- `mamba_or_hybrid_state_space_tbd`: possible state-space/hybrid feature that must remain TBD unless source-backed.
- `mtp_drafter`: multi-token prediction draft model for speculative decoding.
- `eagle_drafter`: EAGLE draft model for speculative decoding.
- `gguf_dynamic_quant`: GGUF dynamic quantization reference; not a native kernel claim.
- `mxfp4`: 4-bit floating-point format; kernel support must be separately proven.
- `fp4_fp8_mixed`: mixed FP4/FP8 precision; kernel support must be separately proven.
- `token_classification`: per-token labeling task, not autoregressive generation.
- `bioes_labels`: begin/inside/outside/end/single span labeling taxonomy.
- `viterbi_span_decoder`: constrained decoding over token labels to produce coherent spans.
