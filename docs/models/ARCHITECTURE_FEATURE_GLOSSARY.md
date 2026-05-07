# Architecture Feature Glossary

Terms in family docs should refer back here. Anything not source-backed must be marked `TBD`, not inferred.

| Term | Definition |
|---|---|
| `dense_decoder` | Autoregressive decoder with dense feed-forward blocks. |
| `moe_decoder` | Autoregressive decoder with routed mixture-of-experts feed-forward blocks. |
| `effective_parameters` | Parameter count represented by a model artifact or quantized form. |
| `active_parameters` | Parameters used per token in sparse/MoE execution. |
| `hybrid_thinking` | Family exposes thinking and non-thinking behavior. |
| `thinking_mode` | Prompt/template mode that requests explicit hidden/reasoning-style output when supported. |
| `reasoning_effort` | Request-level control over reasoning budget or style. |
| `developer_role` | Chat role distinct from system/user/assistant for agentic instruction layering. |
| `tool_calling` | Template/API support for tool or function calls. |
| `structured_outputs` | Output constrained to JSON/schema-like formats. |
| `multimodal_text_image` | Text and image input, generally text output unless otherwise receipt-backed. |
| `multimodal_audio` | Audio input support; requires separate encoder/receipt proof. |
| `video_as_frames` | Video handled through sampled image frames; TBD unless source-backed. |
| `vision_encoder` | Visual feature encoder or projector path. |
| `audio_encoder` | Audio feature encoder path. |
| `per_layer_embeddings` | PLE; layer-specific embedding behavior noted by the model family. |
| `shared_kv_cache` | Global layers share or unify K/V cache behavior. |
| `sliding_window_attention` | Local-window attention over a bounded neighborhood. |
| `global_attention` | Attention layers that can see broader/global context. |
| `p_rope` | p-RoPE positional encoding variant. |
| `yarn` | YaRN context extension; runtime support is future-gated unless receipted. |
| `long_context` | Context beyond normal small smoke lengths; not a runtime claim by itself. |
| `compressed_sparse_attention` | CSA architecture note; implementation details TBD until source and parity backed. |
| `heavily_compressed_attention` | HCA architecture note; implementation details TBD until source and parity backed. |
| `mamba_or_hybrid_state_space_tbd` | Placeholder for state-space/hybrid blocks; must remain TBD until source-backed. |
| `mtp_drafter` | Multi-token prediction draft model for speculative decoding. |
| `eagle_drafter` | EAGLE draft model for speculative decoding. |
| `gguf_dynamic_quant` | Dynamic GGUF quantization reference; not a kernel or quality claim. |
| `mxfp4` | MXFP4 precision format; kernel support TBD unless receipted. |
| `fp4_fp8_mixed` | Mixed FP4/FP8 precision; kernel support TBD unless receipted. |
| `token_classification` | Per-token labeling task, not generation. |
| `bioes_labels` | Begin/Inside/Outside/End/Singleton span tag taxonomy. |
| `viterbi_span_decoder` | Constrained decoding over token labels to coherent spans. |
