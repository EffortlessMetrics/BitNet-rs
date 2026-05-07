# Architecture feature glossary

Unknowns must be marked `TBD`, not inferred. Family docs should link concepts to these terms and state whether the fact is source-backed, partial, or design-only.

| Term | Meaning |
|---|---|
| `dense_decoder` | Autoregressive decoder with dense feed-forward blocks. |
| `moe_decoder` | Autoregressive decoder with router-selected expert feed-forward blocks. |
| `effective_parameters` | Parameters used for planning footprint or runtime; definition must be source-specific. |
| `active_parameters` | Parameters active per token in MoE or sparse models. |
| `hybrid_thinking` | Model supports thinking and non-thinking modes. |
| `thinking_mode` | Prompt/template mode that exposes or controls hidden/visible reasoning behavior. |
| `reasoning_effort` | Request-level reasoning control such as none/high. |
| `developer_role` | Prompt role distinct from system/user/assistant for coding or agent tools. |
| `tool_calling` | Model/template supports function or tool call syntax. |
| `structured_outputs` | Model/template supports JSON or constrained structured output. |
| `multimodal_text_image` | Text plus image input with text output unless otherwise stated. |
| `multimodal_audio` | Audio input support; preprocessing and encoder are separate proof items. |
| `video_as_frames` | Video handled as sampled image frames, not native video unless proven. |
| `vision_encoder` | Vision tower/projector path; not implied by text decoder support. |
| `audio_encoder` | Audio encoder path; not implied by text decoder support. |
| `per_layer_embeddings` | Layer-specific embeddings such as Gemma small-model PLE. |
| `shared_kv_cache` | K/V sharing mechanism for attention layers; implementation details are source-specific. |
| `sliding_window_attention` | Attention constrained to a moving local window. |
| `global_attention` | Attention layer with broader/global token reach. |
| `p_rope` | Positional RoPE variant; exact math is model-specific until source-backed. |
| `yarn` | YaRN context extension; source config and receipt required. |
| `long_context` | Context claims beyond ordinary smoke sizes. |
| `compressed_sparse_attention` | CSA; design note until implemented and parity-tested. |
| `heavily_compressed_attention` | HCA; design note until implemented and parity-tested. |
| `mamba_or_hybrid_state_space_tbd` | Possible state-space/hybrid feature; always TBD unless source-backed. |
| `mtp_drafter` | Multi-token prediction draft model for speculative decoding. |
| `eagle_drafter` | EAGLE draft model for speculative decoding. |
| `gguf_dynamic_quant` | Dynamic GGUF quantization requiring exact quant recipe and receipt. |
| `mxfp4` | MXFP4 precision; kernel/storage support must be proven separately. |
| `fp4_fp8_mixed` | Mixed FP4/FP8 precision; loader and kernel support are future-gated. |
| `token_classification` | Per-token labeling task; not generation. |
| `bioes_labels` | Begin/Inside/Outside/End/Singleton span-label scheme. |
| `viterbi_span_decoder` | Constrained decoding for coherent token spans. |
