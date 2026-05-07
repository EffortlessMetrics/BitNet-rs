# Multimodal lane

This lane covers models with text plus image, audio, or video/frame inputs.

## Required gates

- Modality-specific tokenizer or processor source.
- Vision/audio/projector weight loading.
- Prompt-template placeholders and role rules.
- Text-only receipts that explicitly set `multimodal_claim=false`.
- Modality receipts that record preprocessing, encoder/projector artifacts, and fallback status.

A text-only proof for a multimodal family does not prove image, audio, video, projector, or encoder support.
