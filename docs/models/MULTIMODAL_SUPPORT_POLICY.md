# Multimodal support policy

A text-only proof for a multimodal family proves only the text-only path. Image, audio, and video support require separate loader, preprocessing, projector/encoder, prompt-template, and receipt coverage.

## Required multimodal boundaries

- `multimodal_claim=false` for text-only proofs.
- Projector or encoder artifact identity when used.
- Modality-specific input shape and preprocessing receipt.
- Separate receipts for image, audio, and video-as-frames paths.

