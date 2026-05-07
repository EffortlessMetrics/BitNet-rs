# MoE Decoder Lane

MoE support requires:
- router logits
- top-k expert selection
- shared expert handling if present
- expert weight loading
- active vs total parameter receipts
- per-token expert coverage metrics

Dense receipts are not MoE receipts. Expert routing, expert loading, and shared-expert behavior must be proven separately for every claimed family/variant/backend.
