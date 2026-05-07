# MoE decoder lane

MoE support requires:

- Router logits.
- Top-k expert selection.
- Shared expert handling if present.
- Expert weight loading.
- Active vs total parameter receipts.
- Per-token expert coverage metrics.

Dense sibling proof does not prove MoE routing, expert loading, shared experts, or active-parameter accounting.

