# MoE decoder lane

MoE support requires:

- router logits
- top-k expert selection
- shared expert handling if present
- expert weight loading
- active vs total parameter receipts
- per-token expert coverage metrics

Dense decoder receipts do not prove MoE routing. MoE families stay `design_scaffold` or `design_only` until routing, expert loading, and token coverage are smoke-tested and receipt-backed for the named variant.
