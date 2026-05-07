# MoE decoder lane

MoE support requires:

- router logits;
- top-k expert selection;
- shared expert handling if present;
- expert weight loading;
- active vs total parameter receipts;
- per-token expert coverage metrics.

A dense decoder smoke does not prove MoE support. A source-backed active-parameter count does not prove runtime routing. MoE claims advance only when routing, expert loading, and per-token coverage are receipt-backed for the named model and variant.
