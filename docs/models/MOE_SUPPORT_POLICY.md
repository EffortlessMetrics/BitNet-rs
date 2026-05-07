# MoE support policy

MoE support is not established by loading a dense sibling. MoE claims require router logits, top-k expert selection, shared expert handling when present, expert weight loading, active/total parameter accounting, and per-token expert coverage metrics.

Until those receipts exist, MoE variants remain future-gated even when dense variants in the same family have text-only smoke tests.

