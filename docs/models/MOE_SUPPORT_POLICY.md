# MoE Support Policy

MoE support requires router logits, top-k expert selection, expert weight loading, shared expert handling where present, active/total parameter receipts, and per-token expert coverage metrics.

Dense decoder proof cannot be reused as MoE proof. A family with both dense and MoE variants must keep those claims separate.
