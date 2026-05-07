# MoE support policy

MoE support is not implied by dense decoder support.

A MoE claim requires router logits, top-k expert selection, expert weight loading, shared expert handling when present, active-vs-total parameter accounting, and per-token expert coverage metrics in receipts. Dense one-token receipts must set `moe_claim=false`.
