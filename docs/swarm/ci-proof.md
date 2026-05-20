# Swarm PR Route Proof

This file is intentionally small. It exists to prove that a normal same-repo
BitNet swarm PR can route through the trusted `EM CI Routed Rust` lane after
the source-to-swarm sync.

Proof target:

```text
BitNet Rust Small Result
```

The route must remain `CX53 -> CX43 -> GitHub`; CX33 is not part of the BitNet
rust-small build lane.
