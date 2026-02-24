# bitnet-validation

Architecture-aware LayerNorm and projection weight validation rules for BitNet models.

Provides shared validation logic used by `bitnet-cli` and `bitnet-st-tools` to ensure
consistent validation decisions for the same model inputs.

## Modules

- **`rules`** — threshold rules, built-in rulesets, YAML policy loader
- **`names`** — LayerNorm tensor name detection patterns
