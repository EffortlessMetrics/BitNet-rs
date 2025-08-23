#!/usr/bin/env python3
"""
Render performance JSON to markdown with Methods & Environment box
This ensures all performance docs are generated from measured data only
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

def format_number(num: float, precision: int = 2) -> str:
    """Format number with appropriate precision"""
    if num >= 1000:
        return f"{num:,.0f}"
    elif num >= 10:
        return f"{num:.1f}"
    else:
        return f"{num:.{precision}f}"

def get_methods_environment_box(data: Dict[str, Any]) -> str:
    """Generate the Methods & Environment box"""
    meta = data.get('metadata', {})
    
    # Get system info
    platform = meta.get('platform', 'Unknown')
    bitnet_version = meta.get('bitnet_version', 'Unknown')
    rust_version = meta.get('rust_version', 'Unknown')
    python_version = meta.get('python_version', 'Unknown')
    
    # Get library versions
    transformers = meta.get('transformers_version', 'n/a')
    torch = meta.get('torch_version', 'n/a')
    
    # Get determinism settings
    deterministic = meta.get('deterministic', False)
    seed = meta.get('seed', 42)
    threads = meta.get('threads', 1)
    
    # Get test parameters
    prompts = meta.get('num_prompts', 'N')
    max_tokens = meta.get('max_new_tokens', 128)
    warmup = meta.get('warmup_runs', 1)
    iterations = meta.get('iterations', 10)
    
    box = f"""
## Methods & Environment

```
Platform: {platform}
BitNet CLI: {bitnet_version} | Rust: {rust_version} | Python: {python_version}
Transformers: {transformers} | Torch: {torch}
Determinism: BITNET_DETERMINISTIC={1 if deterministic else 0} BITNET_SEED={seed} RAYON_NUM_THREADS={threads}
Prompts: {prompts} fixed, max_new_tokens={max_tokens}, warmup={warmup}, medians over {iterations} runs
Timestamp: {meta.get('timestamp', datetime.utcnow().isoformat())}
```
"""
    return box

def render_performance_table(measurements: Dict[str, Any]) -> str:
    """Render performance measurements as markdown table"""
    
    table = """
## Performance Metrics

| Metric | Median | P95 | Min | Max | StdDev |
|--------|--------|-----|-----|-----|--------|
"""
    
    # Define metric display names and units
    metrics = {
        'tokens_per_second': ('Tokens/sec', ''),
        'time_to_first_token': ('First Token', 'ms'),
        'memory_mb': ('Memory', 'MB'),
        'latency_per_token': ('Token Latency', 'ms'),
    }
    
    for key, (name, unit) in metrics.items():
        if key in measurements:
            data = measurements[key]
            if isinstance(data, dict):
                median = format_number(data.get('median', 0))
                p95 = format_number(data.get('p95', 0))
                min_val = format_number(data.get('min', 0))
                max_val = format_number(data.get('max', 0))
                stddev = format_number(data.get('stddev', 0))
                
                if unit:
                    table += f"| {name} ({unit}) | {median} | {p95} | {min_val} | {max_val} | {stddev} |\n"
                else:
                    table += f"| {name} | {median} | {p95} | {min_val} | {max_val} | {stddev} |\n"
            elif isinstance(data, (int, float)):
                # Handle single values
                val = format_number(data)
                if unit:
                    table += f"| {name} ({unit}) | {val} | - | - | - | - |\n"
                else:
                    table += f"| {name} | {val} | - | - | - | - |\n"
    
    return table

def render_model_info(data: Dict[str, Any]) -> str:
    """Render model information section"""
    model = data.get('model', {})
    
    info = f"""
## Model Information

- **Model ID**: {model.get('id', 'Unknown')}
- **Format**: {model.get('format', 'Unknown')}
- **Size**: {model.get('size_mb', 0):.1f} MB
- **Parameters**: {model.get('parameters', 'Unknown')}
- **Quantization**: {model.get('quantization', 'None')}
- **Tokenizer**: {model.get('tokenizer_type', 'Unknown')}
"""
    
    if 'scoring_policy' in model:
        policy = model['scoring_policy']
        info += f"""
### Scoring Policy
- Add BOS: {policy.get('add_bos', False)}
- Append EOS: {policy.get('append_eos', False)}
- Mask Padding: {policy.get('mask_pad', True)}
"""
    
    return info

def render_validation_results(data: Dict[str, Any]) -> str:
    """Render validation results if present"""
    validation = data.get('validation', {})
    
    if not validation:
        return ""
    
    results = """
## Validation Results

| Check | Status | Value | Threshold | Details |
|-------|--------|-------|-----------|---------|
"""
    
    # Tokenizer parity
    if 'tokenizer_parity' in validation:
        tp = validation['tokenizer_parity']
        status = "✅ Pass" if tp.get('pass', False) else "❌ Fail"
        results += f"| Tokenizer Parity | {status} | {tp.get('differences', 0)} diffs | 0 | {tp.get('details', '')} |\n"
    
    # Logit correlation
    if 'logit_correlation' in validation:
        lc = validation['logit_correlation']
        tau_b = lc.get('median_tau_b', 0)
        threshold = lc.get('threshold', 0.95)
        status = "✅ Pass" if tau_b >= threshold else "❌ Fail"
        results += f"| Logit τ-b | {status} | {tau_b:.3f} | ≥{threshold} | {lc.get('samples', 0)} samples |\n"
    
    # NLL parity
    if 'nll_parity' in validation:
        nll = validation['nll_parity']
        delta = abs(nll.get('delta_mean_nll', 0))
        threshold = nll.get('threshold', 0.01)
        status = "✅ Pass" if delta <= threshold else "❌ Fail"
        results += f"| NLL Parity | {status} | Δ={delta:.4f} | ≤{threshold} | {nll.get('tokens', 0)} tokens |\n"
    
    return results

def render_charts(measurements: Dict[str, Any]) -> str:
    """Render ASCII charts for key metrics"""
    
    charts = """
## Performance Trends

### Tokens per Second Distribution
"""
    
    if 'tokens_per_second' in measurements:
        tps = measurements['tokens_per_second']
        if 'distribution' in tps:
            # Simple ASCII histogram
            dist = tps['distribution']
            max_count = max(dist.values()) if dist else 1
            
            for bucket, count in sorted(dist.items()):
                bar_len = int((count / max_count) * 40)
                bar = '█' * bar_len
                charts += f"{bucket:>6}: {bar} {count}\n"
    
    return charts

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <performance.json>", file=sys.stderr)
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Extract format from filename
    format_type = "Unknown"
    if "safetensors" in json_file.lower():
        format_type = "SafeTensors"
    elif "gguf" in json_file.lower():
        format_type = "GGUF"
    
    # Generate markdown
    output = f"""# BitNet.rs Performance Report - {format_type}

{get_methods_environment_box(data)}

{render_model_info(data)}

{render_performance_table(data.get('measurements', {}))}

{render_validation_results(data)}

{render_charts(data.get('measurements', {}))}

## Raw Measurements

<details>
<summary>Click to expand raw JSON data</summary>

```json
{json.dumps(data, indent=2)}
```

</details>

---

*Generated from measured data: {os.path.basename(json_file)}*  
*Report generated: {datetime.utcnow().isoformat()}Z*
"""
    
    print(output)

if __name__ == "__main__":
    main()