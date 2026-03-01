#!/usr/bin/env python3
import json, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / 'docs/coverage.md'

def run_json(cmd):
    p = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return json.loads(p.stdout)

summary = None
error = None
if (root / 'target/llvm-cov-target/llvm-cov.json').exists():
    summary = json.loads((root / 'target/llvm-cov-target/llvm-cov.json').read_text())
else:
    try:
      summary = run_json(['cargo','llvm-cov','--workspace','--summary-only','--json'])
    except Exception as e:
      error = str(e)

lines = ['# Coverage Summary','']
if summary:
    data = summary.get('data', [{}])[0]
    totals = data.get('totals', {})
    lines.append('| Metric | Covered | Count | Percent |')
    lines.append('|---|---:|---:|---:|')
    for k in ('functions','lines','regions'):
        v = totals.get(k, {})
        lines.append(f"| {k} | {v.get('covered',0)} | {v.get('count',0)} | {v.get('percent',0):.2f}% |")
else:
    lines.append('Coverage data was not available in this environment.')
    if error:
      lines.append('')
      lines.append(f'> Last error: `{error}`')

out.write_text('\n'.join(lines) + '\n')
print(f'wrote {out}')
