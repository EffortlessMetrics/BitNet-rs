#!/usr/bin/env python3
import collections, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / 'docs/churn.md'

def lines(cmd):
    p = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=True)
    return [l for l in p.stdout.splitlines() if l.strip()]

files = lines(['git','log','--since=90.days','--name-only','--pretty=format:'])
file_counts = collections.Counter(files)
dir_counts = collections.Counter()
for f,c in file_counts.items():
    top = f.split('/',1)[0]
    dir_counts[top] += c

report = ['# Churn Report (90 days)','', 'Generated from `git log --since=90.days`.', '', '## Top directories']
for d,c in dir_counts.most_common(15):
    report.append(f'- `{d}`: {c} touches')
report += ['', '## Most touched files']
for f,c in file_counts.most_common(20):
    report.append(f'- `{f}`: {c} touches')

out.write_text('\n'.join(report)+'\n')
print(f'wrote {out}')
