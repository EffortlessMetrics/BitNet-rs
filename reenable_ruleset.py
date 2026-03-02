import subprocess, json, sys
r = json.loads(subprocess.check_output(['gh', 'api', 'repos/EffortlessMetrics/BitNet-rs/rulesets/9403749']))
r['enforcement'] = 'active'
p = subprocess.Popen(
    ['gh', 'api', 'repos/EffortlessMetrics/BitNet-rs/rulesets/9403749', '-X', 'PUT', '--input', '-'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
out, err = p.communicate(json.dumps(r).encode())
result = json.loads(out) if out else {}
print(f"enforcement: {result.get('enforcement', 'UNKNOWN')}")
if err:
    print(f"stderr: {err.decode()[:200]}", file=sys.stderr)
sys.exit(p.returncode)
