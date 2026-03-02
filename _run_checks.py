import subprocess, os, sys

env = os.environ.copy()
env['PATH'] = os.path.join(os.path.expanduser('~'), '.cargo', 'bin') + ';' + env['PATH']
cwd = r'C:\Code\Rust\BitNet-rs-wt1'

def run(cmd, label):
    print(f"=== {label} ===", flush=True)
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600, env=env)
    out = (r.stdout.decode('utf-8', errors='replace') + '\n' + r.stderr.decode('utf-8', errors='replace')).strip()
    lines = out.split('\n')
    for l in lines[-10:]:
        print(l, flush=True)
    print(f"EXIT: {r.returncode}", flush=True)
    return r.returncode

rc1 = run(['cargo', 'build', '-p', 'bitnet-tokenizers', '--no-default-features', '--features', 'cpu'], 'BUILD')
rc2 = run(['cargo', 'test', '-p', 'bitnet-tokenizers', '--no-default-features', '--features', 'cpu', '--', 'vocab_analyzer'], 'TEST')
rc3 = run(['cargo', 'fmt', '--all'], 'FMT')
print(f"\nSUMMARY: build={rc1} test={rc2} fmt={rc3}")
