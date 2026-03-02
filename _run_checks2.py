import subprocess, os

env = os.environ.copy()
env['PATH'] = os.path.join(os.path.expanduser('~'), '.cargo', 'bin') + ';' + env['PATH']
cwd = r'C:\Code\Rust\BitNet-rs-wt1'
out_file = os.path.join(cwd, '_check_output.txt')

with open(out_file, 'w') as f:
    def run(cmd, label):
        f.write("=== {} ===\n".format(label))
        f.flush()
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600, env=env)
        out = (r.stdout.decode('utf-8', errors='replace') + '\n' + r.stderr.decode('utf-8', errors='replace')).strip()
        lines = out.split('\n')
        for l in lines[-15:]:
            f.write(l + '\n')
        f.write("EXIT: {}\n\n".format(r.returncode))
        f.flush()
        return r.returncode

    rc1 = run(['cargo', 'build', '-p', 'bitnet-tokenizers', '--no-default-features', '--features', 'cpu'], 'BUILD')
    rc2 = run(['cargo', 'test', '-p', 'bitnet-tokenizers', '--no-default-features', '--features', 'cpu', '--', 'vocab_analyzer'], 'TEST')
    rc3 = run(['cargo', 'fmt', '--all'], 'FMT')
    f.write("\nSUMMARY: build={} test={} fmt={}\n".format(rc1, rc2, rc3))
