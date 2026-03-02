import subprocess, os

env = os.environ.copy()
env['PATH'] = os.path.join(os.path.expanduser('~'), '.cargo', 'bin') + os.pathsep + env['PATH']
cwd = r'C:\Code\Rust\BitNet-rs-wt1'
out = os.path.join(cwd, '_results.txt')
done = os.path.join(cwd, '_done.flag')

# Remove old files
for f in [out, done]:
    if os.path.exists(f):
        os.remove(f)

fh = open(out, 'w')

def run(cmd, label):
    fh.write("=== %s ===\n" % label)
    fh.flush()
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600, env=env)
    text = r.stdout.decode('utf-8', errors='replace')
    lines = text.strip().split('\n')
    for l in lines[-20:]:
        fh.write(l + '\n')
    fh.write("EXIT: %d\n\n" % r.returncode)
    fh.flush()
    return r.returncode

rc1 = run(['cargo', 'build', '-p', 'bitnet-tokenizers', '--no-default-features', '--features', 'cpu'], 'BUILD')
rc2 = run(['cargo', 'test', '-p', 'bitnet-tokenizers', '--no-default-features', '--features', 'cpu', '--', 'vocab_analyzer'], 'TEST')
rc3 = run(['cargo', 'fmt', '--all'], 'FMT')
fh.write("SUMMARY: build=%d test=%d fmt=%d\n" % (rc1, rc2, rc3))
fh.close()

# Signal completion
open(done, 'w').write('done')
