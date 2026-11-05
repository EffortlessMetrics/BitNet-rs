"""Route the blocking offline tokenizer lane using the policy's risk paths."""

import argparse
import fnmatch
import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SELF_PATHS = {
    '.github/workflows/compatibility.yml',
    'policy/ci-risk-packs.toml',
    'scripts/ci_tokenizer_plan.py',
    'scripts/test_ci_tokenizer_plan.py',
}


def tokenizer_required(event, paths, labels=(), root=ROOT):
    if event != 'pull_request' or not paths or {'tokenizer', 'full-ci'} & set(labels):
        return True
    with (root / 'policy/ci-risk-packs.toml').open('rb') as source:
        patterns = tomllib.load(source)['risk_pack']['tokenizer']['paths']
    if not patterns or not all(isinstance(pattern, str) for pattern in patterns):
        raise ValueError('tokenizer risk paths must be nonempty strings')
    return any(path in SELF_PATHS or any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
               for path in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', required=True)
    parser.add_argument('--labels-json', default='[]')
    parser.add_argument('--changed-files', default='')
    args = parser.parse_args()
    labels = json.loads(args.labels_json or '[]')
    if labels is None:
        labels = []  # Push events have no pull_request.labels.
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        parser.error('labels must be a JSON string array')
    paths = args.changed_files.splitlines()
    print(str(tokenizer_required(args.event, paths, labels)).lower())


if __name__ == '__main__':
    main()
