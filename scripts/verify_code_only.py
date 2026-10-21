#!/usr/bin/env python3
"""Reject data, credentials and binary/large files from the actual Git index.

Run after git add and before committing/pushing. This is a supplemental guard:
it cannot prove that arbitrary prose or source literals contain no private data.
Only file paths, line numbers and risk labels are printed for sensitive hits.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

MAX_BYTES = 2 * 1024 * 1024
BLOCKED_PARTS = {
    'data', 'work', 'models', 'model', 'outputs', 'output', 'prepared',
    'evaluations', 'evaluation_artifacts', 'logs', 'wandb', 'checkpoints',
    'checkpoint', '.cache', '.venv', '.venv-grpo', '__pycache__', 'node_modules',
    '.ssh', '.aws', '.config', 'secrets', 'credentials',
}
BLOCKED_NAMES = {'.env', '.netrc', '.git-credentials', 'credentials.json',
                 'id_rsa', 'id_ed25519', 'id_ecdsa', '.ds_store'}
ROOT_NAMES = {'.gitignore', 'README.md', 'RAG_GUIDE.md', 'SFT_GUIDE.md',
              'LICENSE', 'LICENSE.txt', 'NOTICE'}
KNOWN_DUMMIES = {'private-test-value', 'test-api-key', 'test-wandb-api-key',
                 'your-api-key', 'YOUR_API_KEY', 'placeholder', 'test-password'}
SECRET_PATTERNS = [
    ('private-key', re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')),
    ('github-token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b')),
    ('api-token', re.compile(r'\bsk-[A-Za-z0-9_-]{20,}\b')),
    ('aws-access-key', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
]
SECRET_LITERAL = re.compile(
    r'''(?i)["']?(?:[A-Za-z_][A-Za-z0-9_]*(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)|api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)["']?\s*[:=]\s*["']([^"'\n]{12,})["']''')


def issue(path, risk, line=None):
    record = {'path': path, 'risk': risk}
    if line is not None:
        record['line'] = line
    return record


def allowed(path):
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or any(part.lower() in BLOCKED_PARTS for part in p.parts):
        return False
    if any(part.lower() in BLOCKED_NAMES or part.lower().startswith('.env.') for part in p.parts):
        return False
    if len(p.parts) == 1:
        return path in ROOT_NAMES or fnmatch.fnmatchcase(path, 'requirements*.txt')
    if p.parts[0] == 'scripts':
        return len(p.parts) == 2 and p.suffix in {'.py', '.sh'}
    if p.parts[0] == 'tests':
        if p.suffix == '.py':
            return True
        prefix = ('tests', 'fixtures', 'official_qa_metrics')
        return p.parts[:-1] == prefix and (
            p.name in {'NOTICE.md', 'sources.json'} or p.name.endswith('.LICENSE.txt'))
    if p.parts[0] == 'configs':
        return len(p.parts) == 2 and p.suffix in {'.json', '.yaml', '.yml'}
    if p.parts[0] == 'docs':
        return len(p.parts) == 2 and p.suffix == '.md'
    return False


def inspect(path, payload, mode='100644'):
    problems = []
    if not allowed(path):
        problems.append(issue(path, 'outside-code-allowlist'))
    if mode not in {'100644', '100755'}:
        problems.append(issue(path, 'symlink-submodule-or-nonregular-file'))
    if len(payload) > MAX_BYTES:
        problems.append(issue(path, 'file-exceeds-2-MiB'))
    if b'\x00' in payload:
        problems.append(issue(path, 'binary-content'))
        return problems
    try:
        content = payload.decode('utf-8')
    except UnicodeDecodeError:
        problems.append(issue(path, 'non-UTF8-content'))
        return problems
    # Accept tab/newline/carriage-return, reject binary control characters.
    if any(ord(ch) < 32 and ch not in '\t\n\r' for ch in content):
        problems.append(issue(path, 'binary-control-character'))
    for number, line in enumerate(content.splitlines(), 1):
        for risk, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                problems.append(issue(path, risk, number))
        for match in SECRET_LITERAL.finditer(line):
            if match.group(1) not in KNOWN_DUMMIES:
                problems.append(issue(path, 'secret-literal-review-required', number))
    return problems


def git_index(root):
    raw = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--stage', '-z'])
    for record in raw.split(b'\0'):
        if not record:
            continue
        meta, name = record.split(b'\t', 1)
        mode, oid, stage = meta.decode('ascii').split()
        path = name.decode('utf-8', errors='strict')
        if stage != '0':
            yield path, b'', 'unmerged'
            continue
        payload = subprocess.check_output(['git', '-C', str(root), 'cat-file', 'blob', oid])
        yield path, payload, mode


def filesystem(root):
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if '.git' in relative.parts:
            continue
        if path.is_symlink():
            yield relative.as_posix(), b'', '120000'
        elif path.is_file():
            yield relative.as_posix(), path.read_bytes(), '100644'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--filesystem', action='store_true', help='Audit all files before Git initialization; default checks staged blobs')
    args = parser.parse_args()
    root = args.root.resolve()
    issues = []
    count = 0
    try:
        entries = filesystem(root) if args.filesystem else git_index(root)
        for path, payload, mode in entries:
            count += 1
            issues.extend(inspect(path, payload, mode))
    except (OSError, UnicodeError, subprocess.CalledProcessError, ValueError):
        print(json.dumps({'risk': 'unable-to-read-repository-index-or-tree'}))
        return 2
    for problem in issues:
        print(json.dumps(problem, ensure_ascii=False, sort_keys=True))
    print(json.dumps({'checked_files': count, 'issues': len(issues), 'mode': 'filesystem' if args.filesystem else 'git-index'}))
    return 1 if issues or count == 0 else 0


if __name__ == '__main__':
    raise SystemExit(main())
