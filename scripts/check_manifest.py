"""Verify the compact release's declared file hashes and sizes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = ROOT / row['path']
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise SystemExit(f"Release file mismatch: {row['path']}")
    print(json.dumps({'status': 'passed', 'files': len(manifest['files']),
                      'bytes': sum(row['bytes'] for row in manifest['files'])}))


if __name__ == '__main__':
    main()
