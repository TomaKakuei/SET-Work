"""Run or independently score the registered five-step 126-case inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from .runtime import HERE, checked_input, load_case, registry, score_endpoint, solve_case


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                       encoding='utf-8')
    pending.replace(path)


def verify_data():
    manifest = json.loads((HERE / 'DATA_MANIFEST.json').read_text(encoding='utf-8'))
    for relative, expected in manifest.items():
        path = HERE / relative
        if path.stat().st_size != expected['bytes']:
            raise AssertionError(('size', relative))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            raise AssertionError(('hash', relative))
    print(json.dumps({'status': 'passed', 'data_files': len(manifest)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list', 'verify-data', 'audit-inputs', 'run', 'score'))
    parser.add_argument('--key', help='One registered case key; omit to run all 126')
    parser.add_argument('--method', choices=('original', 'quad', 'linear_fast', 'repair'))
    parser.add_argument('--output', type=Path, default=Path('results/benchmark126'))
    args = parser.parse_args()
    cases = registry()
    selected = [c for c in cases if args.key is None or c['key'] == args.key]
    if not selected:
        parser.error('Unknown case key')
    if args.command == 'list':
        for case in selected:
            print(json.dumps({k: case[k] for k in ('key', 'task', 'slot', 'group', 'kind')}))
        return
    if args.command == 'verify-data':
        verify_data()
        return
    if args.command in ('run', 'score') and not args.method:
        parser.error('--method is required for run and score')
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(1)
    import cv2
    cv2.setNumThreads(1)
    if args.command in ('audit-inputs', 'run'):
        from .runtime import install_runtime
        install_runtime()
    for index, case in enumerate(selected, 1):
        key = case['key']
        if args.command == 'audit-inputs':
            destination = args.output / 'input_audit' / (key + '.json')
            if destination.exists():
                continue
            with load_case(case) as (problem, _, _):
                value = checked_input(case, problem)
            write(destination, {'key': key, 'status': 'passed', 'signature': value})
            print(json.dumps({'case': index, 'total': len(selected), 'key': key,
                              'status': 'passed'}), flush=True)
            continue
        destination = args.output / ('predictions' if args.command == 'run' else 'scores') / args.method / (key + '.json')
        if destination.exists():
            continue
        try:
            if args.command == 'run':
                value = solve_case(case, args.method)
            else:
                prediction = args.output / 'predictions' / args.method / (key + '.json')
                frozen = json.loads(prediction.read_text(encoding='utf-8'))
                if frozen['status'] != 'ok':
                    value = {'key': key, 'method': args.method, 'status': 'not_scored',
                             'reason': frozen['status']}
                else:
                    value = {'key': key, 'method': args.method, 'status': 'ok',
                             **score_endpoint(case, frozen['theta'])}
        except Exception:
            value = {'key': key, 'method': args.method, 'status': 'failed',
                     'error': traceback.format_exc()}
        write(destination, value)
        print(json.dumps({'case': index, 'total': len(selected), 'key': key,
                          'method': args.method, 'status': value['status']}), flush=True)
    if args.command == 'audit-inputs':
        files = list((args.output / 'input_audit').glob('*.json'))
        if len(files) == 126:
            write(args.output / 'input_audit_complete.json',
                  {'status': 'passed', 'cases': 126})


if __name__ == '__main__':
    main()
