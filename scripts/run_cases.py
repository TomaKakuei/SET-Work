"""Run four generated cases with the shared model and two numerical controls."""
from pathlib import Path
import argparse
import json
import os
import sys

for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import torch
from threadpoolctl import threadpool_limits
from compact_runtime import cases, load_model, run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/compact.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        model = load_model()
        rows = []
        for spec in cases():
            for method in ['csn', 'lm', 'pcg']:
                result, score = run_case(spec, model, method)
                row = dict(case=spec, method=method, steps=result['iterations'],
                           metric=score['metric'], initial_objective=sum(result['initial_cost'])/2,
                           final_objective=sum(result['final_cost'])/2,
                           theta=result['theta'].tolist())
                rows.append(row)
                print(f"{spec.get('condition', spec['family']):12s} {method:3s} "
                      f"steps={row['steps']} metric={row['metric']:.8g}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(scope='Synthetic software examples; each case is scored separately',
        steps=5, checkpoint_count=1, rows=rows), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

