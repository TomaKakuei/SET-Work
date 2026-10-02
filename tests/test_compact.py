"""Small numerical checks: no dataset downloads, training, or long trajectories."""
from pathlib import Path
import json
import os
import sys
import unittest
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tests')]
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from compact_runtime import cases, config, load_model, run_case
from extra_problem import problem_for
from stage9_backends import CountedEngine, engine_for
from setsunet_csn import stage5_curve_ablation
from setsunet_csn.core import minimax
from setsunet_csn.native_stage4 import minimax_c, orthogonalize_c
from setsunet_csn.stage5 import orthogonalize
import core_checks


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.threads = threadpool_limits(limits=1)
        cls.model = load_model()

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_shared_checkpoint(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 49096)
        self.assertFalse(self.model.training)

    def test_core_properties(self):
        report = core_checks.run()
        self.assertEqual(report['random_spd_cases'], 60)

    def test_native_minimax_and_rank(self):
        generator = torch.Generator().manual_seed(970200)
        for n in [1, 2, 8, 16]:
            for scale in [.001, 1., 1000.]:
                matrices = []
                for _ in range(2):
                    matrix = torch.randn(n, n, generator=generator, dtype=torch.float64)
                    matrices.append(scale*(matrix.T@matrix+.1*torch.eye(n, dtype=torch.float64)))
                ga, gb = scale*torch.randn(2, n, generator=generator, dtype=torch.float64)
                reference = minimax(*matrices, ga, gb)
                actual = minimax_c(*matrices, ga, gb)
                torch.testing.assert_close(actual[0], reference[0], rtol=1e-8, atol=1e-9)
                self.assertLess(abs(actual[1]-reference[1]), 1e-8)
        matrix = torch.randn(31, 16, generator=generator, dtype=torch.float64)
        matrix[:, 3] = matrix[:, 0] + 1e-13*matrix[:, 1]
        expected = orthogonalize(matrix, backend='torch')
        actual = orthogonalize_c(matrix, 1e-9)
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual@actual.T, expected@expected.T, atol=1e-9, rtol=1e-9)

    def test_four_cases_five_steps(self):
        for spec in cases():
            for method in ['csn', 'lm', 'pcg']:
                with self.subTest(spec=spec, method=method):
                    result, score = run_case(spec, self.model, method)
                    self.assertEqual(result['iterations'], 5)
                    self.assertTrue(torch.isfinite(result['theta']).all())
                    self.assertTrue(np.isfinite(score['metric']))
                    for state in result['trajectory']:
                        before = np.mean(state['cost_before'])
                        self.assertLessEqual(np.mean(state['cost_after']), before+1e-12*max(1, abs(before)))

    def test_frozen_curve_endpoints(self):
        fixture = json.loads((ROOT / 'tests/fixtures/curve_reference.json').read_text(encoding='utf-8'))
        expected = {row['key']: row for row in fixture['predictions']}
        for spec in fixture['cases']:
            with self.subTest(key=spec['key']):
                problem, evaluate = problem_for(spec)
                result = stage5_curve_ablation.solve(problem, self.model, CountedEngine(engine_for(problem)),
                    tangent='csn', curve_mode='norm_mean_envelope', config=config())
                theta = result['theta'].detach().numpy()
                reference = expected[spec['key']]
                error = np.linalg.norm(theta-np.asarray(reference['theta']))/max(np.linalg.norm(reference['theta']), 1e-20)
                self.assertEqual(result['iterations'], 5)
                self.assertLess(error, 1e-7)
                metric = evaluate(theta)
                self.assertLessEqual(abs(metric-reference['metric']), 1e-10+1e-6*max(abs(metric), abs(reference['metric'])))


if __name__ == '__main__':
    unittest.main()
