"""Portable input compilation, frozen-signature checks, solving, and scoring."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for folder in (ROOT / 'code', HERE / 'source'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from compact_runtime import load_model
from stage7_tasks import compile_case as compile_synthetic
from setsunet_csn.problems_stage2 import known_blur_problem, scalar_factor_problem, hpatches_multiframe_problem
from setsunet_csn.stage5 import Stage5Config, solve as original_solve
from setsunet_csn.stage5_curve_ablation import solve as curve_solve
from stage9_backends import engine_for, CountedEngine

_INSTALLED = False


def install_runtime():
    global _INSTALLED
    if not _INSTALLED:
        from stereo_colored_hessian import install
        install()
        _INSTALLED = True


def registry():
    return json.loads((HERE / 'cases.json').read_text(encoding='utf-8'))['cases']


def _rgb_tensor(array):
    return torch.from_numpy(np.asarray(array).copy()).permute(2, 0, 1).float() / 255.


def _psnr(estimate, target):
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = np.mean((estimate-target)**2)
    return float(-10*np.log10(max(mse, 1e-12)))


def _image_from_theta(theta):
    return np.clip(np.asarray(theta, dtype=np.float32)[:768].reshape(3, 16, 16).transpose(1, 2, 0), 0, 1)


def _known_blur(tile, *, original):
    if original:
        target = _rgb_tensor(tile).mean(0).double()
    else:
        target = torch.from_numpy(np.asarray(tile, dtype=np.float32).copy()).permute(2, 0, 1).mean(0).double()/255.
    axis = torch.arange(-2, 3, dtype=torch.float64)
    kernel = torch.exp(-.5*(axis[:, None]**2+axis[None, :]**2)/1.2**2)
    kernel /= kernel.sum()
    observation = F.conv2d(F.pad(target[None, None], (2, 2, 2, 2), mode='reflect'), kernel[None, None])[0, 0]
    return known_blur_problem(observation, kernel)


def _image(spec):
    from neural_grey_v2.blind_low_light_m27 import BlindLowLightFactorGraph, BlindLowLightConfig
    from neural_grey_v2.blind_noise_m26 import BlindNoiseFactorGraph, BlindNoiseConfig
    record = spec['record']
    with np.load(HERE / record['asset'], allow_pickle=False) as archive:
        observation = archive['observation'].copy()
        target = archive['target'].copy()
    family = record['family']
    original = spec['kind'] == 'registry'
    if family in ('known_blur', 'known_blur_region'):
        problem = _known_blur(observation, original=original)
        if original:
            truth = _rgb_tensor(target).mean(0)
            def evaluate(theta):
                x = torch.as_tensor(theta, dtype=problem.initial.dtype).reshape(16, 16).clamp(0, 1)
                error = (x-truth).square().mean().clamp_min(1e-12)
                return {'metric': float(-10*torch.log10(error))}
        else:
            truth = target.astype(np.float64).mean(2)/255.
            def evaluate(theta):
                return {'metric': _psnr(np.clip(np.asarray(theta).reshape(16, 16), 0, 1), truth)}
        return problem, evaluate, {'higher_better': True}
    if family in ('lowlight', 'lowlight_region'):
        image = (torch.from_numpy(observation.copy())[None] if original else
                 torch.from_numpy(observation.astype(np.float32, copy=True)).permute(2, 0, 1)[None]/255.)
        graph = BlindLowLightFactorGraph(image, BlindLowLightConfig(illumination_grid=(4, 4)))
        problem = scalar_factor_problem(graph, family='lowlight')
    elif family in ('noise', 'sidd_region'):
        image = torch.from_numpy(np.array(observation, dtype=np.float32, copy=True)).permute(2, 0, 1)[None]/255.
        graph = BlindNoiseFactorGraph(image, BlindNoiseConfig())
        problem = scalar_factor_problem(graph, family='sidd_noise')
    else:
        raise ValueError(family)
    if original:
        truth = (torch.from_numpy(target.copy()) if family == 'lowlight'
                 else _rgb_tensor(target))
        def evaluate(theta):
            decoded = graph.decode(torch.as_tensor(theta, dtype=problem.initial.dtype)[None])[0][0]
            error = (decoded.clamp(0, 1)-truth).square().mean().clamp_min(1e-12)
            return {'metric': float(-10*torch.log10(error))}
    else:
        truth = target.astype(np.float64)/255.
        def evaluate(theta):
            return {'metric': _psnr(_image_from_theta(theta), truth)}
    return problem, evaluate, {'higher_better': True}


def _multiframe(spec):
    from neural_grey_v2.hpatches_multiframe_graph import compile_hpatches_multiframe_graph
    from neural_grey_v2.hpatches_homography import read_homography, homography_corner_error
    record = spec['record']
    root = HERE / 'data/hpatches' / record['sequence']
    targets = tuple(record['targets'])
    graph = compile_hpatches_multiframe_graph(root, targets, device='cpu', dtype=torch.float64,
                                               support_mode='hybrid', symmetric_reprojection=True)
    problem = hpatches_multiframe_problem(graph)
    truth = [read_homography(root / f'H_1_{index}') for index in targets]
    def evaluate(theta):
        estimates = graph.decode(torch.as_tensor(theta, dtype=problem.initial.dtype)[None])[0].detach().numpy()
        values = [homography_corner_error(np.asarray(estimate), gt, graph.reference_shape)
                  for estimate, gt in zip(estimates, truth)]
        return {'metric': float(np.mean(values))}
    return problem, evaluate, {'higher_better': False}


def _stereo(spec):
    from neural_grey_v2.middlebury_stereo import (compile_middlebury_scene,
        stereo_scene_factor_graph, load_middlebury_query)
    record = spec['record']
    observation = compile_middlebury_scene(HERE / 'data/stereo' / record['scene'], device='cpu')
    if spec['kind'] == 'scale_stereo':
        index = record['crop']
    else:
        index = next((i for i, valid in enumerate(observation.valid_crop) if valid), 0)
    one = replace(observation, crop_origins=(observation.crop_origins[index],),
                  valid_crop=(observation.valid_crop[index],),
                  left_full=observation.left_full.double(),
                  right_full=observation.right_full.double(),
                  initial_disparity_full=observation.initial_disparity_full.double())
    function, initial, ids, decode = stereo_scene_factor_graph(one)
    problem = scalar_factor_problem(one, family='middlebury_stereo',
                                    factor_function=function, initial=initial[0].double(), block_id=ids)
    def evaluate(theta):
        gt, mask = load_middlebury_query(one)
        estimate = decode(torch.as_tensor(theta, dtype=torch.float64)[None])[0, 0]
        valid = mask[0, 0] & torch.isfinite(estimate)
        if not valid.any():
            raise ValueError('No scored pixels in registered crop')
        return {'metric': float((estimate-gt[0, 0]).abs()[valid].mean())}
    return problem, evaluate, {'higher_better': False}


@contextmanager
def load_case(spec):
    if spec['kind'] == 'synthetic':
        problem, evaluate = compile_synthetic(spec['spec'])
        yield problem, evaluate, {'higher_better': False}
        return
    if spec['record']['family'] == 'multiframe':
        problem, evaluate, meta = _multiframe(spec)
    elif spec['record']['family'] == 'stereo':
        problem, evaluate, meta = _stereo(spec)
    else:
        problem, evaluate, meta = _image(spec)
    from stage9_sampling import prepare as sampling_prepare
    from stage9_hpatches import prepare as hpatches_prepare
    from shared_initialization_contract import prepared_problem
    problem = hpatches_prepare(sampling_prepare(problem))
    with prepared_problem(problem) as (problem, preparation):
        yield problem, evaluate, dict(meta, preparation=preparation)


def signature(problem):
    engine = engine_for(problem)
    x = problem.initial.detach().numpy()
    residual, jacobian = engine.least_squares(x, True)
    if hasattr(jacobian, 'toarray'):
        jacobian = jacobian.toarray()
    digest = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    cost = float(engine.objective(x))
    residual_cost = .5*float(residual@residual)
    return dict(n=len(x), m=len(residual), initial_sha=digest(x),
                r_sha=digest(residual), J_sha=digest(jacobian),
                dtype=str(problem.initial.dtype), native=hasattr(problem, 'native'),
                objective=cost, residual_objective=residual_cost,
                contract_relative=abs(cost-residual_cost)/max(abs(cost), 1e-30))


def checked_input(spec, problem):
    expected = json.loads((HERE / 'input_audit.json').read_text(encoding='utf-8'))[spec['key']]
    actual = signature(problem)
    # SIFT and least-squares initialization may move in their last bits across
    # OpenCV/PyTorch builds. Keep that visible for multiframe inputs while
    # requiring the same dimensions, objective and source assets below.
    exact = ('n', 'm', 'dtype', 'native')
    if any(actual[name] != expected[name] for name in exact):
        raise AssertionError({'key': spec['key'], 'check': 'identity',
                              'expected': expected, 'actual': actual})
    if (actual['initial_sha'] != expected['initial_sha'] and
            spec['record']['family'] != 'multiframe'):
        raise AssertionError({'key': spec['key'], 'check': 'initial_sha',
                              'expected': expected['initial_sha'],
                              'actual': actual['initial_sha']})
    for name in ('objective', 'residual_objective'):
        a, b = actual[name], expected[name]
        if abs(a-b) > 1e-8 + 1e-6*max(abs(a), abs(b)):
            raise AssertionError({'key': spec['key'], 'check': name,
                                  'expected': b, 'actual': a})
    if actual['contract_relative'] > 1e-5:
        raise AssertionError({'key': spec['key'], 'check': 'residual_contract',
                              'actual': actual['contract_relative']})
    return dict(actual, reference_initial_sha_exact=actual['initial_sha'] == expected['initial_sha'],
                reference_residual_sha_exact=actual['r_sha'] == expected['r_sha'],
                reference_jacobian_sha_exact=actual['J_sha'] == expected['J_sha'])


def config(group):
    return Stage5Config(steps=5, measured_directions=8, completion_directions=8,
                        damping_mode='relative', damping=.001, backend='c',
                        residual_cg_iterations=2 if group.startswith('view_graph_')
                        or group in ('broyden', 'tum') else 0,
                        snapshot_steps=(1, 2, 3, 4, 5))


def solve_quad_compatible(problem, network, cfg):
    """Historical observation proxy for AD problems, without changing them."""
    from finite_models_20260925 import backbone, search_completion
    native = engine_for(problem)
    original = native.least_squares
    cache, packet, failures = {}, {}, []
    query = dict(J=0, F=0, hits=0)

    def cached(x, jacobian=True):
        key = np.asarray(x, dtype=np.float64).tobytes()
        if key in cache and (not jacobian or cache[key][1] is not None):
            query['hits'] += 1
            return cache[key] if jacobian else (cache[key][0], None)
        query['J' if jacobian else 'F'] += 1
        value = original(x, jacobian)
        cache[key] = value
        return value

    native.least_squares = cached
    proxy = SimpleNamespace(native=native)
    audit = SimpleNamespace(phase='solver')
    hook = search_completion.polished_hook(proxy, audit, packet)

    def guarded(*args):
        try:
            return hook(*args)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append(dict(kind=type(exc).__name__, message=str(exc),
                                 iteration=int(args[0])+1))
            packet.clear()
            return args[2], args[4], False
    try:
        result = backbone.solve(problem, network, CountedEngine(native),
                                curve_mode='norm_mean_envelope', config=cfg, hook=guarded)
    finally:
        native.least_squares = original
    result.update(refinement_failures=failures, LS_requests=query)
    return result, packet


def solve_case(spec, method, network=None):
    from finite_adaptation_20260926.release import solve as linear_solve
    from finite_repair_20260927.solver import solve as repair_solve
    if method not in ('original', 'quad', 'linear_fast', 'repair'):
        raise ValueError(method)
    torch.set_num_threads(1)
    threadpool_limits(1)
    install_runtime()
    network = network or load_model()
    with load_case(spec) as (problem, evaluate, meta):
        checked_input(spec, problem)
        cfg = config(spec['group'])
        if method == 'original':
            result = original_solve(problem, network, 'csn', cfg)
        elif method == 'quad':
            result, _ = solve_quad_compatible(problem, network, cfg)
        elif method == 'linear_fast':
            result, _ = linear_solve(problem, network, cfg)
        else:
            result, _ = repair_solve(problem, network, cfg)
        theta = np.asarray(result['theta'], dtype=np.float64)
        if not np.isfinite(theta).all() or len(result['trajectory']) != 5:
            raise AssertionError('Invalid five-step endpoint')
        return {'key': spec['key'], 'task': spec['task'], 'group': spec['group'],
                'method': method, 'steps': 5, 'status': 'ok', 'theta': theta.tolist(),
                'final_cost': np.asarray(result['final_cost']).tolist(),
                'actual_joint_J': result.get('actual_joint_J')}


def score_endpoint(spec, theta):
    with load_case(spec) as (problem, evaluate, meta):
        checked_input(spec, problem)
        x = torch.as_tensor(theta, dtype=problem.initial.dtype)
        return {'metric': float(evaluate(x)['metric']),
                'objective': float(engine_for(problem).objective(x.detach().numpy())),
                'higher_better': bool(meta['higher_better'])}
