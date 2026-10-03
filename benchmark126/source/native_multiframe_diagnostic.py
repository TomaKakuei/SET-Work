"""Opt-in diagnostic: deaggregate HPatches factors without changing costs.

The compiler closes over observation-only data. Reading those closures avoids
recompiling SIFT or changing initial correspondences. This is a versioned audit
adapter, not a replacement for registered production or historical interfaces.
"""
from dataclasses import replace
import inspect
import torch
from neural_grey_v2.hpatches_homography import _safe_project
from stage9_sampling import grid_sample


def cells(function):
    return inspect.getclosurevars(function).nonlocals


def coverage(spec, offset, view, like):
    y = torch.arange(spec.height, device=like.device)
    x = torch.arange(spec.width, device=like.device)
    weights = like.new_zeros(spec.height, spec.width)
    for rows, cols in spec.levels:
        region = (y[:, None] * rows // spec.height) * cols + x[None, :] * cols // spec.width
        weights += ((region + offset) % 2 == view).to(like.dtype) / len(spec.levels)
        offset += rows * cols
    return weights


def deaggregate(problem):
    """Return a cost-exact signed vector residual problem and its provenance."""
    scalar = cells(problem.cost_function)['function']
    graph = cells(scalar)
    required = {'photometric_functions', 'aggregate_photometric_levels', 'decode',
                'reference_data', 'cross_data', 'residual_scale_px', 'symmetric_reprojection'}
    if not required <= graph.keys():
        raise ValueError('Diagnostic requires the registered multi-frame compiler closure')
    photo = [cells(f) for f in graph['photometric_functions']]
    for p in photo:
        assert {'decode', 'source', 'sample_side', 'target_shape', 'spec', 'target_pyramid',
                'reference_observations', 'level_weights', 'valid_float'} <= p.keys()
    aggregate = graph['aggregate_photometric_levels']

    def raw(z, view):
        theta = z[None]
        output = []
        offset = 0
        for frame, p in enumerate(photo):
            projection = _safe_project(p['source'], p['decode'](theta[:, frame*8:(frame+1)*8]))
            grid = torch.stack((2*projection[..., 0]/(p['target_shape'][1]-1)-1,
                                2*projection[..., 1]/(p['target_shape'][0]-1)-1), -1)
            grid = grid.reshape(1, p['sample_side'], p['sample_side'], 2)
            for level, (image, observed, weight) in enumerate(zip(
                    p['target_pyramid'], p['reference_observations'], p['level_weights'])):
                mask = coverage(p['spec'], offset if aggregate else offset+level*p['spec'].factors, view, z)
                factor = (mask * p['valid_float'][0, 0].to(z) * float(weight)).sqrt()
                sampled = grid_sample(image.to(z), grid)
                output.append(((sampled-observed.to(z))[0, 0] * factor).reshape(-1))
            offset += p['spec'].factors * (1 if aggregate else len(p['level_weights']))
        homographies = graph['decode'](theta)

        def geometry(source, target, H, chunks):
            nonlocal offset
            for inverse in range(2 if graph['symmetric_reprojection'] else 1):
                a, b, matrix = (target, source, torch.linalg.inv(H)) if inverse else (source, target, H)
                error = (_safe_project(a.to(z), matrix)[0] - b.to(z)) / graph['residual_scale_px']
                # For s=||e||^2, .5||e*sqrt(2/(sqrt(1+s)+1))||^2 = sqrt(1+s)-1.
                # This expression also has finite, signed derivatives at e=0.
                signed = error * (2/(torch.sqrt(1+error.square().sum(-1, keepdim=True))+1)).sqrt()
                mask = z.new_zeros(len(error))
                for j, chunk in enumerate(chunks):
                    if (offset+j) % 2 == view:
                        mask[chunk.to(z.device)] = 1 / len(chunk)
                output.append((signed * mask.sqrt()[:, None]).reshape(-1))
                offset += len(chunks)

        for frame, source, target, chunks in graph['reference_data']:
            geometry(source, target, homographies[:, frame], chunks)
        for left, right, source, target, chunks in graph['cross_data']:
            H = homographies[:, right] @ torch.linalg.inv(homographies[:, left])
            geometry(source, target, H, chunks)
        if offset != problem.metadata['factor_count']:
            raise AssertionError('Factor layout changed')
        return torch.cat(output)[:, None]

    metadata = dict(problem.metadata)
    metadata.update(geometry='native signed photometric and per-correspondence cost-exact pseudo-Huber vectors v1',
                    residual_adapter_credit='shared project diagnostic; no network or external solver credit',
                    curvature_contract='GN of cost-exact signed vectors, not frozen-IRLS or objective Hessian',
                    sampling_semantics='same exact border bilinear AD branch as registered shared interface')
    adapted = replace(problem, raw_function=raw, weight_function=lambda z, v: torch.ones_like(raw(z, v)), metadata=metadata)
    assert not hasattr(problem, 'native'), 'This diagnostic does not replace a compiled native evaluator'
    return adapted, dict(compiler_source=scalar.__code__.co_filename, scalar_factor_count=metadata['factor_count'],
                         frames=len(photo), aggregate_photometric_levels=aggregate,
                         symmetric_reprojection=graph['symmetric_reprojection'], target_truth_read=False)
