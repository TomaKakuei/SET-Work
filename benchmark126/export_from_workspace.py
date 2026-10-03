"""Materialize the registered 126-case inputs from the original workspace.

This one-time exporter is provenance tooling. Users of the published package
run ``python -m benchmark126`` and do not need the original workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_path(root, value):
    """Resolve a historical path without retaining its machine-specific prefix."""
    normalized = str(value).replace('\\', '/')
    token = 'SETSUNET_CSN_CONTINUE/'
    if token in normalized:
        return root / normalized.split(token, 1)[1]
    if normalized.startswith('../datasets/'):
        return root / normalized[3:]
    if normalized.startswith('datasets/'):
        return root / normalized
    return Path(normalized)


def rgb(path):
    encoded = np.fromfile(path, dtype=np.uint8)
    result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if result is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def scale_origin(key, shape, side):
    digest = hashlib.sha256(('registered_20260919/' + key).encode()).digest()
    return (int.from_bytes(digest[:8], 'little') % (shape[0] - side + 1),
            int.from_bytes(digest[8:16], 'little') % (shape[1] - side + 1))


def stage2_origin(case_id, shape, side=16):
    digest = hashlib.sha256(('20260910/' + case_id).encode()).digest()
    return (int.from_bytes(digest[:8], 'big') % (shape[0] - side + 1),
            int.from_bytes(digest[8:16], 'big') % (shape[1] - side + 1))


def image_pair(root, spec):
    family = spec['family']
    if family == 'sidd_region':
        base = root / 'datasets/sidd_validation'
        noisy = np.load(base / 'ValidationNoisyBlocksSrgb.npy', mmap_mode='r')[spec['image'], spec['block']]
        target = np.load(base / 'ValidationGtBlocksSrgb.npy', mmap_mode='r')[spec['image'], spec['block']]
    else:
        source = spec.get('input_path', spec.get('target_path'))
        noisy = rgb(original_path(root, source))
        target = rgb(original_path(root, spec['target_path']))
    side = spec['side']
    top, left = scale_origin(spec['key'], noisy.shape, side)
    observation = np.array(noisy[top:top+16, left:left+16], copy=True)
    truth = np.array(target[top:top+16, left:left+16], copy=True)
    return observation, truth, (top, left)


def registry_pair(root, spec):
    family = spec['family']
    details = spec['spec']
    if family == 'known_blur':
        source = rgb(original_path(root, details['target_path']))
        top, left = stage2_origin(details['case_id'], source.shape)
        tile = np.array(source[top:top+16, left:left+16], copy=True)
        return tile, tile, (top, left)
    if family == 'lowlight':
        def resized(path):
            image = torch.from_numpy(rgb(original_path(root, path))).permute(2, 0, 1).float() / 255.
            return F.interpolate(image[None], size=(16, 16), mode='area')[0].numpy()
        return resized(details['input_path']), resized(details['target_path']), None
    if family == 'noise':
        archive = root / 'datasets/sidd/sidd_stage2_exact4.zip'
        with zipfile.ZipFile(archive) as handle:
            def decoded(name):
                b = np.frombuffer(handle.read(name), dtype=np.uint8)
                return cv2.cvtColor(cv2.imdecode(b, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            source = decoded(details['noisy_member'])
            target = decoded(details['target_member'])
        top, left = stage2_origin(details['case_id'], source.shape)
        return (np.array(source[top:top+16, left:left+16], copy=True),
                np.array(target[top:top+16, left:left+16], copy=True), (top, left))
    raise ValueError(f'Unrecognized registry image family: {family}')


def export(root, destination=HERE):
    root = Path(root).resolve()
    protocol_file = root / 'workspace/results_stage9_dev/runs/paper21x6_two_models_20260925/protocol.json'
    protocol = json.loads(protocol_file.read_text(encoding='utf-8'))
    assert len(protocol['cases']) == 126
    data = destination / 'data'
    data.mkdir(parents=True, exist_ok=True)
    cases = []
    for item in protocol['cases']:
        case = {k: v for k, v in item.items() if k != 'record'}
        record = item.get('record', {})
        kind = item['kind']
        if kind == 'synthetic':
            pass
        elif kind in ('multiframe', 'registry') and record.get('family') == 'multiframe':
            sequence = record['sequence']
            source = root / 'datasets/hpatches/hpatches-sequences-release' / sequence
            target = data / 'hpatches' / sequence
            target.mkdir(parents=True, exist_ok=True)
            for name in ['1.ppm', *(f'{i}.ppm' for i in record['targets']),
                         *(f'H_1_{i}' for i in record['targets'])]:
                if not (target / name).exists():
                    shutil.copyfile(source / name, target / name)
            case['record'] = {k: record[k] for k in ('family', 'sequence', 'targets')}
        elif kind == 'scale_stereo' or (kind == 'registry' and record.get('family') == 'stereo'):
            scene = (record['scene_root'] if kind == 'scale_stereo'
                     else record['spec']['scene_root'])
            source = original_path(root, scene)
            target = data / 'stereo' / source.name
            target.mkdir(parents=True, exist_ok=True)
            for name in ('calib.txt', 'im0.png', 'im1.png', 'disp0GT.pfm', 'mask0nocc.png'):
                if not (target / name).exists():
                    shutil.copyfile(source / name, target / name)
            case['record'] = {'family': 'stereo', 'scene': source.name,
                              'crop': record['crop'] if kind == 'scale_stereo' else None}
        elif kind == 'scale_tile':
            observation, truth, origin = image_pair(root, record)
            path = data / 'tiles' / (item['key'] + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, observation=observation, target=truth)
            case['record'] = {'family': record['family'], 'origin': origin,
                              'asset': str(path.relative_to(destination)).replace('\\', '/')}
        elif kind == 'registry':
            observation, truth, origin = registry_pair(root, record)
            path = data / 'tiles' / (item['key'] + '.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, observation=observation, target=truth)
            case['record'] = {'family': record['family'], 'origin': origin,
                              'asset': str(path.relative_to(destination)).replace('\\', '/')}
        else:
            raise ValueError((item['key'], kind, record.get('family')))
        cases.append(case)
    assert len(cases) == 126 and len({c['key'] for c in cases}) == 126
    (destination / 'cases.json').write_text(json.dumps({
        'schema': 'registered-126-portable-inputs-v1',
        'source_protocol_sha256': sha(protocol_file),
        'steps': 5, 'joint_jacobian_cap': 21, 'cases': cases,
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    audit = root / 'workspace/results_stage9_dev/runs/paper21x6_two_models_20260925/input_audit.json'
    shutil.copyfile(audit, destination / 'input_audit.json')
    manifest = {str(p.relative_to(destination)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(data.rglob('*')) if p.is_file()}
    (destination / 'DATA_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'cases': len(cases), 'data_files': len(manifest),
                      'data_bytes': sum(v['bytes'] for v in manifest.values())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE)
    args = parser.parse_args()
    export(args.research_root, args.output)
