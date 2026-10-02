"""Single-checkpoint loading and fixed five-step synthetic examples."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from setsunet_csn.model_stage5 import NormalizedSchurNet
from setsunet_csn.stage5 import Stage5Config, solve
from stage7_tasks import compile_case

ROOT = Path(__file__).resolve().parent


def load_model():
    folder = ROOT / 'checkpoint'
    architecture = json.loads((folder / 'architecture.json').read_text(encoding='utf-8'))
    provenance = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
    model = NormalizedSchurNet(**architecture).double()
    with np.load(folder / 'weights.npz', allow_pickle=False) as archive:
        state = {}
        if set(archive.files) != set(provenance['tensor_sha256']):
            raise ValueError('Checkpoint tensor names differ from the recorded model')
        for name in archive.files:
            array = archive[name].copy()
            if hashlib.sha256(array.tobytes()).hexdigest() != provenance['tensor_sha256'][name]:
                raise ValueError(f'Checkpoint tensor hash mismatch: {name}')
            state[name] = torch.from_numpy(array)
        model.load_state_dict(state, strict=True)
    if sum(p.numel() for p in model.parameters()) != provenance['parameters']:
        raise ValueError('Unexpected parameter count')
    return model.eval()


def config(**overrides):
    settings = dict(steps=5, measured_directions=8, completion_directions=8,
                    damping_mode='relative', damping=.001, backend='c',
                    residual_cg_iterations=2, cg_iterations=16)
    settings.update(overrides)
    return Stage5Config(**settings)


def cases():
    return [dict(family='view_graph', n=96, seed=20261002, condition=condition)
            for condition in ['balanced', 'noisy_view', 'biased_view']] + [
                dict(family='broyden', n=64, seed=20261003)]


def run_case(spec, model, method='csn'):
    problem, evaluate = compile_case(spec)
    result = solve(problem, model if method == 'csn' else None, method, config())
    theta = result['theta'].detach().numpy()
    return result, evaluate(theta)

