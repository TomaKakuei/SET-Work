"""Opt-in signed decoder normalization for cached multi-frame functions.

The serialized compiler has function-global dictionaries of its own. Patch
only their homography decoder binding, restore every dictionary on exit, and
run serially. Source files, compiled inputs, weights and policy stay untouched.
"""
from contextlib import contextmanager
import types
import torch


def signed_normalize(H):
    h33=H[...,2:3,2:3]
    magnitude=H.abs().amax(dim=(-2,-1),keepdim=True)
    usable=h33.abs()>torch.finfo(H.dtype).eps*64*magnitude
    denominator=torch.where(usable,h33,magnitude.clamp_min(torch.finfo(H.dtype).tiny))
    return H/denominator


@contextmanager
def decoder_normalization(problem):
    seen,bindings=set(),{}
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value,types.FunctionType):
            mapping=value.__globals__
            original=mapping.get('homography_from_theta')
            if callable(original):
                bindings[id(mapping)]=(mapping,original)
            for cell in value.__closure__ or ():
                try:
                    visit(cell.cell_contents)
                except ValueError:
                    pass
            visit(value.__defaults__)
            visit(value.__kwdefaults__)
        elif isinstance(value,(tuple,list)):
            for item in value:
                visit(item)
        elif isinstance(value,dict):
            for item in value.values():
                visit(item)
    for value in (problem.raw_function,problem.cost_function,problem.weight_function,problem.legacy_views):
        visit(value)
    if not bindings:
        raise ValueError('No serialized multi-frame decoder binding found')
    try:
        for mapping,original in bindings.values():
            def wrapped(*args,_original=original,**kwargs):
                return signed_normalize(_original(*args,**kwargs))
            mapping['homography_from_theta']=wrapped
        yield dict(decoder_bindings=len(bindings),fix='signed H33 with relative fallback',source_files_changed=False)
    finally:
        for mapping,original in bindings.values():
            mapping['homography_from_theta']=original
