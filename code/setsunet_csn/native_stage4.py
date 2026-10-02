"""ctypes bridge to the tested float64 C small-space kernels."""
from pathlib import Path
import sys
import ctypes
import numpy as np
import torch

_library=None


def library():
    global _library
    if _library is None:
        path=Path(__file__).resolve().parents[1]/'native_stage4'/('csn_small'+('.dll' if sys.platform=='win32' else '.dylib' if sys.platform=='darwin' else '.so'))
        _library=ctypes.CDLL(str(path))
        pointer=ctypes.POINTER(ctypes.c_double)
        _library.csn_minimax.argtypes=[ctypes.c_int]+[pointer]*6
        _library.csn_minimax.restype=ctypes.c_int
        _library.csn_orth.argtypes=[ctypes.c_int,ctypes.c_int,pointer,ctypes.c_double,ctypes.c_double,pointer]
        _library.csn_orth.restype=ctypes.c_int
    return _library


def array(tensor):
    if tensor.device.type!='cpu' or tensor.dtype!=torch.float64: raise ValueError('C backend requires CPU float64')
    return tensor.detach().contiguous().numpy()


def ptr(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def orthogonalize_c(V,tolerance,reference=None):
    source=array(V)
    output=np.zeros_like(source)
    ref=float(V.detach().norm()) if reference is None else float(reference)
    rank=library().csn_orth(V.shape[0],V.shape[1],ptr(source),tolerance,ref,ptr(output))
    if rank<0: raise RuntimeError(f'C orthogonalization rejected input ({rank})')
    return torch.from_numpy(output[:,:rank].copy())


def minimax_c(Ha,Hb,ba,bb,counts=None):
    values=[array(v) for v in (Ha,Hb,ba,bb)]
    out=np.zeros(ba.numel(),dtype=np.float64)
    stats=np.zeros(6,dtype=np.float64)
    status=library().csn_minimax(ba.numel(),*(ptr(v) for v in values),ptr(out),ptr(stats))
    if status: raise RuntimeError(f'C minimax SPD solve failed ({status})')
    if counts is not None:
        counts['native_calls']+=1
        counts['small_linear_solves']+=int(stats[5])
    weight=float(stats[0])
    if any(v.requires_grad for v in (Ha,Hb,ba,bb)):
        # The dual optimizer is detached, exactly as in the Torch envelope rule.
        H=Hb+weight*(Ha-Hb);b=bb+weight*(ba-bb)
        result=torch.linalg.solve(H,b)
        if counts is not None: counts['small_linear_solves']+=1
        gain=.5*b@result
        changes=torch.stack((.5*result@Ha@result-ba@result,.5*result@Hb@result-bb@result))
        return result,weight,gain,changes.max()+gain,changes
    return torch.from_numpy(out),weight,Ha.new_tensor(stats[1]),Ha.new_tensor(stats[2]),Ha.new_tensor(stats[3:5])
