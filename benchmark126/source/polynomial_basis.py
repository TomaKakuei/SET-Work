"""The frozen inverse-fiber powers/basis/design routines without its CLI."""
import itertools
import numpy as np


def powers(k, d):
    return np.array([a for a in itertools.product(range(d+1), repeat=k)
                     if sum(a) <= d], int)


def basis(z, E):
    z = np.asarray(z)
    return np.prod(z[..., None, :] ** E, axis=-1)


def dbasis(z, E):
    z = np.asarray(z)
    out = []
    for j in range(E.shape[1]):
        D = E.copy()
        D[:, j] = np.maximum(0, D[:, j]-1)
        out.append(np.prod(z[..., None, :] ** D, axis=-1) * E[:, j])
    return np.stack(out, axis=-1)


def design(nodes, E):
    return np.concatenate([np.vstack([basis(z, E), dbasis(z, E).T])
                           for z in nodes])
