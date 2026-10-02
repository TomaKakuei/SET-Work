import numpy as np
import scipy.sparse as sp
from stage7_tasks import AnalyticFactors, pack

def problem_for(spec):
    n = spec['n']
    rng = np.random.default_rng(spec['seed'])
    pairs = sorted({(i, i + stride) for stride in (1, 3) for i in range(n - stride)})
    rr, cc, dd = [], [], []
    for k, (i, j) in enumerate(pairs):
        rr.extend((k, k)); cc.extend((i, j)); dd.extend((1., -1.))
    for i in range(n):
        rr.append(len(pairs) + i); cc.append(i); dd.append(.1)
    A = sp.csr_matrix((dd, (rr, cc)), shape=(len(pairs) + n, n))
    truth = np.sin(np.linspace(0, 2 * np.pi, n)) + .2 * rng.normal(size=n)
    clean = A @ truth
    noise = [rng.normal(size=len(clean)) for _ in (0, 1)]
    bias = .12 * np.sin(np.linspace(0, 4 * np.pi, len(clean)))
    observations = [clean + .01 * noise[0], clean + .01 * noise[1] + bias]
    location = np.array([.5 * (i + j) / (n - 1) for i, j in pairs] + [i / (n - 1) for i in range(n)])
    pattern = np.sin(2 * np.pi * location)
    weights = [np.exp(sign * np.log(spec['contrast']) * pattern) for sign in (1., -1.)]
    roots = [np.sqrt(w / w.mean()) for w in weights]
    matrices = [sp.diags(w) @ A for w in roots]
    targets = [w * y for w, y in zip(roots, observations)]

    def raw_jac(x, view, jac):
        return matrices[view] @ x - targets[view], matrices[view] if jac else None

    engine = AnalyticFactors(n, raw_jac)
    problem = pack(np.zeros(n), np.arange(n), np.array(pairs).T, engine,
        dict(family='view_graph', n=n, seed=spec['seed'], condition='cross_view_precision', contrast=spec['contrast']))

    def evaluate(theta):
        return float(np.sqrt(np.mean((np.asarray(theta) - truth) ** 2)))

    return problem, evaluate
