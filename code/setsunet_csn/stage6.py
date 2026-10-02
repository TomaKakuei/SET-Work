"""Stage 6 milestone: differentiable CSN trajectories, unchanged objective.

Discrete rank, line-search, and damping-policy decisions use the executed branch.
The minimax solution differentiates its interior dual optimum implicitly.
Native inference and differentiable reference linearization share the same GN
definition. No query truth is accepted by this module.
"""
from dataclasses import dataclass, asdict
import time
import torch
from .core import sym, minimax
from .stage4 import Stage4Config, counters, WhiteView, measured_base, orthogonalize
from .curvature_stage4 import DenseView, BlockMetric
from .curvature_stage5 import linearize as native_linearize
from .stage3 import predicted_reduction


@dataclass
class Stage6Config(Stage4Config):
    steps: int = 5
    measured_directions: int = 8
    completion_directions: int = 8
    completion: str = 'consensus'


def load_model(checkpoint, *, fast_inference=True):
    """Load a milestone or original checkpoint without changing any defaults.

The fast model can be passed directly to stage5.solve for normal C inference;
stage6.solve additionally records subspace diagnostics and supports training.
"""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('schema') not in ('setsunet-csn-stage4-v1','setsunet-csn-stage6-milestone1-v1'):
        raise ValueError('Unsupported CSN checkpoint schema')
    if fast_inference:
        from .model_stage5 import NormalizedSchurNet
    else:
        from .model_stage4 import NormalizedSchurNet
    model = NormalizedSchurNet(**payload['architecture']).double()
    model.load_state_dict(payload['state_dict'])
    return model


class _SPDRoot(torch.autograd.Function):
    """Spectral SPD root with a derivative finite at repeated eigenvalues."""
    @staticmethod
    def forward(ctx, matrix, inverse):
        eig, vec = torch.linalg.eigh(matrix)
        if bool((eig <= 0).any()):
            raise ValueError('Differentiable whitening requires positive definite blocks')
        roots = eig.sqrt()
        ctx.save_for_backward(roots, vec)
        ctx.inverse = inverse
        values = roots.reciprocal() if inverse else roots
        return (vec * values.unsqueeze(-2)) @ vec.transpose(-1, -2)

    @staticmethod
    def backward(ctx, upstream):
        roots, vec = ctx.saved_tensors
        denom = roots.unsqueeze(-1) + roots.unsqueeze(-2)
        divided = denom.reciprocal()
        if ctx.inverse:
            divided = -divided / (roots.unsqueeze(-1) * roots.unsqueeze(-2))
        local = vec.transpose(-1, -2) @ (.5*(upstream+upstream.transpose(-1,-2))) @ vec
        return vec @ (divided * local) @ vec.transpose(-1, -2), None


def differentiable_metric(ids, normal):
    # Preserve the original block layout and symmetric whitening convention.
    layout = BlockMetric(ids, normal=normal.detach())
    ix, mask = layout.indices, layout.mask
    blocks = normal[ix[:,:,None], ix[:,None,:]] * mask[:,:,None] * mask[:,None,:]
    blocks = blocks + torch.diag_embed((~mask).to(normal.dtype))
    layout.inverse = _SPDRoot.apply(blocks, True)
    layout.root = _SPDRoot.apply(blocks, False)
    return layout


def implicit_minimax(Ha, Hb, ba, bb, counts=None):
    # The old envelope derivative is sufficient for optimal gain, but not for
    # differentiating the resulting step or an arbitrary downstream loss.
    _, t0, _, _, _ = minimax(Ha.detach(), Hb.detach(), ba.detach(), bb.detach(), counts)
    dH, db = Ha-Hb, ba-bb
    H, b = Hb+t0*dH, bb+t0*db
    alpha = torch.linalg.solve(H, b)
    t = t0
    if 0. < t0 < 1.:
        stationarity = 2*(db @ alpha) - alpha @ dH @ alpha
        u = db-dH@alpha
        slope = 2*(u @ torch.linalg.solve(H, u))
        if float(slope.detach()) > torch.finfo(H.dtype).tiny**.25:
            # Zero-valued correction preserves the bisection forward solution.
            t = t0-(stationarity-stationarity.detach())/slope.detach()
    alpha = torch.linalg.solve(Hb+t*dH, bb+t*db)
    gain = .5*((bb+t*db) @ alpha)
    changes = torch.stack((.5*alpha@Ha@alpha-ba@alpha,
                           .5*alpha@Hb@alpha-bb@alpha))
    if counts is not None: counts['small_linear_solves'] += 3
    return alpha, t0, gain, changes.max()+gain, changes


def reference_linearize(problem, x, damping, counts, config):
    views = []
    for v in (0,1):
        fn = lambda z: problem.raw(z, v).reshape(-1)
        raw = problem.raw(x, v)
        J = torch.func.jacfwd(fn)(x)
        weights = problem.weights(x, v).expand_as(raw).reshape(-1)
        # Frozen IRLS within each local quadratic; weights still depend on
        # the iterate when differentiating an entire trajectory.
        H = sym(J.T @ (weights[:,None]*J))
        g = J.T @ (weights*raw.reshape(-1))
        views.append(DenseView(H, g, counts))
        counts['linearizations'] += 1
        counts['jacobian_evaluations'] += 1
        counts['jvp_directions'] += x.numel()
    diagonal = .5*(views[0].H.diagonal()+views[1].H.diagonal())
    scale = float((problem.metadata or {}).get('objective_scale',1.))
    if config.damping_mode == 'declared': shift = x.new_full(x.shape, damping*scale)
    elif config.damping_mode == 'relative': shift = damping*diagonal.mean().clamp_min(1e-30).expand_as(x)
    elif config.damping_mode == 'diagonal':
        shift = damping*diagonal.clamp_min(max(float(diagonal.detach().max())*1e-12,1e-30))
    else: raise ValueError(config.damping_mode)
    for view in views: view.H = view.H+torch.diag(shift)
    edges = torch.as_tensor(problem.metadata['block_edges'],dtype=torch.long)
    return *views, edges, {'kind':'differentiable_reference_GN'}


def completion(base, a, b, V, config, counts):
    Q, A, Y = base['Q'],base['A'],base['Y']
    mode = config.completion
    if mode == 'base':
        step = base['p0']
        return step, .5*(a(step)+b(step)), dict(rank=0, certificate_fallback=False)
    if mode == 'joint_mean':
        U = orthogonalize(torch.cat((Q,V),dim=1),config.rank_tolerance)
        AU = .5*(a.columns(U)+b.columns(U))
        step = -U@torch.linalg.solve(sym(U.T@AU), U.T@base['g'])
        return step,.5*(a(step)+b(step)),dict(rank=U.shape[1],certificate_fallback=False)
    if mode not in ('consensus','mean'): raise ValueError(mode)
    Z = V
    if Q.shape[1]:
        Z = Z-Q@torch.linalg.solve(A,Y.T@Z)
        U = torch.linalg.qr(Y,mode='reduced')[0]
        for _ in range(2): Z = Z-U@(U.T@Z)
    Z = orthogonalize(Z,config.rank_tolerance,V.norm().detach())
    c = base['c']
    if not Z.shape[1]:
        return base['p0'],-Y@c,dict(rank=0,certificate_fallback=False)
    Za,Zb = a.columns(Z),b.columns(Z)
    Ha,Hb = sym(Z.T@Za),sym(Z.T@Zb)
    ba,bb = Z.T@(a.g-base['Ya']@c),Z.T@(b.g-base['Yb']@c)
    if mode == 'consensus':
        alpha,weight,gain,gap,changes = implicit_minimax(Ha,Hb,ba,bb,counts)
    else:
        alpha = torch.linalg.solve(.5*(Ha+Hb),.5*(ba+bb))
        gain = .25*(ba+bb)@alpha
        changes = torch.stack((.5*alpha@Ha@alpha-ba@alpha,.5*alpha@Hb@alpha-bb@alpha))
        gap,weight = changes.max()+gain,.5
    step,action = base['p0']-Z@alpha,-Y@c-.5*(Za+Zb)@alpha
    lock = (Q.T@(base['g']+action)).norm()/base['g'].norm().clamp_min(1e-300)
    scale = max(float(gain.detach().abs()),float(base['g'].detach().square().sum()),1e-30)
    reject = not bool(torch.isfinite(step).all()) or float(lock.detach())>config.certificate_tolerance
    if mode == 'consensus':
        reject = reject or float(changes.detach().max())>config.certificate_tolerance*scale or abs(float(gap.detach()))>config.certificate_tolerance*scale
    diag = dict(rank=Z.shape[1],weight=weight,gain=float(gain.detach()),gap=float(gap.detach()),
                lock_error=float(lock.detach()),certificate_fallback=reject,
                completion_step_norm=float((Z@alpha).detach().norm()))
    if reject: return base['p0'],-Y@c,diag
    return step,action,diag


def solve(problem, model=None, method='csn', config=None, *, training=False, detach_rounds=False):
    config = config or Stage6Config()
    if method not in ('csn','analytic','lm'): raise ValueError(method)
    x = problem.initial.detach().double().clone()
    if training: x.requires_grad_(True)
    ids = problem.block_id
    counts,trajectory,states,objectives,local_losses = counters(),[],[],[],[]
    history,damping = None,config.damping
    started = time.perf_counter()
    def costs(z):
        counts['objective_evaluations'] += 2
        if not training and hasattr(problem,'native'):
            return torch.stack([problem.native.cost(z,v) for v in (0,1)])
        return torch.stack([problem.cost(z,v) for v in (0,1)])
    initial = costs(x)
    before = initial
    with torch.enable_grad() if training else torch.no_grad():
        for iteration in range(config.steps):
            linearizer = reference_linearize if training else native_linearize
            a,b,edges,note = linearizer(problem,x,damping,counts,config)
            g,H = .5*(a.g+b.g),.5*(a.H+b.H)
            diag = dict(rank=0,certificate_fallback=False)
            if method == 'lm':
                step = torch.linalg.solve(H,-g)
                action = H@step
                directional,quadratic = g@step,step@action
            else:
                metric = differentiable_metric(ids,H) if training else BlockMetric(ids,normal=H)
                wa,wb = WhiteView(a,metric,counts),WhiteView(b,metric,counts)
                base = measured_base(wa,wb,ids,edges,config.measured_directions,
                    None if history is None else metric.apply(history,inverse=False),config,counts)
                if method == 'analytic':
                    V = orthogonalize(torch.cat((base['e'][:,None],base['Y'],base['Ya']-base['Yb']),dim=1),config.rank_tolerance)[:,:config.completion_directions]
                else:
                    V = model(base,wa.g,wb.g,ids,edges,directions=config.completion_directions)
                    counts['network_calls'] += 1
                wstep,waction,diag = completion(base,wa,wb,V,config,counts)
                if training:
                    residual=(base['g']+waction).square().sum()/base['g'].square().sum().clamp_min(1e-30)
                    unit=V/V.norm(dim=0,keepdim=True).clamp_min(1e-20)
                    gram=unit.T@unit
                    local_losses.append(residual+.005*(gram-torch.diag_embed(gram.diagonal())).square().mean())
                step = metric.apply(wstep)
                directional,quadratic = base['g']@wstep,wstep@waction
                full = torch.linalg.solve(.5*(wa.H+wb.H),-base['g'])
                diag['relative_LM_step_distance'] = float((wstep-full).detach().norm()/full.detach().norm().clamp_min(1e-30))
                diag['measured_rank'] = base['Q'].shape[1]
            accepted,alpha,after,rho = False,0.,before,None
            for backtrack in range(config.backtracks):
                scale = .5**backtrack
                candidate = x+scale*step
                observed = costs(candidate)
                if bool(torch.isfinite(observed).all()) and float(observed.detach().mean()) <= float(before.detach().mean())+config.armijo*scale*float(directional.detach())+1e-14*max(float(before.detach().abs().mean()),1e-30):
                    actual = float((before-observed).detach().mean())
                    predicted = predicted_reduction(float(directional.detach()),float(quadratic.detach()),scale)
                    rho = actual/max(predicted,1e-300)
                    history,x = candidate-x,candidate
                    accepted,alpha,after = True,scale,observed
                    break
            old = damping
            if accepted:
                if rho>.75 and alpha==1.: damping=max(config.damping_min,damping*.3)
                elif rho<.25: damping=min(config.damping_max,damping*3.)
            else: history=None;damping=min(config.damping_max,damping*10.)
            states.append(x)
            objectives.append(after.mean())
            trajectory.append(dict(iteration=iteration,accepted=accepted,step_scale=alpha,
                cost_before=before.detach().tolist(),cost_after=after.detach().tolist(),
                damping_before=old,damping_after=damping,**diag))
            before = after
            if detach_rounds:
                x=x.detach().requires_grad_(training)
                history=None if history is None else history.detach()
                before=before.detach()
    return dict(theta=x,initial_cost=initial.detach().tolist(),final_cost=before.detach().tolist(),
                objectives=objectives,states=states,local_losses=local_losses,trajectory=trajectory,iterations=config.steps,
                seconds=time.perf_counter()-started,counts=counts,config=asdict(config))
