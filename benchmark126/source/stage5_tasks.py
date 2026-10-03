"""Stage 3 task definitions with an explicit support-only native factor export."""
from pathlib import Path
import os
import hashlib
import math
import numpy as np
import torch
from setsunet_csn.problems import ResidualProblem, bilinear
from setsunet_csn.native_stage5 import NativeFactors

HP_ROOT = Path(os.environ.get('CSN_HPATCHES_ROOT', os.path.relpath(Path(__file__).resolve().parents[2] / 'datasets/hpatches/hpatches-sequences-release')))


def wrap(initial, ids, raw, edges, family, metadata, delta=.05):
    def cost(x, view):
        r = raw(x, view)
        return (r.square() / (torch.sqrt(1 + r.square() / delta**2) + 1)).sum()
    def weights(x, view):
        return torch.rsqrt(1 + raw(x, view).square() / delta**2)
    def factors(batch, view):
        packed = []
        for x in batch:
            residual = raw(x, view)
            values = residual.square() / (torch.sqrt(1 + residual.square() / delta**2) + 1)
            # Legacy particles form matrices in factor space; raw pixels make
            # that workspace quadratic in thousands of observations. Preserve
            # the exact objective while using 32 fixed contiguous scalar packs.
            packed.append(torch.stack([chunk.sum() for chunk in torch.tensor_split(values, min(32, values.numel()))]))
        return torch.stack(packed)
    return ResidualProblem(initial, ids, raw, cost, weights,
        (lambda x: factors(x, 0), lambda x: factors(x, 1)),
        metadata={"family": family, "block_edges": edges, "robust": "pseudo-Huber frozen IRLS",
            "legacy_factorization": "32 fixed contiguous summed factors per view; exact same support objective", **metadata})


def radiometric(sequence):
    """Shared gamma, per-frame gain/offset/spatial field; no homography GT."""
    from neural_grey_v2.hpatches_homography import SIFTHomographyCompiler, read_grayscale
    root = HP_ROOT / sequence
    compiler = SIFTHomographyCompiler()
    reference = read_grayscale(root / "1.ppm")
    h, w = reference.shape
    axis = torch.linspace(.06, .94, 32, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((xx.flatten()*(w-1), yy.flatten()*(h-1)), dim=1)
    source_values = bilinear(torch.tensor(reference / 255., dtype=torch.float64), xy)
    grid_y, grid_x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    # Whole 4x4 tiles, not adjacent pixel interleaving; split declared in advance.
    partition = ((grid_y//4 + 2*(grid_x//4)) % 4).flatten()
    frame_rows, source_hashes = [], {}
    for image_index in range(1, 7):
        p = root / f"{image_index}.ppm"
        source_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    ref_features = compiler.extract(reference)
    failures = []
    for target in range(2, 7):
        image = read_grayscale(root / f"{target}.ppm")
        observation = compiler.compile_features(ref_features, compiler.extract(image))
        if not observation.succeeded:
            failures.append({"target": target, "status": observation.status})
            continue
        H = torch.tensor(observation.initial_homography, dtype=torch.float64)
        homogeneous = torch.cat((xy, torch.ones(xy.shape[0], 1, dtype=xy.dtype)), dim=1) @ H.T
        mapped = homogeneous[:, :2] / homogeneous[:, 2:]
        valid = (mapped[:, 0] > 2) & (mapped[:, 0] < image.shape[1]-3) & (mapped[:, 1] > 2) & (mapped[:, 1] < image.shape[0]-3)
        # Source saturation only; held-out target values do not choose the mask.
        valid &= (source_values > .03) & (source_values < .97)
        target_values = bilinear(torch.tensor(image / 255., dtype=torch.float64), mapped)
        u, v = 2*mapped[:, 0]/(image.shape[1]-1)-1, 2*mapped[:, 1]/(image.shape[0]-1)-1
        r2 = .5*(u.square()+v.square())
        design = torch.stack((torch.ones_like(u), r2, r2.square(), u, v), dim=1)
        if int((valid & (partition == 3)).sum()) < 32 or int((valid & (partition != 3)).sum()) < 96:
            failures.append({"target": target, "status": "insufficient_valid_tiles"})
            continue
        frame_rows.append((source_values[valid], target_values[valid], design[valid], partition[valid], target))
    if len(frame_rows) < 3:
        raise ValueError(f"Only {len(frame_rows)} compiled frames: {failures}")
    initial = torch.zeros(1 + 6*len(frame_rows), dtype=torch.float64)
    ids = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(1, 1+len(frame_rows)).repeat_interleave(6)))
    edges = torch.stack((torch.zeros(len(frame_rows), dtype=torch.long), torch.arange(1, 1+len(frame_rows))))
    def predict(x, row, index):
        source, _, design, _, _ = row
        local = x[1+6*index:1+6*(index+1)]
        gamma = torch.exp(.5*torch.tanh(x[0]))
        return torch.exp((design @ local[:5]).clamp(-8, 8)) * source.pow(gamma) + .25*torch.tanh(local[5])
    def raw(x, view):
        residuals = []
        for i, row in enumerate(frame_rows):
            support = row[3] != 3
            selected = support & ((row[3] % 2) == view)
            # Both views have comparable objective scale despite unequal tile counts.
            residuals.append((predict(x, row, i)[selected] - row[1][selected]) / math.sqrt(int(selected.sum()) / 128))
        residuals.append(.01*x)
        return torch.cat(residuals)
    def evaluate(x):
        per_frame = [(predict(x, row, i)[row[3] == 3] - row[1][row[3] == 3]).square().mean() for i, row in enumerate(frame_rows)]
        return {"metric": float(torch.stack(per_frame).mean().sqrt()), "query_rmse": float(torch.stack(per_frame).mean().sqrt())}
    metadata = {"sequence": sequence, "frames": len(frame_rows), "frame_failures": failures,
        "source_sha256": source_hashes, "query": "partition 3 of deterministic 4x4 spatial tiles",
        "query_limit": "SIFT compiler sees complete images; query residuals are excluded from fitting, not fully unseen images",
        "support_samples": sum(int((r[3] != 3).sum()) for r in frame_rows),
        "query_samples": sum(int((r[3] == 3).sum()) for r in frame_rows),
        "data_status": "real images, new photometric task on previously evaluated HPatches dataset; not physical parameter ground truth"}
    problem=wrap(initial, ids, raw, edges, "radiometric", metadata)
    views=[]
    for view in (0,1):
        ii,dd=[],[]
        for index,row in enumerate(frame_rows):
            selected=(row[3]!=3)&(row[3]%2==view)
            count=int(selected.sum())
            ii.extend([index]*count)
            dd.append(torch.cat((row[0][selected,None],row[1][selected,None],row[2][selected],torch.full((count,1),math.sqrt(128/count),dtype=torch.float64)),dim=1))
        views.append((ii,torch.cat(dd),len(ii)))
    problem.native=NativeFactors(1,initial.numel(),views)
    return problem,evaluate


def sensor(seed, nodes):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    anchors = torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]], dtype=dtype)
    positions = .1 + .8*torch.rand(nodes, 2, generator=generator, dtype=dtype)
    bias = .025*torch.randn(nodes, generator=generator, dtype=dtype)
    truth = torch.cat((positions, bias[:,None]), dim=1)
    perturbation = .08 if seed % 2 else .2
    initial = truth.clone()
    initial[:,:2] += perturbation*torch.randn(nodes, 2, generator=generator, dtype=dtype)
    initial[:,2] = 0
    initial = initial.flatten()
    all_truth = torch.cat((anchors, positions))
    pairs = set()
    for i in range(nodes):
        pairs.add((i % 4, i+4))
        pairs.add(((i+1) % 4, i+4))
        pairs.add(tuple(sorted((i+4, (i+1)%nodes+4))))
        nearest = torch.argsort((positions-positions[i]).square().sum(1))[1:4]
        for j in nearest.tolist():
            pairs.add(tuple(sorted((i+4, j+4))))
    pair_tensor = torch.tensor(sorted(pairs), dtype=torch.long)
    source, target = pair_tensor.T
    full_bias = torch.cat((torch.zeros(4,dtype=dtype), bias))
    clean = (all_truth[source]-all_truth[target]).square().sum(1).sqrt() + full_bias[source]+full_bias[target]
    sigma = .005 if seed % 3 else .02
    observations = [clean + sigma*torch.randn(clean.shape, generator=generator, dtype=dtype) for _ in (0,1)]
    if seed % 4 == 0:
        for view in (0,1):
            mask = torch.rand(clean.shape, generator=generator) < .1
            observations[view] += mask * .12*torch.randn(clean.shape, generator=generator,dtype=dtype)
    def predictions(x):
        local = x.reshape(nodes, 3)
        p = torch.cat((anchors, local[:,:2]))
        b = torch.cat((torch.zeros(4,dtype=dtype), local[:,2]))
        return torch.sqrt((p[source]-p[target]).square().sum(1)+1e-12) + b[source]+b[target]
    def raw(x, view):
        return torch.cat((predictions(x)-observations[view], .03*x.reshape(nodes,3)[:,2]))
    edge_rows = pair_tensor[(source>=4) & (target>=4)] - 4
    ids = torch.arange(nodes).repeat_interleave(3)
    def evaluate(x):
        local = x.reshape(nodes, 3)
        return {"metric": float((local[:,:2]-positions).square().mean().sqrt()),
            "position_rmse": float((local[:,:2]-positions).square().mean().sqrt()),
            "bias_rmse": float((local[:,2]-bias).square().mean().sqrt()),
            "clean_range_rmse": float((predictions(x)-clean).square().mean().sqrt())}
    metadata = {"seed": seed, "nodes": nodes, "observations_per_view": len(pairs), "sigma": sigma,
        "initial_position_perturbation": perturbation, "outliers": seed%4==0,
        "gauge": "four known corner anchor positions and zero anchor biases",
        "data_status": "fully synthetic independent held-out seeds; no real sensor validation"}
    problem=wrap(initial, ids, raw, edge_rows.T, "sensor", metadata)
    problem.native=NativeFactors(0,initial.numel(),[(pair_tensor,observations[v],len(pairs)) for v in (0,1)])
    return problem,evaluate


def compile_case(spec):
    family=spec['family']
    if family=='sensor':return sensor(spec['seed'],spec['nodes'])
    if family=='radiometric':return radiometric(spec['key'])
    from setsunet_csn.problems_stage2 import ba_problem,se3_problem
    if family=='ba':
        from neural_grey_v2.synthetic_variable_ba import generate_synthetic_variable_ba
        from benchmark_stage2_ba import synthetic_metrics
        graph=generate_synthetic_variable_ba(*spec['size'],seed=spec['seed'])
        problem=ba_problem(graph);views=[]
        for view in (0,1):
            selected=graph.observation_camera%2==view
            ints=torch.stack((graph.observation_camera[selected],graph.observation_landmark[selected]),dim=1)
            anchor=problem.metadata['fixed_coordinate']
            data=torch.cat((graph.observed_uv[selected].double().flatten(),graph.initial_theta[0,anchor,None].double()))
            views.append((ints,data,int(selected.sum())))
        problem.native=NativeFactors(2,problem.initial.numel(),views,nodes=graph.camera_count,
            landmarks=graph.landmark_count,anchor=anchor,delta=graph.reprojection_delta)
        def evaluate(x):
            values=synthetic_metrics(graph,problem,x)
            return {'metric':values['scale_aligned_geometry_rmse_m'],**values}
        return problem,evaluate
    if family!='se3':raise ValueError(family)
    from neural_grey_v2.synthetic_variable_se3 import generate_synthetic_variable_se3
    from neural_grey_v2.variable_se3_pose_graph import align_estimate_to_truth,se3_pose_graph_metrics
    episode=generate_synthetic_variable_se3(spec['nodes'],seed=spec['seed'])
    graph=episode.graph;problem=se3_problem(graph);views=[]
    for view in (0,1):
        selected=torch.arange(graph.source_points.shape[1])%2==view
        count=int(selected.sum());ints=graph.edges[:,None,:].expand(-1,count,-1).reshape(-1,2)
        data=torch.cat((graph.source_points[:,selected].double(),graph.target_points[:,selected].double(),
            graph.correspondence_mask[:,selected,None].double()),dim=-1).reshape(-1,7)
        views.append((ints,data,len(data)))
    problem.native=NativeFactors(3,problem.initial.numel(),views,nodes=graph.node_count,delta=graph.huber_delta_m)
    def evaluate(x):
        R,t=graph.decode(x[None])
        rr,tt=align_estimate_to_truth(R[0].numpy(),t[0].numpy(),episode.truth_translations)
        values=se3_pose_graph_metrics(rr,tt,episode.truth_rotations,episode.truth_translations)
        return {'metric':float(values['ate_translation_rmse_m']),**{k:float(v) for k,v in values.items()}}
    return problem,evaluate
