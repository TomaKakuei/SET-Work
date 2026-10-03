"""Versioned observation-only joint initialization, never a default policy.

Preserves cached correspondences, objective functions, parameterization and
valid photo masks. Candidate matrices are converted to the original theta.
All solvers receive the same resulting initial state.
"""
from dataclasses import replace
from itertools import combinations
import numpy as np
import cv2
import torch
from native_multiframe_diagnostic import cells
from neural_grey_v2.hpatches_homography import HOMOGRAPHY_PARAMETER_SCALES

RULE = dict(weak_count=12, weak_coverage=.005, contradiction_px=6.,
            minimum_consistency_reduction=.25, minimum_objective_reduction=.1,
            maximum_view_cost_increase=.05, maximum_trees=512)


def canonical(matrix):
    H=np.asarray(matrix,dtype=np.float64)
    magnitude=np.max(np.abs(H))
    if not np.isfinite(H).all() or magnitude==0:
        raise ValueError('Invalid projective matrix')
    H=H/magnitude
    if abs(H[2,2])>np.finfo(float).eps*64:
        H=H/H[2,2]
    return H


def project(points,H):
    P=np.column_stack((points,np.ones(len(points))))@canonical(H).T
    if np.any(np.abs(P[:,2])<1e-12):
        raise ValueError('Observed point at a projective pole')
    return P[:,:2]/P[:,2:]


def normalization(shape):
    h,w=shape
    return np.array([[2/(w-1),0,-1],[0,2/(h-1),-1],[0,0,1]],dtype=np.float64)


def theta_from_matrices(matrices,original,shapes):
    values=[]
    scales=HOMOGRAPHY_PARAMETER_SCALES.double().numpy()
    for H,H0,shape in zip(matrices,original,shapes):
        N=normalization(shape)
        D=N@canonical(H)@np.linalg.inv(canonical(H0))@np.linalg.inv(N)
        if abs(D[2,2])<=np.finfo(float).eps*64*np.abs(D).max():
            raise ValueError('Candidate lies outside the original local chart')
        D=D/D[2,2]
        values.append(np.array([D[0,0]-1,D[0,1],D[0,2],D[1,0],D[1,1]-1,D[1,2],D[2,0],D[2,1]])/scales)
    return np.concatenate(values)


def spatial_coverage(points,shape):
    if len(points)<3:
        return 0.
    hull=cv2.convexHull(np.asarray(points,dtype=np.float32))
    h,w=shape
    return min(1.,max(0.,cv2.contourArea(hull)/((h-1)*(w-1))))


def graph_edges(problem):
    graph=cells(cells(problem.cost_function)['function'])
    photo=[cells(f) for f in graph['photometric_functions']]
    ref_shape=cells(photo[0]['decode'])['reference_shape']
    target_shapes=[cells(p['decode'])['target_shape'] for p in photo]
    shapes=[ref_shape]+target_shapes
    original=graph['decode'](problem.initial[None]).detach().numpy()[0]
    edges=[]
    for frame,source,target,chunks in graph['reference_data']:
        edges.append(dict(a=0,b=frame+1,source=source.numpy(),target=target.numpy(),H=canonical(original[frame]),kind='reference'))
    for left,right,source,target,chunks in graph['cross_data']:
        H,_=cv2.findHomography(source.numpy(),target.numpy(),0)
        edges.append(dict(a=left+1,b=right+1,source=source.numpy(),target=target.numpy(),H=None if H is None else canonical(H),kind='cross'))
    for edge in edges:
        a,b=edge['a'],edge['b']
        ca,cb=spatial_coverage(edge['source'],shapes[a]),spatial_coverage(edge['target'],shapes[b])
        edge.update(count=len(edge['source']),source_coverage=ca,target_coverage=cb,
                    confidence=len(edge['source'])*np.sqrt(ca*cb))
    return graph,shapes,original,edges


def graph_score(matrices,edges):
    all_matrices=[np.eye(3)]+list(matrices)
    values=[]
    weighted=weight=0.
    for edge in edges:
        relative=canonical(all_matrices[edge['b']]@np.linalg.inv(all_matrices[edge['a']]))
        forward=np.linalg.norm(project(edge['source'],relative)-edge['target'],axis=1)
        backward=np.linalg.norm(project(edge['target'],np.linalg.inv(relative))-edge['source'],axis=1)
        median=float(np.median(.5*(forward+backward)))
        loss=float(np.sqrt(1+(median/3.)**2)-1)
        confidence=edge['confidence']
        weighted+=confidence*loss
        weight+=confidence
        values.append(dict(a=edge['a'],b=edge['b'],median_symmetric_reprojection_px=median,confidence=confidence))
    return weighted/max(weight,np.finfo(float).tiny),values


def tree_matrices(indices,edges,nodes):
    adjacency=[[] for _ in range(nodes)]
    for index in indices:
        edge=edges[index]
        if edge['H'] is None:
            return None
        adjacency[edge['a']].append((edge['b'],edge['H']))
        adjacency[edge['b']].append((edge['a'],np.linalg.inv(edge['H'])))
    matrices={0:np.eye(3)}
    queue=[0]
    for a in queue:
        for b,transfer in adjacency[a]:
            if b not in matrices:
                matrices[b]=canonical(transfer@matrices[a])
                queue.append(b)
    return [matrices[i] for i in range(1,nodes)] if len(matrices)==nodes else None


def initialize(problem):
    if (problem.metadata or {}).get('family')!='hpatches_multiframe':
        return problem,dict(status='not_applicable',changed=False,rule=RULE,truth_read=False)
    graph,shapes,original,edges=graph_edges(problem)
    baseline,baseline_edges=graph_score(original,edges)
    weak=any(e['count']<=RULE['weak_count'] or min(e['source_coverage'],e['target_coverage'])<RULE['weak_coverage'] for e in edges)
    contradiction=any(v['median_symmetric_reprojection_px']>RULE['contradiction_px'] and edges[i]['count']>RULE['weak_count']
                      for i,v in enumerate(baseline_edges))
    diagnostic=dict(status='unchanged_low_conflict',changed=False,rule=RULE,truth_read=False,
        original_score=baseline,original_edges=baseline_edges,
        edge_quality=[{k:e[k] for k in ('a','b','kind','count','source_coverage','target_coverage','confidence')} for e in edges],
        weak_edge_present=weak,strong_edge_contradiction=contradiction)
    if not (weak and contradiction):
        return problem,diagnostic
    rows=[]
    for indices in combinations(range(len(edges)),len(original)):
        try:
            matrices=tree_matrices(indices,edges,len(original)+1)
            if matrices is None:
                continue
            score,details=graph_score(matrices,edges)
            theta=theta_from_matrices(matrices,original,shapes[1:])
            reconstructed=graph['decode'](torch.tensor(theta,dtype=problem.initial.dtype)[None]).detach().numpy()[0]
            physical_error=max(np.linalg.norm(canonical(a)-canonical(b))/max(np.linalg.norm(canonical(b)),1e-12)
                               for a,b in zip(reconstructed,matrices))
            if physical_error>1e-6:
                raise ValueError('Candidate conversion fails physical matrix replay')
            rows.append(dict(tree=list(indices),score=score,theta=theta.tolist(),
                             physical_matrix_replay_error=physical_error,edges=details))
        except (ValueError,np.linalg.LinAlgError) as exc:
            rows.append(dict(tree=list(indices),failure=str(exc)))
        if len(rows)>=RULE['maximum_trees']:
            break
    valid=sorted((r for r in rows if 'score' in r),key=lambda r:(r['score'],r['tree']))
    diagnostic.update(status='degraded_initialization_no_candidate',candidates=rows)
    if not valid:
        return problem,diagnostic
    selected=valid[0]
    initial_cost=np.array([float(problem.cost(problem.initial,v)) for v in (0,1)])
    candidate=torch.tensor(selected['theta'],dtype=problem.initial.dtype)
    final_cost=np.array([float(problem.cost(candidate,v)) for v in (0,1)])
    diagnostic.update(selected_tree=selected['tree'],selected_score=selected['score'],selected_theta=selected['theta'],
                      support_initial_cost=initial_cost.tolist(),support_candidate_cost=final_cost.tolist())
    consistency_pass=selected['score']<=baseline*(1-RULE['minimum_consistency_reduction'])
    objective_pass=np.isfinite(final_cost).all() and final_cost.mean()<=initial_cost.mean()*(1-RULE['minimum_objective_reduction'])
    views_pass=np.all(final_cost<=initial_cost*(1+RULE['maximum_view_cost_increase']))
    diagnostic.update(consistency_pass=bool(consistency_pass),objective_pass=bool(objective_pass),views_pass=bool(views_pass))
    if not (consistency_pass and objective_pass and views_pass):
        diagnostic['status']='degraded_candidate_rejected_by_observation_gate'
        return problem,diagnostic
    diagnostic.update(status='joint_initialization_applied',changed=True)
    metadata=dict(problem.metadata)
    metadata.update(initialization_contract='shared observation graph spanning-tree v1',initialization_credit='project compiler, independent of CSN/LM/PCG16')
    return replace(problem,initial=candidate,metadata=metadata),diagnostic
