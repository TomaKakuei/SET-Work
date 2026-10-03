"""Complete the actually selected full-state path using one endpoint jet."""
import time
import numpy as np
import torch
from .core import ModelConfig,hermite_coefficients,squared_polynomial,polynomial_segment
from .hooks import path_hook,finish
from . import backbone
from run_fiber_system_20260924 import PhysicalAudit
from dictionary_matching_core_20260924 import monomials


def make_hook(problem,audit,packet):
    config=ModelConfig('path_compact');parent_packet={};parent=path_hook(config,problem,audit,parent_packet)
    def hook(*args):
        if args[0]!=1:return args[2],args[4],False
        parent(*args);audit.phase='intervention';tic=time.perf_counter();a=parent_packet['arrays'];native=problem.native
        detail=dict(parent_packet['detail']);states=list(a['candidate_states']);residuals=list(a['candidate_residuals'])
        labels=['path','path_half','path_quarter','GN','GN_half','GN_quarter','anchor','anchor_half','anchor_quarter']
        index=detail['selected_index'];diagnostic=dict(active=False,reason='incumbent',extra_endpoint_J=False)
        if index>0:
            end=states[index-1];send=np.linalg.solve(a['T'],end-a['proposal'])
            r1=residuals[index-1]/a['normalizer'];rr=a['rr'];v=a['J']@send;curvature=a['full']@monomials(send,a['pairs'])
            coeff=squared_polynomial([rr,v,curvature]);coeff[2]+=.5*float(a['mu'])*(send@send)
            t,sol=polynomial_segment(coeff)
            diagnostic=dict(active=True,reason='quadratic_radial',model_t=t,model_solve=sol,extra_endpoint_J=False)
            z=a['proposal']+a['T']@(t*send)
            states.append(z);residuals.append(native.least_squares(z,False)[0]);labels.append('radial_quadratic')
            # A nonzero fitted nonlinear response justifies a physical witness.
            if np.linalg.norm(curvature)>1e-12*max(np.linalg.norm(v),1e-30):
                mid=a['proposal']+.5*(end-a['proposal']);rm,_=native.least_squares(mid,False);rm=rm/a['normalizer']
                states.append(mid);residuals.append(rm*a['normalizer']);labels.append('radial_midpoint')
                midpoint_error=float(np.linalg.norm(rm-rr-.5*v-.25*curvature))
                endpoint_error=float(np.linalg.norm(r1-rr-v-curvature))
                diagnostic.update(midpoint_error=midpoint_error,endpoint_error=endpoint_error)
                if max(midpoint_error,endpoint_error)>config.path_witness_tolerance:
                    r_end,G_end=native.least_squares(end,True)
                    h=hermite_coefficients(rr,r_end/a['normalizer'],v,G_end@(end-a['proposal'])/a['normalizer'])
                    hm=np.array([1,.5,.25,.125])@h
                    herror=float(np.linalg.norm(hm-rm));diagnostic.update(extra_endpoint_J=True,hermite_midpoint_error=herror,reason='hermite_rejected')
                    if herror<=midpoint_error+config.path_witness_tolerance:
                        hc=squared_polynomial(h);hc[2]+=.5*float(a['mu'])*(send@send)
                        ht,hsol=polynomial_segment(hc);z=a['proposal']+ht*(end-a['proposal'])
                        states.append(z);residuals.append(native.least_squares(z,False)[0]);labels.append('radial_hermite')
                        diagnostic.update(reason='hermite_accepted',hermite_t=ht,hermite_solve=hsol)
                        a['radial_hermite_coefficients']=h
            a['radial_end']=send
        selected,cost,changed,acceptance=finish(native,args[2],args[4],states,residuals,labels,config)
        a.update(selected=selected,selected_cost=cost,candidate_states=np.array(states),candidate_residuals=np.array(residuals))
        detail.update(method='path_radial_verified',radial=diagnostic,**acceptance);detail['seconds']+=time.perf_counter()-tic
        packet.update(arrays=a,detail=detail);audit.phase='solver'
        return torch.from_numpy(selected).to(args[2]),torch.from_numpy(cost).to(args[4]),changed
    return hook


def solve(problem,network,engine,outer_config,method='path_radial_verified'):
    if method!='path_radial_verified' or outer_config.steps!=5:raise ValueError('Registered radial five-step route only')
    native=problem.native;original=native.least_squares;cache={};hits=0
    def cached(x,jacobian=True):
        nonlocal hits
        key=np.asarray(x,dtype=np.float64).tobytes()
        if key in cache and (not jacobian or cache[key][1] is not None):
            hits+=1;return cache[key] if jacobian else (cache[key][0],None)
        value=original(x,jacobian);cache[key]=value;return value
    native.least_squares=cached;audit=PhysicalAudit(native);packet={};failures=[];hook=make_hook(problem,audit,packet)
    def guarded(*args):
        try:return hook(*args)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failures.append(dict(kind=type(exc).__name__,message=str(exc),iteration=int(args[0])+1));packet.clear();audit.phase='solver';return args[2],args[4],False
    try:
        result=backbone.solve(problem,network,engine,curve_mode='norm_mean_envelope',config=outer_config,hook=guarded)
        result.update(physical_counts=audit.counts(),cache_hits=hits,refinement_failures=failures)
    finally:audit.close();native.least_squares=original
    return result,packet
