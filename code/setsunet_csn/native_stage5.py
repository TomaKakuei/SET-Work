"""Explicit support-only factor interface for fused CPU analytic derivatives."""
from pathlib import Path
import sys
import ctypes as ct
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
_LIB=None
D=ct.POINTER(ct.c_double)
I=ct.POINTER(ct.c_int)


def library():
    global _LIB
    if _LIB is None:
        _LIB=ct.CDLL(str(ROOT/'native_stage5'/('csn_factors'+('.dll' if sys.platform=='win32' else '.dylib' if sys.platform=='darwin' else '.so'))))
        fn=_LIB.csn5_factors
        fn.argtypes=[ct.c_int]*6+[I,D,D,ct.c_int,ct.c_double]+[D]*6
        fn.restype=ct.c_int
        _LIB.csn5_block_gram.argtypes=[ct.c_int]*3+[I,D,D]
        _LIB.csn5_block_gram.restype=None
        _LIB.csn5_measured.argtypes=[ct.c_int]*3+[D]*3+[ct.c_double]+[D]*3
        _LIB.csn5_measured.restype=ct.c_int
        _LIB.csn5_features.argtypes=[ct.c_int]*4+[I,I]+[D]*9
        _LIB.csn5_features.restype=None
        _LIB.csn5_proposal.argtypes=[ct.c_int]*3+[I]+[D]*3
        _LIB.csn5_proposal.restype=None
    return _LIB


def ptr(a):return None if a is None else a.ctypes.data_as(D)
def array(x,dtype=np.float64):
    if isinstance(x,torch.Tensor):x=x.detach().cpu().numpy()
    return np.ascontiguousarray(x,dtype=dtype)


class NativeFactors:
    """One immutable problem; no ground truth or scoring closure is stored here."""
    evaluator_backend = 'project_C_analytic_factors'
    evaluates_in_c = True
    def __init__(self,kind,n,views,*,nodes=0,landmarks=0,anchor=-1,delta=.05):
        if kind not in range(4) or n<1 or not np.isfinite(delta) or delta<=0 or len(views)!=2:
            raise ValueError('Invalid native factor declaration')
        if kind==0 and n%3:raise ValueError('Sensor layout requires three coordinates per node')
        if kind==1 and (n-1)%6:raise ValueError('Photometric layout requires global gamma plus six coordinates per frame')
        if kind==2 and (nodes<2 or landmarks<1 or n!=6*(nodes-1)+3*landmarks-1 or not 0<=anchor<=n):
            raise ValueError('BA gauge/layout mismatch')
        if kind==3 and n!=6*(nodes-1):raise ValueError('SE3 pose layout mismatch')
        self.kind,self.n,self.nodes,self.landmarks,self.anchor,self.delta=kind,n,nodes,landmarks,anchor,delta
        self.views=[]
        for integers,data,observations in views:
            ints=array(integers,np.int32);values=array(data)
            observations=int(observations)
            if observations<0 or not np.isfinite(values).all():raise ValueError('Invalid observations')
            if kind==0:
                if ints.shape!=(observations,2) or values.size!=observations or np.any(ints<0) or np.any(ints>=4+n//3):
                    raise ValueError('Invalid sensor incidence/data shape')
            elif kind==1:
                if ints.shape!=(observations,) or values.size!=8*observations or np.any(ints<0) or np.any(ints>=(n-1)//6):
                    raise ValueError('Invalid photometric incidence/data shape')
                values=values.reshape(-1,8)
                if np.any(values[:,0]<=0) or np.any(values[:,7]<=0):raise ValueError('Photometric source and normalization must be positive')
            else:
                if ints.shape!=(observations,2) or np.any(ints<0) or np.any(ints[:,0]>=nodes):raise ValueError('Invalid pose incidence')
                if kind==2 and (values.size!=2*observations+1 or np.any(ints[:,1]>=landmarks)):
                    raise ValueError('Invalid BA observations')
                if kind==3 and (values.size!=7*observations or np.any(ints[:,1]>=nodes)):
                    raise ValueError('Invalid SE3 observations')
                if kind==3 and np.any(values.reshape(-1,7)[:,6]<0):raise ValueError('Negative correspondence mask')
            m=observations+n//3 if kind==0 else observations+n if kind==1 else observations*(2 if kind==2 else 3)
            self.views.append((ints,values,int(observations),int(m)))
        library()
        self.calls=[0,0,0]

    def evaluate(self,x,view,mode=2,*,jacobian=False):
        x=array(x)
        if x.shape!=(self.n,) or not np.isfinite(x).all():raise ValueError('Expected finite CPU float64 parameter vector')
        ints,data,observations,m=self.views[view]
        r=np.empty(m);w=np.empty(m)
        J=np.empty((m,self.n)) if jacobian else None
        g=np.empty(self.n) if mode else None
        H=np.empty((self.n,self.n)) if mode==2 else None
        cost=np.empty(1)
        code=library().csn5_factors(self.kind,self.n,observations,self.nodes,self.landmarks,self.anchor,
            ints.ctypes.data_as(I),ptr(data),ptr(x),mode,self.delta,ptr(r),ptr(J),ptr(g),ptr(H),ptr(w),ptr(cost))
        self.calls[mode]+=1
        if code<0:raise RuntimeError(f'Native residual failure {code}')
        return dict(r=r,J=J,g=g,H=H,w=w,cost=float(cost[0]),group_width=2 if self.kind==2 else 3 if self.kind==3 else 1)

    def cost(self,x,view):return torch.tensor(self.evaluate(x,view,0)['cost'],dtype=torch.float64)

    def objective(self,x,gradient=False):
        outputs=[self.evaluate(x,v,int(gradient)) for v in (0,1)]
        value=.5*sum(o['cost'] for o in outputs)
        return (value,.5*(outputs[0]['g']+outputs[1]['g'])) if gradient else value

    def least_squares(self,x,jacobian=True):
        """Exact pseudo-Huber objective as residual vectors, preserving group norm.

        The transformed Jacobian differentiates the robust transform as well;
        its GN matrix differs from frozen IRLS, but its objective/gradient do not.
        """
        fs,js=[],[]
        for view in (0,1):
            o=self.evaluate(x,view,int(jacobian),jacobian=jacobian)
            d=o['group_width'];r=o['r'].reshape(-1,d);s=(r*r).sum(1)
            t=np.sqrt(1+s/self.delta**2)
            mask=o['w'].reshape(-1,d)[:,0]*t
            a=np.sqrt(2*mask/(t+1))
            fs.append((a[:,None]*r).reshape(-1)/np.sqrt(2))
            if jacobian:
                J=o['J'].reshape(-1,d,self.n)
                da=-a/(4*self.delta**2*t*(t+1))
                transformed=a[:,None,None]*J+2*da[:,None,None]*r[:,:,None]*np.einsum('mi,min->mn',r,J)[:,None,:]
                js.append(transformed.reshape(-1,self.n)/np.sqrt(2))
        return np.concatenate(fs),np.concatenate(js) if jacobian else None


def block_gram(carriers,ids,blocks):
    V=array(carriers);indices=array(ids,np.int32)
    out=np.empty((blocks,V.shape[1]*V.shape[1]))
    library().csn5_block_gram(V.shape[0],blocks,V.shape[1],indices.ctypes.data_as(I),ptr(V),ptr(out))
    return torch.from_numpy(out)


def measured(Ha,Hb,seeds,count,tolerance):
    a,b,s=map(array,(Ha,Hb,seeds));n=len(a)
    Q,Ya,Yb=[np.zeros((n,count)) for _ in range(3)]
    rank=library().csn5_measured(n,count,s.shape[1],ptr(a),ptr(b),ptr(s),tolerance,ptr(Q),ptr(Ya),ptr(Yb))
    if rank<0:raise RuntimeError(f'Native measured-space failure {rank}')
    return tuple(torch.from_numpy(v[:,:rank].copy()) for v in (Q,Ya,Yb))


def features(base,ga,gb,ids,edges):
    ga,gb,e,h,ya,yb,q=map(array,(ga,gb,base['e'],base['history'],base['Ya'],base['Yb'],base['Q']))
    ii=array(ids,np.int32);ee=array(edges.T,np.int32)
    n=len(ga);blocks=int(ii.max())+1;k=ya.shape[1]
    if k>8:raise ValueError('Frozen C feature schema supports at most 8 probes; use an explicit probe-window adapter')
    V=np.empty((n,20));F=np.empty((blocks,436))
    library().csn5_features(n,blocks,k,len(ee),ii.ctypes.data_as(I),ee.ctypes.data_as(I),
        ptr(ga),ptr(gb),ptr(e),ptr(h),ptr(ya),ptr(yb),ptr(q),ptr(V),ptr(F))
    return torch.from_numpy(V),torch.from_numpy(F)


def proposal(carriers,raw,ids,directions):
    v,h=map(array,(carriers,raw));ii=array(ids,np.int32);out=np.empty((len(v),directions))
    library().csn5_proposal(len(v),len(h),directions,ii.ctypes.data_as(I),ptr(v),ptr(h),ptr(out))
    return torch.from_numpy(out)
