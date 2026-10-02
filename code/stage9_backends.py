"""Shared objective/Jacobian adapters, with explicit project ownership."""
import numpy as np
import scipy.sparse as sp
import torch


class TorchFactors:
    evaluator_backend='project_Torch_AD_adapter'
    evaluates_in_c=False
    def __init__(self,problem):self.problem=problem;self.n=problem.initial.numel();self.calls=[0,0,0]
    def tensor(self,x):return torch.as_tensor(x,dtype=torch.float64)
    def raw(self,x,v):return self.problem.raw(x.to(self.problem.initial.dtype),v).to(torch.float64)
    def scalar(self,x,v):return self.problem.cost(x.to(self.problem.initial.dtype),v).to(torch.float64)
    def jacobian(self,fn,x):
        # Reverse AD is inexpensive for scalarized factor tasks; forward AD for
        # tall image residuals avoids a residual-count sized reverse batch.
        count=fn(x).numel()
        return (torch.func.jacfwd(fn)(x) if count>=self.n else torch.func.jacrev(fn)(x)).detach()
    def evaluate(self,x,view,mode=2,*,jacobian=False):
        x=self.tensor(x);r=self.raw(x,view);self.calls[mode]+=1
        cost=self.scalar(x,view).detach();J=H=g=None
        w=torch.broadcast_to(self.problem.weights(x.to(self.problem.initial.dtype),view),r.shape).reshape(-1).double().detach()
        if mode==2 or jacobian:
            J=self.jacobian(lambda z:self.raw(z,view).reshape(-1),x)
            g=J.T@(w*r.reshape(-1))
            if mode==2:H=J.T@(w[:,None]*J)
        elif mode==1:g=torch.func.grad(lambda z:self.scalar(z,view))(x).detach()
        return dict(cost=float(cost),r=r.detach().numpy().reshape(-1),w=w.numpy(),
            J=None if J is None else J.numpy(),H=None if H is None else H.numpy(),g=None if g is None else g.numpy())
    def cost(self,x,v):return torch.tensor(self.evaluate(x,v,0)['cost'],dtype=torch.float64)
    def objective(self,x,gradient=False):
        x=self.tensor(x)
        if gradient:
            grad,value=torch.func.grad_and_value(lambda z:.5*(self.scalar(z,0)+self.scalar(z,1)))(x)
            self.calls[1]+=2
            return float(value.detach()),grad.detach().numpy()
        self.calls[0]+=2
        return float(.5*(self.scalar(x,0)+self.scalar(x,1)))
    def residual(self,x):
        values=[]
        for v in (0,1):
            raw=self.raw(x,v)
            if (self.problem.metadata or {}).get('robust'):
                delta=self.problem.metadata['robust_delta']
                square=raw.square().sum(-1,keepdim=True);t=torch.sqrt(1+square/delta**2)
                w=self.problem.weights(x.to(self.problem.initial.dtype),v).double()
                mask=(w*t).detach()
                raw=raw*torch.sqrt(2*mask/(t+1))
            values.append(raw.reshape(-1)/np.sqrt(2))
        return torch.cat(values)
    def least_squares(self,x,jacobian=True):
        x=self.tensor(x);f=self.residual(x).detach()
        J=self.jacobian(self.residual,x).numpy() if jacobian else None
        self.calls[int(jacobian)]+=2
        return f.numpy(),J


def engine_for(problem):return problem.native if hasattr(problem,'native') else TorchFactors(problem)


class CountedEngine:
    """Counts adapter requests, distinct from actual factor-kernel invocations."""
    def __init__(self,engine):
        self.engine=engine;self.requests=dict(objective=0,gradient=0,residual=0,jacobian=0,normal=0)
        self.evaluator_backend=getattr(engine,'evaluator_backend',type(engine).__name__)
    def objective(self,x,gradient=False):
        self.requests['gradient' if gradient else 'objective']+=1
        return self.engine.objective(x,gradient)
    def least_squares(self,x,jacobian=True):
        self.requests['jacobian' if jacobian else 'residual']+=1
        return self.engine.least_squares(x,jacobian)
    def evaluate(self,x,v,mode=2):
        self.requests['normal' if mode==2 else 'gradient' if mode==1 else 'objective']+=1
        return self.engine.evaluate(x,v,mode)


def dense(J):return J.toarray() if sp.issparse(J) else np.asarray(J)
