"""Shared analytic HPatches factors, preserving the frozen two-view objective.

This is project interface engineering, available to every Stage9 optimizer.
No learned proposal is used. Frozen Stage2 keeps its original AD execution path.
"""
import inspect
import numpy as np
import torch


class HPatchesFactors:
    evaluator_backend='project_NumPy_analytic_HPatch_factors'
    evaluates_in_c=False
    def __init__(self,problem):
        raw=inspect.getclosurevars(problem.raw_function).nonlocals
        project=inspect.getclosurevars(raw['project']).nonlocals
        decode=inspect.getclosurevars(project['decode']).nonlocals
        self.images=[im.numpy() for im in raw['target_levels']]
        self.references=np.stack([ref.numpy() for ref in raw['ref_levels']])
        self.weights=np.stack([w.numpy() for w in raw['view_sqrt_weights']])
        H0=decode['H0'].numpy();h,w=decode['target_shape']
        norm=np.array([[2/(w-1),0,-1],[0,2/(h-1),-1],[0,0,1]],dtype=np.float64)
        inverse=np.linalg.inv(norm)
        from neural_grey_v2.hpatches_homography import HOMOGRAPHY_PARAMETER_SCALES
        scales=HOMOGRAPHY_PARAMETER_SCALES.double().numpy();matrices=[]
        for i in range(8):
            E=np.zeros((3,3));E.flat[i]=scales[i];matrices.append(inverse@E@norm@H0)
        self.matrices=np.stack(matrices);self.H0=H0
        source=project['source'].numpy();S=np.column_stack((source,np.ones(len(source))))
        self.base=S@H0.T;self.directions=np.einsum('kij,sj->sik',self.matrices,S)
        self.dc=self.matrices[:,2,2];self.calls=[0,0,0]
    def components(self,x,derivative):
        x=x.detach().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
        z=self.base+self.directions@x;c=max(float(self.H0[2,2]+self.dc@x),1e-8)
        dc=self.dc if self.H0[2,2]+self.dc@x>=1e-8 else self.dc*0
        projected=z/c;den=projected[:,2];safe=np.where(den>=0,np.maximum(den,1e-6),np.minimum(den,-1e-6))
        xy=projected[:,:2]/safe[:,None]
        if derivative:
            dp=self.directions/c-z[:,:,None]*dc[None,None,:]/(c*c)
            dd=dp[:,2,:]*(np.abs(den)>=1e-6)[:,None]
            dxy=(dp[:,:2,:]*safe[:,None,None]-projected[:,:2,None]*dd[:,None,:])/safe[:,None,None]**2
        errors=[];jac=[]
        for im,ref in zip(self.images,self.references):
            h,w=im.shape;xx=np.clip(xy[:,0],0,w-1);yy=np.clip(xy[:,1],0,h-1)
            x0=np.floor(xx).astype(np.int64);y0=np.floor(yy).astype(np.int64);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
            a=xx-x0;b=yy-y0;q00=im[y0,x0];q10=im[y0,x1];q01=im[y1,x0];q11=im[y1,x1]
            values=(1-a)*(1-b)*q00+a*(1-b)*q10+(1-a)*b*q01+a*b*q11
            errors.append(values-ref)
            if derivative:
                gx=((1-b)*(q10-q00)+b*(q11-q01))*((xy[:,0]>=0)&(xy[:,0]<=w-1))
                gy=((1-a)*(q01-q00)+a*(q11-q10))*((xy[:,1]>=0)&(xy[:,1]<=h-1))
                jac.append(gx[:,None]*dxy[:,0,:]+gy[:,None]*dxy[:,1,:])
        return np.stack(errors),np.stack(jac) if derivative else None
    def evaluate(self,x,view,mode=2,*,jacobian=False):
        values,J=self.components(x,mode>0 or jacobian);r=(values*self.weights[view]).reshape(-1);self.calls[mode]+=1
        if J is not None:J=(J*self.weights[view,:,:,None]).reshape(-1,8)
        return dict(r=r,w=np.ones_like(r),cost=.5*float(r@r),J=J,g=J.T@r if mode else None,H=J.T@J if mode==2 else None)
    def cost(self,x,view):return torch.tensor(self.evaluate(x,view,0)['cost'],dtype=torch.float64)
    def least_squares(self,x,jacobian=True):
        values,J=self.components(x,jacobian);self.calls[int(jacobian)]+=2
        r=(values[None]*self.weights).reshape(-1)/np.sqrt(2)
        return r,((J[None]*self.weights[:,:,:,None]).reshape(-1,8)/np.sqrt(2) if jacobian else None)
    def objective(self,x,gradient=False):
        r,J=self.least_squares(x,gradient);f=.5*float(r@r)
        return (f,J.T@r) if gradient else f


def prepare(problem):
    if (problem.metadata or {}).get('family')=='hpatches' and not hasattr(problem,'native'):
        problem.native=HPatchesFactors(problem)
    return problem
