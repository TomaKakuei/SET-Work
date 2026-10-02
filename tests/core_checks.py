import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import torch
from setsunet_csn.core import measured_base,completion,minimax,orth
from setsunet_csn.model import SchurProposalNet

class Matrix:
    def __init__(self,B): self.B=B
    def __call__(self,v): return self.B@v
    def columns(self,V): return self.B@V

def run():
    torch.set_num_threads(1)
    torch.manual_seed(81)
    worst_gap=worst_lock=0.
    for _ in range(60):
        n=24
        T,S=torch.randn(n,n,dtype=torch.float64),torch.randn(n,n,dtype=torch.float64)
        A,B=T.T@T+.3*torch.eye(n),S.T@S+.3*torch.eye(n)
        ga,gb=torch.randn(2,n,dtype=torch.float64)
        oa,ob=Matrix(A),Matrix(B)
        base=measured_base(ga,gb,oa,ob)
        V=torch.randn(n,3,dtype=torch.float64)
        p,gain,d=completion(base,ga,gb,oa,ob,V)
        for H,g,change in zip((A,B),(ga,gb),d["changes"]):
            f=lambda v:g@v+.5*v@H@v
            assert abs(float(f(p)-f(base["p0"]))-change)<1e-8
            assert change<=-float(gain)+1e-8
        assert d["lock_error"]<1e-8 and abs(d["gap"])<1e-8
        worst_gap=max(worst_gap,abs(d["gap"]))
        worst_lock=max(worst_lock,d["lock_error"])
        p2,_,_=completion(base,ga,gb,oa,ob,V@torch.tensor([[2.,1.,0.],[0.,3.,1.],[1.,0.,2.]],dtype=V.dtype))
        assert torch.allclose(p,p2,atol=1e-8,rtol=1e-8)
    H=torch.tensor([[1.,.9],[.9,1.]],dtype=torch.float64)
    g=torch.tensor([-1.,0.],dtype=torch.float64)
    base=measured_base(g,g,Matrix(H),Matrix(H),k=1)
    p,_,_=completion(base,g,g,Matrix(H),Matrix(H),torch.eye(2,dtype=g.dtype))
    assert torch.allclose(p,torch.linalg.solve(H,-g),atol=1e-9)
    a,t,_,gap,_=minimax(torch.eye(2),torch.eye(2),torch.tensor([2.,1.]),torch.tensor([-1.,1.]))
    assert abs(t-1/3)<1e-6 and abs(float(a[0]))<1e-6
    model=SchurProposalNet().double()
    n=24
    T=torch.randn(n,n,dtype=torch.float64)
    A=T.T@T+torch.eye(n)
    ga=torch.randn(n,dtype=torch.float64)
    gb=ga+.2*torch.randn(n,dtype=torch.float64)
    base=measured_base(ga,gb,Matrix(A),Matrix(A))
    ids=torch.arange(n)//3
    V=model(base,ga,gb,ids)
    _,gain,_=completion(base,ga,gb,Matrix(A),Matrix(A),V)
    (-gain).backward()
    gradient_norm=sum(float(p.grad.square().sum()) for p in model.parameters())**.5
    assert gradient_norm>1e-7
    rotation=torch.block_diag(*[torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))[0] for _ in range(8)])
    rt_base=measured_base(rotation@ga,rotation@gb,Matrix(rotation@A@rotation.T),Matrix(rotation@A@rotation.T))
    actual=model(rt_base,rotation@ga,rotation@gb,ids)
    assert torch.allclose(actual,rotation@V,atol=1e-9)
    permutation=torch.randperm(n)
    permuted={k:(v[permutation] if k in ("g","Q","Ya","Yb","Y","p0","e") else v) for k,v in base.items()}
    assert torch.allclose(model(permuted,ga[permutation],gb[permutation],ids[permutation]),V[permutation],atol=1e-9)
    return dict(random_spd_cases=60,max_duality_gap=worst_gap,max_relative_locked_residual=worst_lock,
                neural_gradient_norm=gradient_norm,full_space_newton=True,conflicting_views=True,
                invertible_basis_invariance=True,block_rotation_equivariance=True,coordinate_permutation=True,
                parameters=model.parameter_count)
if __name__=="__main__":
    import json
    print(json.dumps(run(),indent=2))

