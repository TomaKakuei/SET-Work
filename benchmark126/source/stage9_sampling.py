"""Exact border bilinear sampling with higher-order/forward AD support.

Shared compatibility work belonging to this project, used by every optimizer.
The frozen legacy source is unchanged.
"""
import types,torch
import torch.nn.functional as F


def grid_sample(image,grid,mode='bilinear',padding_mode='border',align_corners=True):
    if mode!='bilinear' or padding_mode!='border' or not align_corners:
        raise ValueError('Stage9 compatibility adapter supports the declared legacy sampling mode only')
    n,c,h,w=image.shape;out_h,out_w=grid.shape[1:3]
    xy=grid.reshape(n,-1,2)
    x=((xy[...,0]+1)*(w-1)/2).clamp(0,w-1);y=((xy[...,1]+1)*(h-1)/2).clamp(0,h-1)
    x0=x.floor().long();y0=y.floor().long();x1=(x0+1).clamp_max(w-1);y1=(y0+1).clamp_max(h-1)
    a=(x-x0)[:,None];b=(y-y0)[:,None];flat=image.reshape(n,c,-1)
    def take(xx,yy):return torch.gather(flat,2,(yy*w+xx)[:,None].expand(-1,c,-1))
    result=(1-a)*(1-b)*take(x0,y0)+a*(1-b)*take(x1,y0)+(1-a)*b*take(x0,y1)+a*b*take(x1,y1)
    return result.reshape(n,c,out_h,out_w)


def prepare(problem):
    if (problem.metadata or {}).get('family')!='hpatches_multiframe':return problem
    seen=set();replacement=types.SimpleNamespace(grid_sample=grid_sample,avg_pool2d=F.avg_pool2d)
    def visit(value):
        if id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,types.FunctionType):
            if value.__code__.co_filename.endswith('hpatches_homography.py') and 'F' in value.__globals__:
                value.__globals__['F']=replacement
            for cell in value.__closure__ or ():
                try:visit(cell.cell_contents)
                except ValueError:pass
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        elif isinstance(value,dict):
            for item in value.values():visit(item)
    for value in (problem.raw_function,problem.cost_function,problem.weight_function,problem.legacy_views):visit(value)
    problem.metadata['sampling_backend']='project exact bilinear compatibility adapter; all methods share'
    return problem
