import torch
from torch.autograd import Function
from . import quick_convex_ext

class QuickConvexSortFunction(Function):

    @staticmethod
    def forward(ctx, pts, inplace):
        idx = quick_convex_ext.quick_convex_sort(pts, inplace)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, grad_output):
        return ()

quick_convex_sort_func = QuickConvexSortFunction.apply

def quick_convex_sort(pts: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return quick_convex_sort_func(pts, inplace)
