#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include "quick_convex_sort_utils.h"

template <typename T>
__global__ void quick_convex_sort_kernel(
    const int nbs, const int npts, T* data, T* deque, int64_t* convex_index, const bool inplace) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nbs) {
    T* sub_data = data + i * npts * 2;
    T* sub_deque = deque + i * (npts * 2 + 1) * 3;
    int64_t* sub_index = convex_index + i * npts;
    simple_hull_2d<T>(sub_data, sub_deque, sub_index, npts, inplace);
  }
}

at::Tensor quick_convex_sort_cuda(
    at::Tensor & pts, const bool inplace) {
  AT_ASSERTM(pts.device().is_cuda(), "pts must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(pts.device());

  int nbs = pts.size(0);
  int npts = int(pts.size(1) / 2);

  at::Tensor convex_index_t = at::full({nbs, npts}, -1, pts.options().dtype(at::kLong));
  if (npts == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return convex_index_t;
  }
  at::Tensor deque_t = at::full({nbs, (2 * npts + 1) * 3}, -1, pts.options());

  dim3 blocks(THCCeilDiv(nbs, 512));
  dim3 threads(512);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(pts.scalar_type(), "quick_convex_sort", [&] {
      quick_convex_sort_kernel<<<blocks, threads, 0, stream>>>(
	  nbs, npts, pts.data_ptr<scalar_t>(), deque_t.data_ptr<scalar_t>(), convex_index_t.data_ptr<int64_t>(), inplace);
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return convex_index_t;
}
