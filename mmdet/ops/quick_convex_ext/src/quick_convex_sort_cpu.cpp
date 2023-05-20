#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <torch/extension.h>

#include "quick_convex_sort_utils.h"

template <typename scalar_t>
at::Tensor quick_convex_sort_cpu_kernel(
    at::Tensor & pts, const bool inplace) {

    auto nbs = pts.size(0);
    auto npts = int(pts.size(1) / 2);
    if (nbs == 0) {
        return at::empty({nbs, npts}, pts.options().dtype(at::kLong));
    }

    at::Tensor convex_index_t = at::full({nbs, npts}, -1, pts.options().dtype(at::kLong));
    at::Tensor deque_t = at::full({nbs, (2 * npts + 1) * 3}, -1, pts.options());

    int64_t* convex_index = convex_index_t.data_ptr<int64_t>();
    scalar_t* data = pts.data_ptr<scalar_t>(); 
    scalar_t* deque = deque_t.data_ptr<scalar_t>();
    for (auto i = 0; i < nbs; i++) {
      scalar_t* sub_data = data + i * npts * 2;
      scalar_t* sub_deque = deque + i * (npts * 2 + 1) * 3;
      int64_t* sub_index = convex_index + i * npts;
      simple_hull_2d<scalar_t>(sub_data, sub_deque, sub_index, npts, inplace);
    }
    return convex_index_t;
}

at::Tensor quick_convex_sort_cpu(
    at::Tensor& pts, const bool inplace) {
  AT_ASSERTM(pts.device().is_cpu(), "pts must be a CPU tensor");
  at::Tensor convex_index;
  AT_DISPATCH_FLOATING_TYPES(pts.scalar_type(), "quick_convex_sort", [&] {
    convex_index = quick_convex_sort_cpu_kernel<scalar_t>(pts, inplace);
  });

  return convex_index;
}
