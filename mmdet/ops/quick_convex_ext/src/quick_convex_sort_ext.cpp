#include <ATen/ATen.h>
#include <torch/extension.h>

#ifdef WITH_CUDA
at::Tensor quick_convex_sort_cuda(at::Tensor& pts, const bool inplace);
#endif
at::Tensor quick_convex_sort_cpu(at::Tensor& pts, const bool inplace);


at::Tensor quick_convex_sort(
    at::Tensor& pts, const bool inplace=false) {
  if (pts.device().is_cuda()) {
#ifdef WITH_CUDA
    return quick_convex_sort_cuda(pts, inplace);
#else
    AT_ERROR("sort_vert is not compiled with GPU support");
#endif
  }
  return quick_convex_sort_cpu(pts, inplace);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quick_convex_sort", &quick_convex_sort, "quickly sort convex points");
}
