#include <torch/extension.h>

void forward(at::Tensor& Q, at::Tensor& K, at::Tensor& V, at::Tensor& M, at::Tensor& O);
void backward(at::Tensor& Q, at::Tensor& K, at::Tensor& V, at::Tensor& M, at::Tensor& dO,
    at::Tensor& dQ, at::Tensor& dK, at::Tensor& dV, at::Tensor& dM);

// void cuda_fast_block_fw(at::Tensor& Q, at::Tensor& K, at::Tensor& V, at::Tensor& M, at::Tensor& O);
// void cuda_fast_block_bw(at::Tensor& Q, at::Tensor& K, at::Tensor& V, at::Tensor& M, at::Tensor& dO,
//     at::Tensor& dQ, at::Tensor& dK, at::Tensor& dV, at::Tensor& dM);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "forward");
    m.def("backward", &backward, "backward");
}

TORCH_LIBRARY(la, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
