// Sparse convolution forward + backward timing (CPU or CUDA).
//
//   core_conv_bench [--device cpu|cuda] [--points N] [--extent E] [--batches B] [--cin C]
//                   [--cout C] [--stride S] [--warmup W] [--repeats R] [--double] [--seed S]
//
// Builds one random sparse tensor (unique coordinates in [0, E)^3 per batch), runs W warm-up
// iterations (the first one builds and caches the kernel map), then R timed iterations of
// forward and backward with loss = sum(out * R). Prints "key   value" lines (medians in ms) and
// the per-iteration times.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>
#ifndef CPU_ONLY
#include <torch/cuda.h>
#endif

#include "MinkowskiEngine/minkowski.hpp"

using namespace minkowski;
using clk = std::chrono::steady_clock;

namespace {

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n == 0 ? 0.0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

void sync(bool cuda) {
#ifndef CPU_ONLY
    if (cuda) torch::cuda::synchronize();
#else
    (void)cuda;
#endif
}

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string device = "cpu";
    int64_t points = 50000, extent = 96, batches = 2, cin = 32, cout = 32, stride = 1;
    int warmup = 2, repeats = 7;
    bool use_double = false;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : "0"); };
        if (a == "--device") device = next();
        else if (a == "--points") points = std::stoll(next());
        else if (a == "--extent") extent = std::stoll(next());
        else if (a == "--batches") batches = std::stoll(next());
        else if (a == "--cin") cin = std::stoll(next());
        else if (a == "--cout") cout = std::stoll(next());
        else if (a == "--stride") stride = std::stoll(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--repeats") repeats = std::stoi(next());
        else if (a == "--double") use_double = true;
        else if (a == "--seed") seed = std::stoull(next());
        else {
            std::cerr << "unknown argument " << a << "\n";
            return 2;
        }
    }
    const bool cuda = device == "cuda";
    const auto dev = cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    const auto dtype = use_double ? torch::kFloat64 : torch::kFloat32;

    torch::manual_seed(seed);
    const int64_t cells = extent * extent * extent;
    std::vector<at::Tensor> parts;
    for (int64_t b = 0; b < batches; ++b) {
        auto lin = torch::randperm(cells, torch::kInt64).slice(0, 0, std::min(points, cells));
        auto c = torch::empty({lin.size(0), 4}, torch::kInt64);
        c.select(1, 0).fill_(b);
        c.select(1, 3).copy_(lin.remainder(extent));
        c.select(1, 2).copy_(lin.div(extent, "floor").remainder(extent));
        c.select(1, 1).copy_(lin.div(extent * extent, "floor"));
        parts.push_back(c);
    }
    auto coords = torch::cat(parts).to(torch::kInt32).to(dev);
    auto feats = torch::randn({coords.size(0), cin}, dtype).to(dev).requires_grad_(true);

    const int threads_before = at::get_num_threads();
    SparseTensor x(feats, coords);
    const int threads_after = at::get_num_threads();

    MinkowskiConvolution conv(cin, cout, 3, stride, 1, false, false, ConvolutionMode::DEFAULT, 3);
    conv->to(dev, dtype);

    std::vector<double> fwd, bwd, total;
    for (int it = 0; it < warmup + repeats; ++it) {
        if (feats.grad().defined()) feats.mutable_grad().zero_();
        for (auto& p : conv->parameters())
            if (p.grad().defined()) p.mutable_grad().zero_();
        sync(cuda);
        auto t0 = clk::now();
        auto out = conv->forward(x);
        sync(cuda);
        const double f_ms = ms_since(t0);
        auto t1 = clk::now();
        out.F().sum().backward();
        sync(cuda);
        const double b_ms = ms_since(t1);
        if (it == 0) {
            std::cout << "first_fwd_ms_incl_kernel_map  " << f_ms << "\n";
            std::cout << "out_rows  " << out.F().size(0) << "\n";
        }
        if (it >= warmup) {
            fwd.push_back(f_ms);
            bwd.push_back(b_ms);
            total.push_back(f_ms + b_ms);
        }
    }

    std::cout << "device  " << (cuda ? 1 : 0) << "\n";
    std::cout << "in_rows  " << coords.size(0) << "\n";
    std::cout << "threads_before_tensor  " << threads_before << "\n";
    std::cout << "threads_after_tensor  " << threads_after << "\n";
    std::cout << "fwd_ms  " << median(fwd) << "\n";
    std::cout << "bwd_ms  " << median(bwd) << "\n";
    std::cout << "fwd_bwd_ms  " << median(total) << "\n";
    std::cout << "# per-iteration fwd_bwd_ms:";
    for (double t : total) std::cout << " " << t;
    std::cout << "\n";
    return 0;
}
