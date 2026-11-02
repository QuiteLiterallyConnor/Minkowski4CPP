[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-minkowski4cpp-0db7ed?logo=docker&logoColor=white&style=for-the-badge)](https://hub.docker.com/r/acaicia/minkowski4cpp)

# Minkowski4CPP

**A C++ port of NVIDIA's MinkowskiEngine with full CUDA support for Windows and Linux.**

**Documentation:** [GitHub Pages](https://quiteliterallyconnor.github.io/Minkowski4CPP/)
or browse the [documentation source](docs/index.md) for installation, API guides,
examples, testing, and troubleshooting.

Minkowski4CPP brings the power of sparse tensor operations from [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) into native C++ workflows, with first-class CUDA acceleration on both major platforms, something the original Python-based library does not support.

> ### Gaussian splats are built in
> Minkowski4CPP includes an opt-in native **3D Gaussian** subsystem (`MinkowskiEngine/gaussian`). Load Gaussian-splat `.ply` files (standard 3DGS and SuperSplat/PlayCanvas compressed, detected automatically) into a batched, LibTorch-backed `GaussianTensor`, then compute covariances, filter by opacity or bounds, select, merge, apply rigid transforms, and optimize parameters with Adam. It also **renders and trains**: a differentiable rasterizer (fused CUDA forward/backward kernels, with a LibTorch fallback on CPU), spherical harmonics up to degree 3, a 3DGS-style `GaussianTrainer` with densification, pruning and opacity reset, and PLY saving. Works on CPU and CUDA.
>
> - **Overview and API:** [Gaussian splats](#gaussian-splats)
> - **Usage examples:** [Gaussian examples](#gaussian-examples) and [rendering and training](#gaussian-rendering-and-training)
> - **Measured:** a 139K-Gaussian scene loads in about 17-37 ms, its covariances compute in about 0.3 ms, and a 512x512 render takes about 6 ms (28 ms per train step) on an RTX 3080 (see [Gaussian benchmarks](#gaussian-benchmarks)).

> ### Adaptive (mixed-resolution) voxelization is built in
> Beyond the standard uniform sparse tensor, this library includes **adaptive resolution voxelization**: a single sparse tensor can hold voxels at **different resolutions** (octree-style, up to a configurable depth). Refine only the regions that matter, such as a region of interest or dense or high-variance areas, and keep the rest coarse. Convolution, pooling, interpolation and resampling all work across resolution boundaries, are differentiable, and run on CPU and CUDA. It is opt-in and leaves the uniform `SparseTensor` API unchanged.
>
> - **Overview and API:** [Adaptive (mixed-resolution) voxelization](#adaptive-mixed-resolution-voxelization)
> - **Usage examples:** [Adaptive examples](#adaptive-examples)
> - **Measured benefit:** 41,376 adaptive voxels versus about 35 million for the equivalent uniform grid in the included benchmark.
> - **Sphere volume accuracy benchmark:** [benchmarks/README.md](benchmarks/README.md) (standalone, links against an installed MinkowskiEngine package)

---

## Platform Support

| Platform | CUDA Version | Torch Version | Status |
|---|---|---|---|
| Ubuntu 22.04 | 12.4.0 | 2.5.1 | Verified |
| Ubuntu 24.04 (apt/Docker) | 13.2.1 | 2.15.0 / cu132 | Builder profile |
| Ubuntu 24.04 (apt/Docker, default) | 13.4.3 | 2.15.0 / cu134 | Builder profile |
| Windows | 13.2 | 2.15.0 | Verified |
| Windows | 13.4 | 2.15.0 | Verified |

---

## Prerequisites

### Windows

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 13.2 or 13.4
- [cuDNN 9.27.0](https://developer.nvidia.com/cudnn-downloads)
- [LibTorch](https://pytorch.org/get-started/locally/) matching your CUDA version (pre-built, extract only)
- Visual Studio 2022 with the C++ workload
- [Ninja](https://ninja-build.org/) (extract only)
- [CMake](https://cmake.org/download/) (install)

### Linux

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 13.4 (default), 13.2, or legacy 12.4
- [LibTorch](https://pytorch.org/get-started/locally/) matching CUDA: 2.15.0/cu134 or cu132, or 2.5.1/cu124 (Linux CXX11 ABI)
- `ninja-build` â€” install via your package manager
- `cmake` â€” install via your package manager

---

## Building

### Windows

Run the provided PowerShell build script, passing the CUDA version (`-CudaVersion`: `132` or `134`) and the paths to your CUDA installation, LibTorch directory, and Ninja executable. All paths must be specified explicitly:

```powershell
scripts\build-all-windows.ps1 `
  -CudaVersion [132 | 134] `
  -CudaPath    [PATH TO CUDA] `
  -LibTorchPath [PATH TO TORCH] `
  -NinjaPath   [PATH TO NINJA EXECUTABLE]
```

**Example:**

```powershell
scripts\build-all-windows.ps1 `
  -CudaVersion  134 `
  -CudaPath     "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" `
  -LibTorchPath "E:\Code\libtorch" `
  -NinjaPath    "C:\Program Files\Ninja\ninja.exe"
```

  When switching CUDA versions, add `-Clean` to remove the previous build directory.

  ---

### Linux

Navigate to the project root, create a clean build directory, configure with CMake, and build with Ninja:

```bash
cd /path/to/minkowski4cpp/
mkdir build && cd build

cmake -G Ninja \
  -DCMAKE_PREFIX_PATH=[PATH TO LIBTORCH] \
  -DCMAKE_CUDA_COMPILER=[PATH TO NVCC] \
  ..

ninja -j4
```

**Example:**

```bash
cd /path/to/minkowski4cpp/
mkdir build && cd build

cmake -G Ninja \
  -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  ..

ninja -j4
```

### Linux â€” Debian Package (.deb)

**From Windows** (requires Docker Desktop in Linux containers mode â€” no Linux toolchain needed):

```powershell
.\apt\build-apt.ps1
.\apt\build-apt.ps1 -CudaVersion 132
.\apt\build-apt.ps1 -CudaVersion 124
```

**From Linux** (native, no Docker â€” requires CMake, Ninja, and LibTorch):

```bash
chmod +x apt/build-apt.sh
./apt/build-apt.sh --libtorch /opt/libtorch
```

Both scripts write CUDA-specific packages to the `apt/` directory. The Docker
default is CUDA 13.4 on Ubuntu 24.04 (package version 1.3.0); the native script
detects the installed toolkit (package version 1.0.0). Install on the matching
Ubuntu release with matching CUDA and LibTorch:

```bash
sudo dpkg -i libminkowski-cpp-dev_1.3.0_cu134_amd64.deb
sudo ldconfig
```

See [apt/README.md](apt/README.md) for all parameters and details.
See [docker/README.md](docker/README.md) for container builds with the same profiles.

---

## Usage Examples

### Creating a Network

Define a sparse convolution network by subclassing `torch::nn::Module` and using Minkowski modules. This example creates a simple classifier with two convolution stages, global pooling, and a linear head:

```cpp
#include <torch/torch.h>
#include "MinkowskiEngine/minkowski.hpp"

using namespace minkowski;

struct ExampleNetworkImpl : torch::nn::Module {
    MinkowskiConvolution conv1_conv{nullptr};
    MinkowskiBatchNorm   conv1_bn{nullptr};
    MinkowskiReLU        conv1_relu{nullptr};

    MinkowskiConvolution conv2_conv{nullptr};
    MinkowskiBatchNorm   conv2_bn{nullptr};
    MinkowskiReLU        conv2_relu{nullptr};

    MinkowskiGlobalPooling pooling{nullptr};
    MinkowskiLinear        linear{nullptr};

    ExampleNetworkImpl(int in_feat, int out_feat, int D) {
        // First convolution block: Conv â†’ BN â†’ ReLU
        conv1_conv = register_module("conv1_conv", MinkowskiConvolution(
            /*in_channels=*/in_feat, /*out_channels=*/64,
            /*kernel_size=*/3, /*stride=*/2, /*dilation=*/1,
            /*bias=*/false, /*expand_coordinates=*/false,
            ConvolutionMode::DEFAULT, /*dimension=*/D));
        conv1_bn   = register_module("conv1_bn", MinkowskiBatchNorm(64));
        conv1_relu = register_module("conv1_relu", MinkowskiReLU());

        // Second convolution block: Conv â†’ BN â†’ ReLU
        conv2_conv = register_module("conv2_conv", MinkowskiConvolution(
            /*in_channels=*/64, /*out_channels=*/128,
            /*kernel_size=*/3, /*stride=*/2, /*dilation=*/1,
            /*bias=*/false, /*expand_coordinates=*/false,
            ConvolutionMode::DEFAULT, /*dimension=*/D));
        conv2_bn   = register_module("conv2_bn", MinkowskiBatchNorm(128));
        conv2_relu = register_module("conv2_relu", MinkowskiReLU());

        // Global pooling and linear classifier
        pooling = register_module("pooling", MinkowskiGlobalPooling());
        linear  = register_module("linear", MinkowskiLinear(128, out_feat));
    }

    SparseTensor forward(SparseTensor x) {
        auto out = conv1_relu->forward(conv1_bn->forward(conv1_conv->forward(x)));
        out = conv2_relu->forward(conv2_bn->forward(conv2_conv->forward(out)));
        out = pooling->forward(out);
        return linear->forward(out);
    }
};
TORCH_MODULE(ExampleNetwork);
```

### Forward and Backward Pass

Create a `SparseTensor` from coordinates and features, run a forward pass, and compute the loss:

```cpp
int main() {
    // Choose device and backend
    auto device  = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    auto backend = torch::cuda::is_available()
        ? CoordinateMapBackend::CUDA
        : CoordinateMapBackend::CPU;

    // Instantiate network (in_feat=3, out_feat=5, dimension=2)
    ExampleNetwork net(3, 5, 2);
    net->to(device);

    // Dummy data: 6 points across 2 batches, 2D coordinates
    //   Column layout: [batch_index, x, y]
    auto coords = torch::tensor(
        {{0,0,0}, {0,1,0}, {0,0,1},
         {1,0,0}, {1,1,0}, {1,1,1}}, torch::kInt32).to(device);
    auto feats = torch::randn({6, 3}).to(device);     // 3 input features per point
    auto labels = torch::tensor({0, 2}, torch::kLong).to(device);  // 1 label per batch

    // Build SparseTensor with a shared coordinate manager
    auto mgr = std::make_shared<CoordinateManager>(2, backend);
    SparseTensor input(feats, coords, mgr);

    // Forward
    auto output = net->forward(input);
    auto logits = output.F();  // [batch_size, out_feat]

    // Loss and backward
    auto loss = torch::nn::functional::cross_entropy(logits, labels);
    loss.backward();

    std::cout << "Loss: " << loss.item<float>() << "\n";
    return 0;
}
```

### Advanced Example: 3D UNet with Residual Blocks and Skip Connections

A more realistic architecture for dense prediction on 3D point clouds, e.g. semantic segmentation. This UNet uses strided convolutions to downsample, transposed convolutions to upsample, residual blocks at each level, and skip connections between encoder and decoder:

```cpp
// Reusable Conv â†’ BN â†’ ReLU block
struct ConvBNReLUImpl : torch::nn::Module {
    MinkowskiConvolution conv{nullptr};
    MinkowskiBatchNorm   bn{nullptr};
    MinkowskiReLU        relu{nullptr};

    ConvBNReLUImpl(int c_in, int c_out, int D,
                   int ks = 3, int stride = 1, int dilation = 1) {
        conv = register_module("conv", MinkowskiConvolution(
            c_in, c_out, ks, stride, dilation,
            false, false, ConvolutionMode::DEFAULT, D));
        bn   = register_module("bn", MinkowskiBatchNorm(c_out));
        relu = register_module("relu", MinkowskiReLU());
    }

    SparseTensor forward(SparseTensor x) {
        return relu->forward(bn->forward(conv->forward(x)));
    }
};
TORCH_MODULE(ConvBNReLU);

// Residual block: two 3Ã—3 convolutions with a skip connection
struct ResBlockImpl : torch::nn::Module {
    ConvBNReLU         conv1{nullptr};
    MinkowskiConvolution conv2{nullptr};
    MinkowskiBatchNorm bn2{nullptr};
    MinkowskiReLU      relu{nullptr};

    ResBlockImpl(int c, int D) {
        conv1 = register_module("conv1", ConvBNReLU(c, c, D));
        conv2 = register_module("conv2", MinkowskiConvolution(
            c, c, 3, 1, 1, false, false, ConvolutionMode::DEFAULT, D));
        bn2   = register_module("bn2", MinkowskiBatchNorm(c));
        relu  = register_module("relu", MinkowskiReLU());
    }

    SparseTensor forward(SparseTensor x) {
        auto out = bn2->forward(conv2->forward(conv1->forward(x)));
        return relu->forward(out + x);  // residual addition
    }
};
TORCH_MODULE(ResBlock);

// 3D UNet for semantic segmentation
struct UNet3DImpl : torch::nn::Module {
    // Encoder
    ConvBNReLU                stem{nullptr};
    ResBlock                  enc1{nullptr};
    MinkowskiConvolution      down1{nullptr};
    ResBlock                  enc2{nullptr};
    MinkowskiConvolution      down2{nullptr};
    ResBlock                  bottleneck{nullptr};

    // Decoder
    MinkowskiConvolutionTranspose up1{nullptr};
    ResBlock                      dec1{nullptr};
    MinkowskiConvolutionTranspose up2{nullptr};
    ResBlock                      dec2{nullptr};

    // Segmentation head
    MinkowskiConvolution head{nullptr};

    UNet3DImpl(int in_channels, int num_classes, int c_mid = 32) {
        int D = 3;
        int c_bot = c_mid * 4;

        // Encoder
        stem  = register_module("stem", ConvBNReLU(in_channels, c_mid, D));
        enc1  = register_module("enc1", ResBlock(c_mid, D));
        down1 = register_module("down1", MinkowskiConvolution(
            c_mid, c_mid * 2, 2, 2, 1, false, false,
            ConvolutionMode::DEFAULT, D));
        enc2  = register_module("enc2", ResBlock(c_mid * 2, D));
        down2 = register_module("down2", MinkowskiConvolution(
            c_mid * 2, c_bot, 2, 2, 1, false, false,
            ConvolutionMode::DEFAULT, D));
        bottleneck = register_module("bottleneck", ResBlock(c_bot, D));

        // Decoder
        up1  = register_module("up1", MinkowskiConvolutionTranspose(
            c_bot, c_mid * 2, 2, 2, 1, false, false,
            ConvolutionMode::DEFAULT, D));
        dec1 = register_module("dec1", ResBlock(c_mid * 2, D));
        up2  = register_module("up2", MinkowskiConvolutionTranspose(
            c_mid * 2, c_mid, 2, 2, 1, false, false,
            ConvolutionMode::DEFAULT, D));
        dec2 = register_module("dec2", ResBlock(c_mid, D));

        // Per-point classification head (1Ã—1 conv)
        head = register_module("head", MinkowskiConvolution(
            c_mid, num_classes, 1, 1, 1, true, false,
            ConvolutionMode::DEFAULT, D));
    }

    SparseTensor forward(SparseTensor x) {
        // Encoder
        auto e0 = stem->forward(x);
        auto e1 = enc1->forward(e0);
        auto e2 = enc2->forward(down1->forward(e1));
        auto e3 = bottleneck->forward(down2->forward(e2));

        // Decoder with skip connections
        auto d1 = dec1->forward(up1->forward(e3) + e2);
        auto d2 = dec2->forward(up2->forward(d1) + e1);

        return head->forward(d2);
    }
};
TORCH_MODULE(UNet3D);
```

Training loop for the segmentation network:

```cpp
int main() {
    auto device  = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    auto backend = torch::cuda::is_available()
        ? CoordinateMapBackend::CUDA : CoordinateMapBackend::CPU;

    int num_classes = 13;
    UNet3D model(/*in_channels=*/3, num_classes, /*c_mid=*/32);
    model->to(device);

    auto optimizer = torch::optim::Adam(
        model->parameters(), torch::optim::AdamOptions(1e-3));

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->train();

        // --- Replace with your data loader ---
        // coords: [N, 4] int32 with batch index in column 0
        // feats:  [N, 3] float32 (e.g. RGB or normals)
        // labels: [N] int64 per-point class labels
        auto coords = torch::tensor(
            {{0,0,0,0},{0,1,0,0},{0,0,1,0},{0,1,1,0},{0,0,0,1},
             {1,0,0,0},{1,1,0,0},{1,0,1,0}}, torch::kInt32).to(device);
        auto feats  = torch::randn({8, 3}).to(device);
        auto labels = torch::randint(0, num_classes, {8}, torch::kLong).to(device);

        // Create a fresh coordinate manager each forward pass
        auto mgr = std::make_shared<CoordinateManager>(3, backend);
        SparseTensor input(feats, coords, mgr);

        // Forward
        auto output = model->forward(input);
        auto logits = output.F();  // [N, num_classes]

        // Per-point cross-entropy loss
        auto loss = torch::nn::functional::cross_entropy(logits, labels);

        // Backward and step
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        std::cout << "Epoch " << epoch << " loss: " << loss.item<float>() << "\n";
    }

    // Save trained model
    save_model(model, "unet3d_segmentation.pt");
    return 0;
}
```

For a complete real-world training example with data loading, checkpointing, and evaluation, see [tests/train.cpp](tests/train.cpp).
Run it by passing in the data directory found in the tests directory as --data_dir

---

## Adaptive (mixed-resolution) voxelization

`MinkowskiEngine/adaptive/` adds an opt-in sparse tensor whose leaves live at different resolutions
(quadtree/octree-style), alongside the unchanged uniform `SparseTensor`. Include
`MinkowskiEngine/adaptive/adaptive.hpp`; everything is in `minkowski::adaptive`.

- **Hierarchy**: level 0 has voxel size `base_voxel_size`; each level halves it, down to
  `min_voxel_size` (`base / min == 2^max_depth`). A leaf is `(level, integer coords)`; a point belongs
  to the half-open box `[c*s, (c+1)*s)`.
- **Topology edits**: `refine(region, target_level, RefinePolicy::Replicate|Split)`,
  `coarsen(region, target_level, Reduction::Mean|Sum|Max)`, `refine_if`, `refine_by_density`,
  `refine_by_variance`, `revoxelize_from_points`. Topologies are immutable and versioned.
- **Convolution** (`adaptive_convolution`, plus transpose, pooling, global average pool and
  `interpolate`): the kernel is applied at each output leaf's own scale, sampling the neighbor at
  `center + offset * output_voxel_size`. Modes: `CenterSampled`, `VolumeWeighted`,
  `OverlapWeighted` (weights = overlap volume / output volume). Kernel maps are cached per topology.
- **Differentiable**: feature ops are built from torch ops, so autograd works; topology decisions are not
  differentiable. The code runs on CPU and on CUDA (tensor ops run on the feature tensor's device;
  hierarchy and kernel-map construction run on the host). There is no ROCm backend in this repo.

```cpp
using namespace minkowski::adaptive;

AdaptiveVoxelConfig cfg;
cfg.base_voxel_size = {8.0, 8.0, 8.0};
cfg.min_voxel_size  = {0.5, 0.5, 0.5};
cfg.max_depth = 4;

auto t = AdaptiveSparseTensor::from_points(points, features, cfg);   // [N,3] double, [N,C]
t.refine(AxisAlignedRegion{{0, 0, 0}, {16, 16, 16}}, /*target_level=*/4);  // fine only in the ROI

AdaptiveKernelConfig kc;                       // 3x3x3, CenterSampled
auto out = adaptive_convolution(t, weight, kc);  // weight: [27, Cin, Cout]
```

### Adaptive examples

All snippets assume `using namespace minkowski::adaptive;` and the `cfg` defined above
(`base_voxel_size` 8, `max_depth` 4, so level 4 voxels are 0.5 wide).

**1. Inspect a tensor**

```cpp
auto t = AdaptiveSparseTensor::from_points(points, features, cfg);
std::cout << t.size() << " leaves, " << t.channels() << " channels\n";
auto coords = t.coordinates();     // [N,3] int64, integer cell index at each leaf's own level
auto levels = t.levels();          // [N]  int32,  0 = coarsest
auto centers = t.centers();        // [N,3] double, physical leaf centers
auto counts = t.counts_by_level(); // max_depth + 1 entries
```

**2. Refine and coarsen a region.** `Replicate` copies the parent feature to all 2^D children;
`Split` divides it by 2^D so that summed features are conserved.

```cpp
AxisAlignedRegion roi{{0, 0, 0}, {16, 16, 16}};   // half-open [lo, hi)

t.refine(roi, 4);                           // leaves in the ROI go to level 4
t.refine(roi, 4, RefinePolicy::Split);      // same, conserving sums of extensive features
t.coarsen(roi, 2, Reduction::Mean);         // merge back up to level 2 (Mean, Sum or Max)
t.coarsen(AxisAlignedRegion::everything(3), 0, Reduction::Sum);
```

A parent is only merged if its whole box lies inside the region. Reductions use occupied
children only.

**3. Point-driven refinement.** These need a tensor made with `from_points`, which keeps the
source points.

```cpp
t.refine_by_density(/*points_per_leaf_threshold=*/32);   // split crowded leaves
t.refine_by_variance(/*feature_variance_threshold=*/0.5);
t.revoxelize_from_points();                              // recompute leaf features from points

// Arbitrary rule: refine leaves whose center is near the origin
t.refine_if([](int64_t, const std::array<double, 4>& lo, const std::array<double, 4>& hi,
               uint32_t level) {
    double c = 0.5 * (lo[0] + hi[0]);
    return level < 3 && std::abs(c) < 10.0;
});
```

**4. Build from explicit voxels.** Each row is a leaf at its own level; leaves must not overlap.

```cpp
auto coords = torch::tensor({{0, 0, 0}, {2, 0, 0}, {2, 1, 0}}, torch::kInt64);
auto levels = torch::tensor({0, 1, 1}, torch::kInt32);   // level 1 cells are half the size
auto feats  = torch::randn({3, 8});
auto v = AdaptiveSparseTensor::from_voxels(coords, levels, feats, cfg);
```

**5. Convolution.** The weight is `[K^D, Cin, Cout]` (27 for the default 3x3x3 kernel, first axis
varying slowest). The kernel is applied at each output leaf's own scale, so mixed-resolution
neighbors are handled automatically.

```cpp
auto weight = torch::randn({27, 8, 16}).set_requires_grad(true);

AdaptiveKernelConfig kc;                             // kernel_size = 3
kc.mode = AdaptiveConvMode::OverlapWeighted;         // or CenterSampled / VolumeWeighted
auto out = adaptive_convolution(t, weight, kc);      // same leaves as t, 16 channels

out.features().sum().backward();                     // gradients reach weight and features
```

Strided convolution and its transpose:

```cpp
AdaptiveKernelConfig down;
down.output_level_shift = -1;                        // outputs are one level coarser
auto coarse = adaptive_convolution(t, weight, down);

auto w_up = torch::randn({27, 16, 8});
auto restored = adaptive_convolution_transpose(coarse, w_up, down, /*reference=*/t);
// restored lives on t's leaves
```

Kernel maps are cached per topology, so repeated convolutions on the same tensor only build the
map once. `KernelMapCache::instance().stats()` reports hits and misses, and `.clear()` resets it.

**6. Pooling, interpolation and resampling**

```cpp
auto global = adaptive_global_avg_pool(t);                 // [C], volume-weighted
auto level1 = adaptive_pool(t, /*target_level=*/1, Reduction::Max);

auto query = torch::tensor({{1.0, 2.0, 3.0}, {40.0, 5.0, 5.0}}, torch::kFloat64);
auto nearest = interpolate(t, query);                      // value of the containing leaf
auto smooth  = interpolate(t, query, InterpMode::Multilinear);
// query points outside every leaf give zeros

auto uniform = t.resample_uniform(2);                      // everything at level 2
auto aligned = other.resample_to(t);                       // put `other` on t's leaves
auto sum = t + aligned;                                    // + needs identical topology
```

`resample_uniform` can be very large for sparse mixed-resolution data; the benchmark skips it above
5M voxels.

**7. Use a GPU.** Feature tensors can live on CUDA; the hierarchy and kernel maps stay on the host.

```cpp
auto gt = t.to(torch::kCUDA);
auto gout = adaptive_convolution(gt, weight.to(torch::kCUDA), kc);
auto cpu_out = gout.to(torch::kCPU);
```

**8. Complete example: refine around a region of interest and run a small network**

```cpp
#include "MinkowskiEngine/adaptive/adaptive.hpp"
using namespace minkowski::adaptive;

int main() {
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {8.0, 8.0, 8.0};
    cfg.min_voxel_size  = {0.5, 0.5, 0.5};
    cfg.max_depth = 4;

    auto points = torch::rand({20000, 3}, torch::kFloat64) * 64.0;
    auto feats  = torch::randn({20000, 4}, torch::kFloat64);

    auto t = AdaptiveSparseTensor::from_points(points, feats, cfg);
    t.refine(AxisAlignedRegion{{0, 0, 0}, {16, 16, 16}}, 4);   // fine detail only in the ROI

    auto w1 = torch::randn({27, 4, 16}, torch::kFloat64).mul_(0.1).set_requires_grad(true);
    auto w2 = torch::randn({27, 16, 2}, torch::kFloat64).mul_(0.1).set_requires_grad(true);

    AdaptiveKernelConfig kc;
    auto h = adaptive_convolution(t, w1, kc);
    h = h.with_features(torch::relu(h.features()));
    auto y = adaptive_convolution(h, w2, kc);

    auto logits = adaptive_global_avg_pool(y);        // [2]
    logits.sum().backward();
    std::cout << t.size() << " leaves, logits " << logits << "\n";
}
```

Link against `minkowski_cpp` like any other target in this repo; `tests/adaptive/` contains
further runnable usage of every call shown here, and `adaptive_bench` shows timing.

Tests: `adaptive_tests` (66 tests, CPU; CUDA agreement is checked when CUDA is available),
`adaptive_bench`, and the standalone `sphere` JSON exporter, built by default
(`-DMINK_BUILD_ADAPTIVE_TESTS=OFF` to skip):

```
cmake --build build --target adaptive_tests adaptive_bench sphere
```

Run `build/sphere` to print a JSON document containing each voxel's grid coordinates,
refinement level, world-space bounds, center, size, and occupancy.

The standalone sphere benchmark compares uniform grids against an adaptive octree using
volume error, voxel counts, boundary-cell counts, and construction/evaluation timings. See
[benchmarks/README.md](benchmarks/README.md) for setup and run instructions.

Measured with `adaptive_bench` (20k points, depth 4, CPU, Windows/CUDA 13.2 build): 41,376 adaptive
voxels versus about 35 million for the equivalent uniform grid at the finest resolution;
kernel-map build 221 ms, forward 14 ms, forward+backward 50 ms.

Limitations: no uniform fast path (the legacy convolution is not used for single-level tensors),
`output_level_shift > 0` is unsupported, kernel-map construction is host-side, and
`OverlapWeighted` cost grows with the level gap between neighbors. The all-ones-kernel comparison with
the legacy convolution agrees only in row count and total sum; per-row agreement was not established.

---

## Gaussian splats

Everything lives in `minkowski::gaussian`; include `MinkowskiEngine/minkowski.hpp` or the individual headers in `MinkowskiEngine/gaussian/`. Enabled by default (`-DMINKOWSKI_ENABLE_GAUSSIAN=OFF` to disable).

| Piece | What it does |
|---|---|
| `Gaussian3D<T>` | Host value type: mean, orientation (w,x,y,z), scale, opacity; covariance, density, Mahalanobis distance, overlap integral, sampling, bounds. |
| `GaussianTensor` | Batched storage: positions `[N,3]`, rotations `[N,4]`, scales `[N,3]`, opacities `[N,1]`, colors `[N,3]`, optional features and velocities. Float32/float64, CPU or CUDA. |
| `load_gaussian_ply` | Loads binary little-endian PLY, either standard 3DGS (`x,y,z,f_dc_*,opacity,scale_*,rot_*`) or SuperSplat compressed. Applies sigmoid to opacity, exp to scales, `0.5 + C0*f_dc` to colors and normalizes quaternions. Spherical-harmonic rest coefficients are skipped by `load_gaussian_ply`; use `load_gaussian_model_ply` to keep them. Throws `std::runtime_error` on missing or malformed files. |
| `GaussianOptimizer` | LibTorch Adam over the Gaussian fields, with RGB/depth reconstruction losses, opacity and scale regularizers, and constraint projection (scale and opacity clamps). |
| CUDA kernels | Used for covariance and rigid transforms on CUDA tensors when no gradients are required. |
| `GaussianModel` | Trainable raw parameters (xyz, `f_dc`, `f_rest`, logit opacity, log scale, quaternion) with activated views, SH degree 0-3, `from_points`, `select`, `clone`, `to`. |
| `render` / `render_gaussians` | Differentiable tile-based rasterizer (EWA projection, depth sort, front-to-back alpha compositing) returning image, depth, alpha, screen-space means and radii. Gradients reach every parameter. On CUDA tensors it uses fused forward and backward CUDA kernels (16x16 tiles, shared-memory batching, warp-reduced gradient accumulation); `RenderOptions::use_cuda_kernels = false` falls back to the pure LibTorch path, which is also used on CPU. |
| `GaussianTrainer` | 3DGS-style training: L1 + SSIM loss, per-parameter Adam, position LR decay, SH degree growth, gradient-driven densification (clone/split), pruning and periodic opacity reset. |
| `save_gaussian_ply` / `load_gaussian_model_ply` | Write and read standard 3DGS PLYs including SH (round-trips raw parameters). |

Not implemented yet: voxel bridges and adaptive indexing. Projection and SH evaluation still run as LibTorch ops (differentiated by autograd); only tile compositing is fused. The compressed-PLY decode, SH bands included, is verified against the uncompressed export of the same scene (`test_gaussian_compressed_ply.cpp`). See the [Gaussian notes](MinkowskiEngine/gaussian/README.md).

### Gaussian rendering and training

Train a PLY (a camera needs a `[4,4]` float world-to-camera matrix in the OpenCV convention):

```cpp
GaussianModel model = load_gaussian_model_ply("scene.ply").to(torch::Device(torch::kCUDA, 0));

Camera camera = Camera::look_at({0, 0, -4}, {0, 0, 0}, {0, -1, 0}, 512, 512, /*fov_x=*/1.0);
at::Tensor image = render(model, camera).image;                  // [H,W,3]

GaussianTrainerConfig config;
config.scene_extent = 4.0;                                       // scales position LR and size thresholds
GaussianTrainer trainer(model, config);
for (const auto &[camera, target] : views)                        // target: [H,W,3] float in [0,1]
  trainer.step(camera, target);                                  // loss, backward, Adam, densify/prune

save_gaussian_ply(model, "trained.ply");
```

Initialize from a point cloud with `GaussianModel::from_points(points, colors)`.

### Gaussian examples

```cpp
#include "MinkowskiEngine/minkowski.hpp"
using namespace minkowski::gaussian;
```

Load a PLY (format detected from the header):

```cpp
GaussianTensor scene = load_gaussian_ply("scene.ply");          // or scene_compressed.ply
std::cout << scene.size() << " gaussians\n";
```

Covariances, filtering and selection:

```cpp
at::Tensor cov = scene.covariances();                            // [N,3,3]
GaussianTensor visible = scene.filter_opacity(0.5);              // opacity >= 0.5

at::Tensor lo = std::get<0>(scene.positions().min(0));
at::Tensor hi = std::get<0>(scene.positions().max(0));
GaussianTensor half = scene.filter_bounds(lo, 0.5 * (lo + hi));  // axis-aligned box

GaussianTensor even = scene.select(at::arange(0, scene.size(), 2, at::kLong));
GaussianTensor rest = scene.remove(at::arange(0, scene.size(), 2, at::kLong));
GaussianTensor merged = even.append(rest);
```

Rigid transform (quaternion w,x,y,z and translation; shared or per-Gaussian):

```cpp
at::Tensor q = at::tensor({0.9238795f, 0.f, 0.3826834f, 0.f});   // 45 deg about y
at::Tensor t = at::tensor({0.5f, 1.0f, -1.0f});
GaussianTensor moved = scene.transform(q, t);
```

CUDA (CUDA builds only):

```cpp
if (torch::cuda::is_available()) {
  GaussianTensor gpu = scene.to(torch::kCUDA);
  at::Tensor cov_gpu = gpu.covariances();                        // CUDA kernel
  GaussianTensor moved_gpu = gpu.transform(q.cuda(), t.cuda());  // CUDA kernel
}
```

Optimization (the optimizer takes ownership of gradients on the tensor you pass; it must outlive the optimizer):

```cpp
GaussianTensor work = scene.clone();
GaussianOptimizer optimizer(work, /*learning_rate=*/1e-2);
at::Tensor target = at::zeros_like(work.colors());
for (int i = 0; i < 100; ++i) {
  at::Tensor loss = GaussianOptimizer::rgb_reconstruction_loss(work.colors(), target);
  optimizer.step(loss);            // zero_grad, backward, Adam step, project constraints
}
```

Host-side Gaussian math with `Gaussian3D<double>` (`Gaussian3Dd`) is covered in `tests/gaussian/test_gaussian3d.cpp`.

### Gaussian tests

```bat
cmake --build build --target gaussian_tests
build\gaussian_tests.exe
```

`gaussian_tests` (35 tests) covers the value math, tensor operations, optimizer, CUDA parity, PLY loading/saving, SH, the differentiable renderer (including a finite-difference gradient check) and the trainer. `tests/gaussian/test_gaussian_ply.cpp` loads `tests/data/test.ply` (uncompressed) and `tests/data/test_compressed.ply`; set the `MINK_TEST_DATA_DIR` environment variable to use another data directory. Tests skip if the files are missing.

### Gaussian benchmarks

```bat
cmake --build build --target gaussian_bench gaussian_ply_bench
build\gaussian_bench.exe --count 100000 --output benchmarks\gaussian_benchmark_results.json
build\gaussian_ply_bench.exe --input tests\data\test.ply --input tests\data\test_compressed.ply --iterations 10
```

`gaussian_bench` times covariance, transform and optimization on synthetic data (CPU and CUDA). `gaussian_ply_bench` times cold and warm PLY loading (file size, Gaussians/s, MB/s) and then covariance, transforms, filters, an optimizer step and host-to-device transfer on the first input. Both write JSON. Measured on an RTX 3080 build with the 139,410-Gaussian scene: warm load about 37 ms (uncompressed, 32.9 MB) and 17 ms (compressed, 8.5 MB); covariances 7.8 ms on CPU versus 0.3 ms on CUDA. Disable with `-DMINK_BUILD_GAUSSIAN_BENCHMARKS=OFF`. `gaussian_ply_bench` also times model loading with SH, forward rendering and a full training step: on the RTX 3080 the full 139,410-Gaussian scene at 512x512 takes about 6 ms per render and 28 ms per train step with the CUDA rasterizer (versus 772 ms and 3.2 s on the LibTorch fallback, about 128x and 113x slower); 5,000 Gaussians at 128x128 on CPU take about 48 ms and 197 ms.

---

## About

MinkowskiEngine is a library for sparse tensor operations commonly used in 3D deep learning (e.g., point cloud processing). Minkowski4CPP ports this functionality to C++, enabling integration into native applications and pipelines that cannot depend on a Python runtime, while preserving GPU acceleration via CUDA.

---

Credit to:

    4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19, [pdf]

@inproceedings{choy20194d,
  title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3075--3084},
  year={2019}
}


## License

This project is a derivative work of [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) by NVIDIA. Please refer to the original project's license for terms governing the underlying implementation.
