---
title: Quick start
description: Build a downstream application and train a small sparse classifier.
permalink: /quickstart/
---

{% raw %}
## Download prebuilt binaries

Download binaries from the
[GitHub Releases page](https://github.com/QuiteLiterallyConnor/Minkowski4CPP/releases).
Choose the release asset matching your operating system and CUDA version,
and read that release's installation instructions. Install the matching
LibTorch distribution separately unless the release explicitly bundles it.

If no asset matches your platform or toolchain, follow the
[source installation guide](installation.md) or use the
[Docker and Debian builders](builders.md).

## Include and link conventions

Inside the repository, examples include `"MinkowskiEngine/minkowski.hpp"` and
link `minkowski_cpp`. After installation, use **`<minkowski.hpp>`** and link
**`MinkowskiEngine::minkowski_cpp`**; exported include directories already
point into `include/minkowski`.

Adaptive APIs require `<adaptive/adaptive.hpp>` separately. Gaussian APIs
are included by the umbrella header when that build feature is enabled.

## Downstream CMake project

Create `CMakeLists.txt` and `main.cpp` in your application directory:

```cmake
cmake_minimum_required(VERSION 3.24)
project(SparseExample LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EXAMPLE_WITH_CUDA "Use a CUDA-enabled Minkowski install" ON)
option(EXAMPLE_WITH_OPENMP "Use an OpenMP-enabled Minkowski install" ON)

find_package(Torch REQUIRED)
if(EXAMPLE_WITH_OPENMP)
  find_package(OpenMP REQUIRED)
endif()
if(EXAMPLE_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
find_package(MinkowskiEngine CONFIG REQUIRED)

add_executable(sparse_example main.cpp)
target_link_libraries(sparse_example PRIVATE MinkowskiEngine::minkowski_cpp)
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(sparse_example PRIVATE ${TORCH_FLAGS})
```

The two example switches describe how your **installed library** was built;
they do not change that library. For a CPU-only install, configure the
application with `-DEXAMPLE_WITH_CUDA=OFF`. Set the OpenMP switch to `OFF`
if the library was installed without OpenMP.

The exported package currently does not automatically call
`find_dependency`, so explicitly discover these dependencies first. Windows
OpenMP builds may export the LibTorch runtime import library from their
build tree; keep that tree available or install a build without OpenMP if
you need to relocate the package independently.

## A complete classifier

This example uses CPU execution so it works with either a CPU-only or a
CUDA-enabled library. It has two sparse convolution blocks, global pooling,
and a linear classification head:

```cpp
#include <minkowski.hpp>
#include <iostream>
#include <memory>

using namespace minkowski;

struct ClassifierImpl : torch::nn::Module {
  MinkowskiConvolution conv1{nullptr}, conv2{nullptr};
  MinkowskiBatchNorm bn1{nullptr}, bn2{nullptr};
  MinkowskiReLU relu1{nullptr}, relu2{nullptr};
  MinkowskiGlobalPooling pool{nullptr};
  MinkowskiLinear head{nullptr};

  ClassifierImpl() {
    conv1 = register_module("conv1", MinkowskiConvolution(
        3, 64, 3, 2, 1, false, false, ConvolutionMode::DEFAULT, 2));
    bn1 = register_module("bn1", MinkowskiBatchNorm(64));
    relu1 = register_module("relu1", MinkowskiReLU());
    conv2 = register_module("conv2", MinkowskiConvolution(
        64, 128, 3, 2, 1, false, false, ConvolutionMode::DEFAULT, 2));
    bn2 = register_module("bn2", MinkowskiBatchNorm(128));
    relu2 = register_module("relu2", MinkowskiReLU());
    pool = register_module("pool", MinkowskiGlobalPooling());
    head = register_module("head", MinkowskiLinear(128, 5));
  }

  SparseTensor forward(SparseTensor x) {
    x = relu1->forward(bn1->forward(conv1->forward(x)));
    x = relu2->forward(bn2->forward(conv2->forward(x)));
    return head->forward(pool->forward(x));
  }
};
TORCH_MODULE(Classifier);

int main() {
  torch::manual_seed(0);
  Classifier model;
  torch::optim::Adam optimizer(model->parameters(),
                               torch::optim::AdamOptions(1e-3));

  auto coords = torch::tensor(
      {{0, 0, 0}, {0, 1, 0}, {0, 0, 1},
       {1, 0, 0}, {1, 1, 0}, {1, 1, 1}}, torch::kInt32);
  auto feats = torch::randn({6, 3});
  auto labels = torch::tensor({0, 2}, torch::kLong);
  auto manager = std::make_shared<CoordinateManager>(
      2, CoordinateMapBackend::CPU);
  SparseTensor input(feats, coords, manager);

  model->train();
  optimizer.zero_grad();
  auto output = model->forward(input);
  auto loss = torch::nn::functional::cross_entropy(output.F(), labels);
  loss.backward();
  optimizer.step();
  std::cout << "Loss: " << loss.item<float>() << '\n';

  save_model(model, "classifier.pt");
}
```

Coordinates have shape `[N, D+1]`, with **batch index first**. Features have
shape `[N, Cin]`. The global pooling layer produces one row per batch, so
the classifier needs one `int64` label per batch, not one label per input row.

## Configure and run

Linux:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH='/opt/libtorch;/usr/local'
cmake --build build --parallel 4
LD_LIBRARY_PATH=/opt/libtorch/lib:${LD_LIBRARY_PATH:-} ./build/sparse_example
```

Windows, from Developer PowerShell:

```powershell
cmake -S . -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="D:\libtorch_cu134;D:\Minkowski4CPP\install"
cmake --build build --parallel 4
$env:PATH = "D:\libtorch_cu134\lib;$env:PATH"
.\build\sparse_example.exe
```

Set `MinkowskiEngine_DIR` explicitly if CMake cannot locate its config.
Keep LibTorch runtime libraries discoverable when running the application.

## Move to CUDA

In a CUDA-enabled Minkowski build, move model, features, coordinates, and
labels to the same CUDA device and create the manager with
`CoordinateMapBackend::CUDA`. Do not select that backend merely because
`torch::cuda::is_available()` is true: a CPU-only Minkowski build can still
be linked to CUDA-enabled LibTorch.

Next: [coordinate semantics and network design](sparse-tensors.md),
[adaptive example](adaptive.md#complete-example), or
[Gaussian scene loading](gaussian.md#load-and-inspect-a-scene).
{% endraw %}
