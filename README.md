[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-minkowski4cpp-0db7ed?logo=docker&logoColor=white&style=for-the-badge)](https://hub.docker.com/r/acaicia/minkowski4cpp)

# Minkowski4CPP

**A C++ port of NVIDIA's MinkowskiEngine with full CUDA support for Windows and Linux.**

Minkowski4CPP brings the power of sparse tensor operations from [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) into native C++ workflows, with first-class CUDA acceleration on both major platforms — something the original Python-based library does not support.

---

## Platform Support

| Platform | CUDA Version | Torch Version | Status |
|---|---|---|---|
| Ubuntu 22.04 | 12.4.0 | 2.5.1 | Verified |
| Windows | 13.1 | 2.10.0 | Verified |

---

## Prerequisites

### Windows

- [CUDA Toolkit 13.1](https://developer.nvidia.com/cuda-downloads)
- [LibTorch 2.10.0](https://pytorch.org/get-started/locally/) (pre-built, extract only)
- [Ninja](https://ninja-build.org/) (extract only)
- [CMake](https://cmake.org/download/) (install)

### Linux

- [CUDA Toolkit 12.4.0](https://developer.nvidia.com/cuda-downloads)
- [LibTorch 2.5.1](https://pytorch.org/get-started/locally/) (pre-built, extract only)
- `ninja-build` — install via your package manager
- `cmake` — install via your package manager

---

## Building

### Windows

Run the provided PowerShell build script, passing paths to your CUDA installation, LibTorch directory, and Ninja executable:

```powershell
scripts\build-all-windows.ps1 `
  -CudaPath    [PATH TO CUDA] `
  -LibTorchPath [PATH TO TORCH] `
  -NinjaPath   [PATH TO NINJA EXECUTABLE]
```

**Example:**

```powershell
scripts\build-all-windows.ps1 `
  -CudaPath     "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" `
  -LibTorchPath "E:\Code\libtorch" `
  -NinjaPath    "C:\Program Files\Ninja\ninja.exe"
```

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
        // First convolution block: Conv → BN → ReLU
        conv1_conv = register_module("conv1_conv", MinkowskiConvolution(
            /*in_channels=*/in_feat, /*out_channels=*/64,
            /*kernel_size=*/3, /*stride=*/2, /*dilation=*/1,
            /*bias=*/false, /*expand_coordinates=*/false,
            ConvolutionMode::DEFAULT, /*dimension=*/D));
        conv1_bn   = register_module("conv1_bn", MinkowskiBatchNorm(64));
        conv1_relu = register_module("conv1_relu", MinkowskiReLU());

        // Second convolution block: Conv → BN → ReLU
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
// Reusable Conv → BN → ReLU block
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

// Residual block: two 3×3 convolutions with a skip connection
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

        // Per-point classification head (1×1 conv)
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
