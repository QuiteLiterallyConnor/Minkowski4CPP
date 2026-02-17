/*
 * Phase 1 Test Example
 * 
 * This file demonstrates the usage of Phase 1 components:
 * - KernelGenerator
 * - CoordinateManager
 * - SparseTensor
 * - TensorField
 */

#include "mink_cpp/minkowski.hpp"
#include <torch/torch.h>
#include <iostream>

using namespace minkowski;

void test_kernel_generator() {
    std::cout << "=== Testing KernelGenerator ===" << std::endl;
    
    // Create a 3D kernel generator
    torch::IntArrayRef kernel_size = {3, 3, 3};
    torch::IntArrayRef stride = {1, 1, 1};
    torch::IntArrayRef dilation = {1, 1, 1};
    
    KernelGenerator kernel_gen(
        kernel_size, stride, dilation, false,
        RegionType::HYPER_CUBE, at::Tensor(), false, {}, 3);
    
    std::cout << "Kernel volume: " << kernel_gen.kernel_volume() << std::endl;
    std::cout << "Dimension: " << kernel_gen.dimension() << std::endl;
    
    // Get kernel parameters for a specific tensor stride
    stride_type tensor_stride = {1, 1, 1};
    auto params = kernel_gen.get_kernel(tensor_stride, false);
    std::cout << "Kernel size: " << params.kernel_size[0] << "x" 
              << params.kernel_size[1] << "x" << params.kernel_size[2] << std::endl;
}

void test_sparse_tensor() {
    std::cout << "\n=== Testing SparseTensor ===" << std::endl;
    
    // Create sample coordinates (batch_idx, x, y, z)
    auto coords = torch::tensor({
        {0, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 1, 1, 0},
        {1, 0, 0, 0},  // Second batch
        {1, 1, 0, 0}
    }, torch::kInt32);
    
    // Create sample features
    auto feats = torch::randn({6, 128});
    
    // Create sparse tensor with random subsample quantization
    auto sparse_tensor = SparseTensor(
        feats, coords, nullptr,
        SparseTensorQuantizationMode::RANDOM_SUBSAMPLE);
    
    std::cout << "Feature shape: [" << sparse_tensor.F().size(0) << ", " 
              << sparse_tensor.F().size(1) << "]" << std::endl;
    std::cout << "Dimension: " << sparse_tensor.dimension() << std::endl;
    std::cout << "Device: " << sparse_tensor.device() << std::endl;
    
    // Test arithmetic operations
    auto doubled = sparse_tensor * 2.0;
    auto sum = sparse_tensor + sparse_tensor;
    std::cout << "Arithmetic operations: OK" << std::endl;
    
    // Test batch decomposition
    auto decomposed_coords = sparse_tensor.decomposed_coordinates();
    std::cout << "Number of batches: " << decomposed_coords.size() << std::endl;
    std::cout << "Batch 0 size: " << decomposed_coords[0].size(0) << std::endl;
    std::cout << "Batch 1 size: " << decomposed_coords[1].size(0) << std::endl;
}

void test_tensor_field() {
    std::cout << "\n=== Testing TensorField ===" << std::endl;
    
    // Create continuous coordinates (batch_idx, x, y, z)
    auto field_coords = torch::tensor({
        {0.0f, 0.5f, 0.3f, 0.1f},
        {0.0f, 1.2f, 0.8f, 0.5f},
        {0.0f, 2.1f, 1.9f, 1.2f},
        {1.0f, 0.3f, 0.7f, 0.4f}  // Second batch
    }, torch::kFloat);
    
    // Create features
    auto field_feats = torch::randn({4, 64});
    
    // Create tensor field
    auto tensor_field = TensorField(field_feats, field_coords);
    
    std::cout << "Field feature shape: [" << tensor_field.F().size(0) << ", " 
              << tensor_field.F().size(1) << "]" << std::endl;
    std::cout << "Dimension: " << tensor_field.dimension() << std::endl;
    
    // Convert to sparse with quantization
    auto sparse_from_field = tensor_field.sparse();
    std::cout << "Converted to sparse, shape: [" << sparse_from_field.F().size(0) 
              << ", " << sparse_from_field.F().size(1) << "]" << std::endl;
    
    // Test splat (linear interpolation to 2^D neighbors)
    auto splatted = tensor_field.splat();
    std::cout << "Splatted tensor shape: [" << splatted.F().size(0) << ", " 
              << splatted.F().size(1) << "]" << std::endl;
}

void test_coordinate_manager() {
    std::cout << "\n=== Testing CoordinateManager ===" << std::endl;
    
    // Create coordinate manager
    auto manager = std::make_shared<CoordinateManager>(
        3,  // 3D spatial dimension
        CoordinateMapBackend::CPU);
    
    // Insert coordinates
    auto coords = torch::tensor({
        {0, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0}
    }, torch::kInt32);
    
    stride_type tensor_stride = {1, 1, 1};
    auto [key, maps] = manager->insert_and_map(coords, tensor_stride, "");
    auto& [unique_map, inverse_map] = maps;
    
    std::cout << "Inserted " << coords.size(0) << " coordinates" << std::endl;
    std::cout << "Unique count: " << manager->size(key) << std::endl;
    
    // Get coordinates back
    auto retrieved_coords = manager->get_coordinates(key);
    std::cout << "Retrieved coordinates shape: [" << retrieved_coords.size(0) 
              << ", " << retrieved_coords.size(1) << "]" << std::endl;
}

int main() {
    std::cout << "MinkowskiEngine C++ Phase 1 Tests" << std::endl;
    std::cout << "==================================" << std::endl;
    
    try {
        test_kernel_generator();
        test_sparse_tensor();
        test_tensor_field();
        test_coordinate_manager();
        
        std::cout << "\n==================================" << std::endl;
        std::cout << "All Phase 1 tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
