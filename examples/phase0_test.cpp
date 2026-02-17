#include "minkowski.hpp"
#include <iostream>

using namespace minkowski;

int main() {
  std::cout << "MinkowskiEngine C++ Version: " << VERSION << std::endl;
  std::cout << "Phase 0 Complete - Build System Test" << std::endl;
  
  // This example demonstrates the intended API
  // Implementation will be completed in Phase 1
  
  try {
    // Create a 3D coordinate manager
    auto manager = std::make_shared<CoordinateManager>(
      3,  // dimension
#ifdef CPU_ONLY
      CoordinateMapBackend::CPU
#else
      CoordinateMapBackend::CUDA
#endif
    );
    
    std::cout << "CoordinateManager created with dimension: " << manager->D() << std::endl;
    
    // TODO: Phase 1 will enable:
    // - Creating sparse tensors from coordinates and features
    // - Building networks with MinkowskiConvolution, etc.
    // - Training loops with autograd
    
    std::cout << "\nPhase 0 Status: ✓ Build system functional" << std::endl;
    std::cout << "Next: Implement Phase 1 (Core data structures)" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
