#ifndef MINK_CPP_MINKOWSKI_HPP
#define MINK_CPP_MINKOWSKI_HPP

// Core types and data structures
#include "types.hpp"
#include "coordinate_manager.hpp"
#include "kernel_generator.hpp"
#include "sparse_tensor.hpp"
#include "tensor_field.hpp"

// Autograd functions (will be added in Phase 2)
// #include "autograd/convolution_fn.hpp"
// #include "autograd/pooling_fn.hpp"
// ... etc

// Modules (will be added in Phase 3)
// #include "modules/convolution.hpp"
// #include "modules/pooling.hpp"
// ... etc

// Utilities (will be added in Phase 4)
// #include "utils/collation.hpp"
// #include "utils/quantization.hpp"
// #include "utils/init.hpp"

namespace minkowski {

// Version information
constexpr const char* VERSION = "0.1.0-cpp";
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;

} // namespace minkowski

#endif // MINK_CPP_MINKOWSKI_HPP
