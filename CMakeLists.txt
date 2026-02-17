cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MinkowskiEngineCPP LANGUAGES CXX)

# Options
option(CPU_ONLY "Build CPU-only version without CUDA support" OFF)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# MSVC-specific: Use standards-conforming preprocessor for CUDA compatibility
if(MSVC)
    add_compile_options(/Zc:preprocessor)
    # Also disable warnings about the preprocessor switch
    add_compile_options(/wd5105)
endif()

# Find Python (needed for torch/extension.h includes)
find_package(Python COMPONENTS Interpreter Development)

# Find CUDA (if not CPU_ONLY)
if(NOT CPU_ONLY)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# OpenMP (used by the kernel code)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Include directories
include_directories(
    /usr/include/python3.10
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mink
    ${CMAKE_CURRENT_SOURCE_DIR}/mink/3rdparty
)

# Collect kernel source files
file(GLOB MINK_CPU_SOURCES
    "mink/*.cpp"
)

# Exclude files that are not part of the kernel library
list(FILTER MINK_CPU_SOURCES EXCLUDE REGEX ".*test.*")
list(FILTER MINK_CPU_SOURCES EXCLUDE REGEX ".*example.*")

if(NOT CPU_ONLY)
    file(GLOB MINK_GPU_SOURCES
        "mink/*.cu"
    )
    list(FILTER MINK_GPU_SOURCES EXCLUDE REGEX ".*test.*")
    list(FILTER MINK_GPU_SOURCES EXCLUDE REGEX ".*example.*")
endif()

# Create kernel library target
if(CPU_ONLY)
    add_library(minkowski_kernel STATIC ${MINK_CPU_SOURCES})
    target_compile_definitions(minkowski_kernel PUBLIC CPU_ONLY)
else()
    add_library(minkowski_kernel STATIC ${MINK_CPU_SOURCES} ${MINK_GPU_SOURCES})
    
    # CUDA compilation flags
    set_target_properties(minkowski_kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # CUDA architectures - reduced set to lower memory usage during compilation
    # 75: Turing (RTX 20xx), 80: Ampere (RTX 30xx, A100), 86: Ampere (RTX 30xx laptop)
    set_property(TARGET minkowski_kernel PROPERTY CUDA_ARCHITECTURES 75 80 86)
endif()

# Link libraries for kernel
target_link_libraries(minkowski_kernel PUBLIC ${TORCH_LIBRARIES})

if(NOT CPU_ONLY)
    target_link_libraries(minkowski_kernel PUBLIC CUDA::cusparse CUDA::cublas CUDA::cudart)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(minkowski_kernel PUBLIC OpenMP::OpenMP_CXX)
endif()

# Compile definitions for kernel library
target_compile_definitions(minkowski_kernel PUBLIC MINK_NO_PYBIND)

# Windows-specific definitions to prevent macro conflicts
if(WIN32)
    target_compile_definitions(minkowski_kernel PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Include directories for kernel
target_include_directories(minkowski_kernel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/mink
    ${CMAKE_CURRENT_SOURCE_DIR}/mink/3rdparty
)

if(NOT CPU_ONLY)
    target_include_directories(minkowski_kernel PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}/cccl
    )
endif()

if(Python_FOUND)
    target_include_directories(minkowski_kernel PRIVATE ${Python_INCLUDE_DIRS})
endif()

# ============================================================================
# New C++ Layer Library
# ============================================================================

# Collect new C++ layer sources
file(GLOB MINK_CPP_SOURCES
    "mink_cpp/*.cpp"
)

# Collect CUDA sources in mink_cpp (if not CPU_ONLY)
if(NOT CPU_ONLY)
    file(GLOB MINK_CPP_CUDA_SOURCES
        "mink_cpp/*.cu"
    )
endif()

# Create the new C++ layer library
if(CPU_ONLY)
    add_library(minkowski_cpp STATIC ${MINK_CPP_SOURCES})
else()
    add_library(minkowski_cpp STATIC ${MINK_CPP_SOURCES} ${MINK_CPP_CUDA_SOURCES})
    
    # CUDA compilation flags for mink_cpp
    set_target_properties(minkowski_cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Same reduced architecture set as kernel library
    set_property(TARGET minkowski_cpp PROPERTY CUDA_ARCHITECTURES 75 80 86)
endif()

# Link against kernel library and LibTorch
target_link_libraries(minkowski_cpp PUBLIC
    minkowski_kernel
    ${TORCH_LIBRARIES}
)

# Include directories for C++ layer
target_include_directories(minkowski_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mink_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mink
)

# Compile definitions
if(CPU_ONLY)
    target_compile_definitions(minkowski_cpp PUBLIC CPU_ONLY)
endif()

if(WIN32)
    target_compile_definitions(minkowski_cpp PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ============================================================================
# Tests (optional)
# ============================================================================

option(BUILD_TESTS "Build test executables" ON)

if(BUILD_TESTS)
    # Phase 1 test
    add_executable(phase1_test tests/phase1_test.cpp)
    target_link_libraries(phase1_test PRIVATE minkowski_cpp)
    if(WIN32)
        target_compile_definitions(phase1_test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
    
    # Add more test executables here when ready
    # add_executable(test_sparse_tensor tests/test_sparse_tensor.cpp)
    # target_link_libraries(test_sparse_tensor PRIVATE minkowski_cpp)
endif()

# ============================================================================
# Examples (optional)
# ============================================================================

option(BUILD_EXAMPLES "Build example executables" ON)

if(BUILD_EXAMPLES)
    # Phase 0 test example
    add_executable(phase0_test examples/phase0_test.cpp)
    target_link_libraries(phase0_test PRIVATE minkowski_cpp)
    
    # Uncomment when ready:
    # add_executable(example_unet examples/example_unet.cpp)
    # target_link_libraries(example_unet PRIVATE minkowski_cpp)
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS minkowski_kernel minkowski_cpp
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY mink_cpp/
    DESTINATION include/minkowski
    FILES_MATCHING PATTERN "*.hpp"
)

# ============================================================================
# Info
# ============================================================================

message(STATUS "========================================")
message(STATUS "MinkowskiEngine C++ Configuration:")
message(STATUS "  CPU_ONLY: ${CPU_ONLY}")
message(STATUS "  LibTorch: ${TORCH_LIBRARIES}")
if(NOT CPU_ONLY)
    message(STATUS "  CUDA: ${CUDAToolkit_VERSION}")
endif()
message(STATUS "========================================")
