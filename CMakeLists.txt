cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MinkowskiEngineCPP LANGUAGES CXX)

# Options
option(CPU_ONLY "Build CPU-only version without CUDA support" OFF)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find LibTorch
# Set CMAKE_PREFIX_PATH to your LibTorch installation:
#   Windows: cmake -DCMAKE_PREFIX_PATH=E:/Code/libtorch ..
#   Linux:   cmake -DCMAKE_PREFIX_PATH=/opt/libtorch/share/cmake/Torch ..
#   Or set it here: set(CMAKE_PREFIX_PATH "E:/Code/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# MSVC-specific: Use standards-conforming preprocessor for CUDA compatibility
if(MSVC)
    # Apply MSVC flags only to C++ compilation, not CUDA
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd5105>)
    # Suppress C4267: conversion from size_t to int/uint (common in PyTorch headers)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4267>)
endif()

# Find CUDA (if not CPU_ONLY)
if(NOT CPU_ONLY)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Define USE_CUDA so PyTorch headers enable correct Windows+CUDA code paths
    # (e.g. compiled_autograd.h uses this to bypass if-constexpr that fails with NVCC+MSVC)
    add_compile_definitions(USE_CUDA)
    
    # MSVC + CUDA: Pass compiler flags to host compiler via CUDA
    if(MSVC)
        # Force-include cuda_common.hpp to set up MSVC workarounds before any other includes
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FI${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/cuda_common.hpp>)
        # Suppress CCCL warning about traditional preprocessor
        add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING>)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd5105>)
        # Suppress C4267: conversion from size_t to int/uint (common in PyTorch headers)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4267>)
        # Suppress C4244: conversion from larger to smaller type (common in PyTorch headers)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244>)
        # Suppress C4804: unsafe bool in operation (from cudf wrapper_types.hpp)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4804>)
    endif()
endif()

# OpenMP (used by the kernel code)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    if(MSVC)
        # Use LLVM-based OpenMP on MSVC for modern OpenMP 3.1+ features (atomic capture, etc.)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp:llvm")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Include directories are set per-target below (not globally) so that
# absolute source paths don't leak into exported INTERFACE_INCLUDE_DIRECTORIES.

# ============================================================================
# Collect ALL source files (kernel + C++ layer)
# ============================================================================

file(GLOB MINK_CPU_SOURCES
    "MinkowskiEngine/src/*.cpp"
    "MinkowskiEngine/*.cpp"
    "MinkowskiEngine/autograd/*.cpp"
    "MinkowskiEngine/modules/*.cpp"
    "MinkowskiEngine/utils/*.cpp"
    "MinkowskiEngine/training/*.cpp"
)

# Exclude files that are not part of the library
list(FILTER MINK_CPU_SOURCES EXCLUDE REGEX ".*test.*")
list(FILTER MINK_CPU_SOURCES EXCLUDE REGEX ".*example.*")

if(NOT CPU_ONLY)
    file(GLOB MINK_GPU_SOURCES
        "MinkowskiEngine/src/*.cu"
        "MinkowskiEngine/*.cu"
        "MinkowskiEngine/autograd/*.cu"
        "MinkowskiEngine/modules/*.cu"
        "MinkowskiEngine/utils/*.cu"
        "MinkowskiEngine/training/*.cu"
    )
    list(FILTER MINK_GPU_SOURCES EXCLUDE REGEX ".*test.*")
    list(FILTER MINK_GPU_SOURCES EXCLUDE REGEX ".*example.*")
endif()

# ============================================================================
# Unified minkowski_cpp library (kernel + C++ layer)
# ============================================================================

if(CPU_ONLY)
    add_library(minkowski_cpp STATIC ${MINK_CPU_SOURCES})
    target_compile_definitions(minkowski_cpp PUBLIC CPU_ONLY)
else()
    add_library(minkowski_cpp STATIC ${MINK_CPU_SOURCES} ${MINK_GPU_SOURCES})

    # CUDA compilation flags
    # Note: CUDA_SEPARABLE_COMPILATION can cause issues with MSVC+Ninja dependency tracking
    set_target_properties(minkowski_cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )

    # CUDA architectures - reduced set to lower memory usage during compilation
    # 75: Turing (RTX 20xx), 80: Ampere (RTX 30xx, A100), 86: Ampere (RTX 30xx laptop)
    set_property(TARGET minkowski_cpp PROPERTY CUDA_ARCHITECTURES 75 80 86)
endif()

# Link libraries
target_link_libraries(minkowski_cpp PUBLIC ${TORCH_LIBRARIES})

if(NOT CPU_ONLY)
    target_link_libraries(minkowski_cpp PUBLIC CUDA::cusparse CUDA::cublas CUDA::cudart)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(minkowski_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

# Compile definitions
target_compile_definitions(minkowski_cpp PUBLIC MINK_NO_PYBIND)

if(WIN32)
    target_compile_definitions(minkowski_cpp PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Include directories
# PRIVATE: source-tree paths needed only for building this target
target_include_directories(minkowski_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine
    ${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src
    ${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/3rdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
# INTERFACE: install-tree paths for downstream consumers after install
target_include_directories(minkowski_cpp INTERFACE
    $<INSTALL_INTERFACE:include/minkowski>
    $<INSTALL_INTERFACE:include/minkowski/src>
    $<INSTALL_INTERFACE:include/minkowski/src/3rdparty>
)

if(NOT CPU_ONLY)
    target_include_directories(minkowski_cpp PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}/cccl
    )
endif()

# ============================================================================
# Test executable: train
# ============================================================================

add_executable(train tests/train.cpp)
target_link_libraries(train PRIVATE minkowski_cpp ${TORCH_LIBRARIES})
target_include_directories(train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine
    ${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src
    ${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/3rdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
if(CPU_ONLY)
    target_compile_definitions(train PRIVATE CPU_ONLY)
endif()
if(WIN32)
    target_compile_definitions(train PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
target_compile_definitions(train PRIVATE MINK_NO_PYBIND)

# ============================================================================
# Installation and CMake package config
# ============================================================================

install(TARGETS minkowski_cpp
    EXPORT MinkowskiEngineTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY MinkowskiEngine/
    DESTINATION include/minkowski
    FILES_MATCHING
      PATTERN "*.hpp"
      PATTERN "*.h"
      PATTERN "*.hh"
      PATTERN "*.hxx"
      PATTERN "*.inl"
      PATTERN "*.cuh"
)

# Create namespaced alias for consumption: use `MinkowskiEngine::MinkowskiEngine`
add_library(MinkowskiEngine::MinkowskiEngine ALIAS minkowski_cpp)
# Also provide alias matching exported target name (export uses the target name `minkowski_cpp`)
add_library(MinkowskiEngine::minkowski_cpp ALIAS minkowski_cpp)

# On Windows, when building shared libs, this helps export symbols automatically.
if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Export target for use with find_package(... CONFIG)
include(CMakePackageConfigHelpers)
set(CONFIG_INSTALL_DIR lib/cmake/MinkowskiEngine)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MinkowskiEngineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MinkowskiEngineConfig.cmake
    INSTALL_DESTINATION ${CONFIG_INSTALL_DIR}
    NO_SET_AND_CHECK_MACRO
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MinkowskiEngineConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

install(EXPORT MinkowskiEngineTargets
    FILE MinkowskiEngineTargets.cmake
    NAMESPACE MinkowskiEngine::
    DESTINATION ${CONFIG_INSTALL_DIR}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MinkowskiEngineConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MinkowskiEngineConfigVersion.cmake
    DESTINATION ${CONFIG_INSTALL_DIR}
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
