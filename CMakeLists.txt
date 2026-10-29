
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(MinkowskiEngineCPP
    VERSION 1.0.0
    LANGUAGES CXX
)

# ============================================================
# Options
# ============================================================

option(CPU_ONLY "Build without Minkowski CUDA kernels" OFF)
option(MINK_BUILD_TESTS "Build test executables" ON)
option(MINK_ENABLE_OPENMP "Enable OpenMP support" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ============================================================
# CUDA configuration
# ============================================================

if(NOT CPU_ONLY)

    # Establish the toolkit before Torch/Caffe2 discovery.
    # The Windows script should supply CUDAToolkit_ROOT and
    # CMAKE_CUDA_COMPILER explicitly.
    
    if(MSVC)
        set(CMAKE_CUDA_FLAGS_INIT
            "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler"
        )
    endif()

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)

    # Respect -DCMAKE_CUDA_ARCHITECTURES from the caller.
    # Default to Ampere sm_86 for the current build machine.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING
            "CUDA architectures" FORCE)
    endif()

    message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # Normalize cuDNN paths supplied by build scripts.
    if(DEFINED CUDNN_INCLUDE_DIR AND CUDNN_INCLUDE_DIR)
        set(CUDNN_INCLUDE_PATH
            "${CUDNN_INCLUDE_DIR}"
            CACHE PATH "cuDNN include directory" FORCE)
    endif()

    if(DEFINED CUDNN_LIBRARY AND CUDNN_LIBRARY)
        set(CUDNN_LIBRARY_PATH
            "${CUDNN_LIBRARY}"
            CACHE FILEPATH "cuDNN import library" FORCE)
    endif()

endif()

# ============================================================
# LibTorch
# ============================================================

# LibTorch remains required in both CPU and CUDA builds.
# CPU_ONLY disables Minkowski CUDA kernels, not CUDA
# dependencies of a CUDA-enabled LibTorch distribution.

find_package(Torch REQUIRED)

# Older LibTorch configs hardcode the legacy NvToolsExt SDK on Windows. It is not
# shipped with CUDA 13.x (NVTX 3 is header-only in the toolkit), so drop the
# missing paths instead of failing at generate time.
if(MSVC AND NOT EXISTS "C:/Program Files/NVIDIA Corporation/NvToolsExt/include")
    list(FILTER TORCH_INCLUDE_DIRS EXCLUDE REGEX "NvToolsExt")
    list(FILTER TORCH_LIBRARIES EXCLUDE REGEX "nvToolsExt")
    if(TARGET torch)
        set_target_properties(torch PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}")
    endif()
endif()

message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# Torch may provide compiler flags needed for ABI
# compatibility. Keep these as separate list elements.
separate_arguments(
    MINK_TORCH_CXX_FLAGS
    NATIVE_COMMAND
    "${TORCH_CXX_FLAGS}"
)

# ============================================================
# OpenMP
# ============================================================

if(MINK_ENABLE_OPENMP)
    if(MSVC)
        # '#pragma omp atomic capture' requires the LLVM OpenMP runtime on MSVC.
        set(OpenMP_RUNTIME_MSVC llvm)
        unset(OpenMP_CXX_FLAGS CACHE)
        unset(OpenMP_CXX_LIB_NAMES CACHE)
    endif()
    find_package(OpenMP COMPONENTS CXX QUIET)

    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP enabled")
    else()
        message(WARNING
            "OpenMP was requested but not found. "
            "Continuing without OpenMP.")
    endif()
endif()

# ============================================================
# Source files
# ============================================================

file(GLOB MINK_CPU_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/autograd/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/modules/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/utils/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/training/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/adaptive/*.cpp"
)

list(FILTER MINK_CPU_SOURCES EXCLUDE REGEX ".*test.*")
list(FILTER MINK_CPU_SOURCES EXCLUDE REGEX ".*example.*")

set(MINK_GPU_SOURCES)

if(NOT CPU_ONLY)

    file(GLOB MINK_GPU_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/autograd/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/modules/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/utils/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/training/*.cu"
    )

    list(FILTER MINK_GPU_SOURCES EXCLUDE REGEX ".*test.*")
    list(FILTER MINK_GPU_SOURCES EXCLUDE REGEX ".*example.*")

endif()

# ============================================================
# Main library
# ============================================================

add_library(minkowski_cpp STATIC
    ${MINK_CPU_SOURCES}
    ${MINK_GPU_SOURCES}
)

add_library(MinkowskiEngine::minkowski_cpp
    ALIAS minkowski_cpp)

add_library(MinkowskiEngine::MinkowskiEngine
    ALIAS minkowski_cpp)

set_target_properties(minkowski_cpp PROPERTIES
    EXPORT_NAME minkowski_cpp
    POSITION_INDEPENDENT_CODE ON
)

if(NOT CPU_ONLY)
    set_target_properties(minkowski_cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

# ============================================================
# Include directories
# ============================================================

target_include_directories(minkowski_cpp
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/3rdparty"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"

    PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/minkowski>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/minkowski/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/minkowski/src/3rdparty>
)

# ============================================================
# Compile definitions
# ============================================================

target_compile_definitions(minkowski_cpp
    PUBLIC
        MINK_NO_PYBIND
)

if(CPU_ONLY)
    target_compile_definitions(minkowski_cpp
        PUBLIC CPU_ONLY)
else()
    target_compile_definitions(minkowski_cpp
        PUBLIC USE_CUDA)
endif()

if(WIN32)
    target_compile_definitions(minkowski_cpp
        PUBLIC
            NOMINMAX
            WIN32_LEAN_AND_MEAN
    )
endif()

# ============================================================
# Compiler configuration
# ============================================================

if(MINK_TORCH_CXX_FLAGS)
    target_compile_options(minkowski_cpp
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${MINK_TORCH_CXX_FLAGS}>
    )
endif()

if(MSVC)

    target_compile_options(minkowski_cpp PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    )

    target_compile_options(minkowski_cpp PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/wd5105>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
    )

    if(NOT CPU_ONLY)

        set(MINK_CUDA_COMMON
            "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/cuda_common.hpp"
        )

        if(EXISTS "${MINK_CUDA_COMMON}")

            # Forward the forced-include option to MSVC
            # when compiling CUDA translation units.
            target_compile_options(minkowski_cpp PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FI${MINK_CUDA_COMMON}>
            )

        else()
            message(WARNING
                "CUDA workaround header not found: "
                "${MINK_CUDA_COMMON}")
        endif()

        target_compile_definitions(minkowski_cpp PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING>
        )

        # CCCL (Thrust/CUB) requires the conforming MSVC preprocessor.
        target_compile_options(minkowski_cpp PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd5105>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4267>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4804>
        )

    endif()

endif()

# ============================================================
# Dependencies
# ============================================================

target_link_libraries(minkowski_cpp
    PUBLIC
        ${TORCH_LIBRARIES}
)

if(NOT CPU_ONLY)

    target_link_libraries(minkowski_cpp
        PUBLIC
            CUDA::cudart
            CUDA::cusparse
            CUDA::cublas
    )

    target_include_directories(minkowski_cpp PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        "${CUDAToolkit_INCLUDE_DIRS}/cccl"
    )

endif()

if(MINK_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(minkowski_cpp
        PUBLIC OpenMP::OpenMP_CXX)
endif()

# ============================================================
# Test executable
# ============================================================

if(MINK_BUILD_TESTS)

    add_executable(train
        tests/train.cpp
    )

    target_link_libraries(train
        PRIVATE minkowski_cpp
    )

    target_include_directories(train PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/3rdparty"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    )

    if(MINK_TORCH_CXX_FLAGS)
        target_compile_options(train PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${MINK_TORCH_CXX_FLAGS}>
        )
    endif()

endif()

# Adaptive voxelization tests and benchmark (see plan.txt).
option(MINK_BUILD_ADAPTIVE_TESTS "Build adaptive voxelization tests and benchmarks" ON)

if(MINK_BUILD_ADAPTIVE_TESTS)

    add_executable(adaptive_tests
        tests/adaptive/main.cpp
        tests/adaptive/test_geometry.cpp
        tests/adaptive/test_topology.cpp
        tests/adaptive/test_numerical.cpp
        tests/adaptive/test_gradients.cpp
        tests/adaptive/test_regression.cpp
    )

    add_executable(adaptive_bench
        tests/adaptive/bench_adaptive.cpp
    )

    foreach(tgt adaptive_tests adaptive_bench)
        target_link_libraries(${tgt} PRIVATE minkowski_cpp)
        target_include_directories(${tgt} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine"
            "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/3rdparty"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/adaptive"
        )
        if(MINK_TORCH_CXX_FLAGS)
            target_compile_options(${tgt} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:${MINK_TORCH_CXX_FLAGS}>
            )
        endif()
    endforeach()

    enable_testing()
    add_test(NAME adaptive_tests COMMAND adaptive_tests)

endif()

# ============================================================
# Installation
# ============================================================

install(TARGETS minkowski_cpp
    EXPORT MinkowskiEngineTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

install(DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/minkowski"
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
        PATTERN "*.hh"
        PATTERN "*.hxx"
        PATTERN "*.inl"
        PATTERN "*.cuh"
)

# ============================================================
# Package configuration
# ============================================================

set(CONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/MinkowskiEngine"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MinkowskiEngineConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MinkowskiEngineConfig.cmake"
    INSTALL_DESTINATION "${CONFIG_INSTALL_DIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MinkowskiEngineConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(EXPORT MinkowskiEngineTargets
    FILE MinkowskiEngineTargets.cmake
    NAMESPACE MinkowskiEngine::
    DESTINATION "${CONFIG_INSTALL_DIR}"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MinkowskiEngineConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MinkowskiEngineConfigVersion.cmake"
    DESTINATION "${CONFIG_INSTALL_DIR}"
)

# ============================================================
# Configuration summary
# ============================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "MinkowskiEngine C++ Configuration")
message(STATUS "========================================")
message(STATUS "Version:       ${PROJECT_VERSION}")
message(STATUS "Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "CPU only:      ${CPU_ONLY}")
message(STATUS "Torch:         ${Torch_VERSION}")
message(STATUS "OpenMP:        ${OpenMP_CXX_FOUND}")
message(STATUS "Build tests:   ${MINK_BUILD_TESTS}")

if(NOT CPU_ONLY)
    message(STATUS "CUDA:          ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

message(STATUS "========================================")
