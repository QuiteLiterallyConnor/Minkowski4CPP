
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
option(MINKOWSKI_ENABLE_GAUSSIAN
    "Enable native Gaussian representation support" ON)
option(MINK_BUILD_GAUSSIAN_BENCHMARKS
    "Build Gaussian tensor and optimization benchmarks" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ============================================================
# Compiler cache (speeds up rebuilds after header/unrelated edits)
# ============================================================

option(MINK_USE_CCACHE "Use ccache/sccache when available" ON)

if(MINK_USE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(MINK_CCACHE_PROGRAM NAMES ccache sccache)
    if(MINK_CCACHE_PROGRAM)
        message(STATUS "Compiler cache: ${MINK_CCACHE_PROGRAM}")
        set(CMAKE_C_COMPILER_LAUNCHER "${MINK_CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${MINK_CCACHE_PROGRAM}")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${MINK_CCACHE_PROGRAM}")
        if(MSVC)
            # Caches cannot handle /Zi PDBs; embed debug info in objects.
            if(POLICY CMP0141)
                cmake_policy(SET CMP0141 NEW)
                set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
            endif()
        endif()
    endif()
endif()

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

if(MINKOWSKI_ENABLE_GAUSSIAN)
    file(GLOB MINK_GAUSSIAN_CPU_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/gaussian/*.cpp"
    )
    list(APPEND MINK_CPU_SOURCES ${MINK_GAUSSIAN_CPU_SOURCES})
endif()

set(MINK_GPU_SOURCES)

if(NOT CPU_ONLY)

    file(GLOB MINK_GPU_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/autograd/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/modules/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/utils/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/adaptive/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/training/*.cu"
    )

    list(FILTER MINK_GPU_SOURCES EXCLUDE REGEX ".*test.*")
    list(FILTER MINK_GPU_SOURCES EXCLUDE REGEX ".*example.*")

    if(MINKOWSKI_ENABLE_GAUSSIAN)
        file(GLOB MINK_GAUSSIAN_GPU_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/gaussian/*.cu"
        )
        list(APPEND MINK_GPU_SOURCES ${MINK_GAUSSIAN_GPU_SOURCES})
    endif()

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

if(MINKOWSKI_ENABLE_GAUSSIAN)
    target_compile_definitions(minkowski_cpp
        PUBLIC MINKOWSKI_ENABLE_GAUSSIAN)
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
    set(MINK_IOMP_DLL "${TORCH_INSTALL_PREFIX}/lib/libiomp5md.dll")
    set(MINK_IOMP_LIB "${CMAKE_BINARY_DIR}/libiomp5md.lib")
    if(MSVC AND TORCH_INSTALL_PREFIX AND EXISTS "${MINK_IOMP_DLL}")
        # LibTorch loads libiomp5md, and a second LLVM-style runtime
        # (libomp140) in the same process aborts with "OMP: Error #15".
        # LibTorch ships no import library, so build one from the DLL and link
        # the engine against the runtime that is already loaded.
        if(NOT EXISTS "${MINK_IOMP_LIB}")
            get_filename_component(_mink_msvc_bin "${CMAKE_LINKER}" DIRECTORY)
            execute_process(
                COMMAND "${_mink_msvc_bin}/dumpbin.exe" /nologo /exports "${MINK_IOMP_DLL}"
                OUTPUT_VARIABLE _mink_exports RESULT_VARIABLE _mink_dumpbin_rc)
            set(_mink_def "LIBRARY libiomp5md\nEXPORTS\n")
            string(REGEX MATCHALL "\n +[0-9]+ +[0-9A-F]+ +[0-9A-F]+ +[A-Za-z_0-9@?$]+"
                   _mink_rows "${_mink_exports}")
            foreach(_row IN LISTS _mink_rows)
                string(REGEX REPLACE "^\n +[0-9]+ +[0-9A-F]+ +[0-9A-F]+ +" "" _name "${_row}")
                string(APPEND _mink_def "    ${_name}\n")
            endforeach()
            file(WRITE "${CMAKE_BINARY_DIR}/libiomp5md.def" "${_mink_def}")
            execute_process(
                COMMAND "${_mink_msvc_bin}/lib.exe" /nologo /machine:x64
                        "/def:${CMAKE_BINARY_DIR}/libiomp5md.def" "/out:${MINK_IOMP_LIB}"
                RESULT_VARIABLE _mink_lib_rc)
        endif()
    endif()

    if(EXISTS "${MINK_IOMP_LIB}")
        message(STATUS "OpenMP runtime: LibTorch libiomp5md")
        target_compile_options(minkowski_cpp PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>)
        target_link_libraries(minkowski_cpp PUBLIC "${MINK_IOMP_LIB}")
        target_link_options(minkowski_cpp PUBLIC /NODEFAULTLIB:libomp140.x86_64.lib)
    else()
        target_link_libraries(minkowski_cpp
            PUBLIC OpenMP::OpenMP_CXX)
    endif()
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
        tests/adaptive/test_sphere.cpp
    )

    add_executable(adaptive_bench
        tests/adaptive/bench_adaptive.cpp
    )

    add_executable(sphere
        tests/sphere.cpp
    )

    # Adaptive vs sparse-uniform voxelization benchmark (driver:
    # tests/adaptive/run_adaptive_vs_uniform.py).
    add_executable(adaptive_vs_uniform_bench
        tests/adaptive/bench_adaptive_vs_uniform.cpp
    )
    target_compile_definitions(adaptive_vs_uniform_bench PRIVATE
        MINK_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    )

    foreach(tgt adaptive_tests adaptive_bench sphere adaptive_vs_uniform_bench)
        target_link_libraries(${tgt} PRIVATE minkowski_cpp)
        target_include_directories(${tgt} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine"
            "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/3rdparty"
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
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
    # Smoke run only (tiny sizes, one repetition): keeps the benchmark building and running.
    add_test(NAME adaptive_vs_uniform_bench_smoke
        COMMAND adaptive_vs_uniform_bench --datasets clustered,shell
                --methods u_coarse,u_fine,u_matched,a_roi,a_dens16,a_var
                --points 3000 --ml-queries 200 --warmup 0 --reps 1)

endif()

# ============================================================
# Gaussian tests
# ============================================================

if(MINK_BUILD_TESTS AND MINKOWSKI_ENABLE_GAUSSIAN)
    add_executable(gaussian_tests
        tests/gaussian/main.cpp
        tests/gaussian/test_gaussian3d.cpp
        tests/gaussian/test_gaussian_tensor.cpp
        tests/gaussian/test_gaussian_ply.cpp
        tests/gaussian/test_gaussian_render.cpp
    )

    target_compile_definitions(gaussian_tests PRIVATE
        MINK_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
    )
    target_link_libraries(gaussian_tests PRIVATE minkowski_cpp)
    target_include_directories(gaussian_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/src/3rdparty"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/adaptive"
    )
    if(MINK_TORCH_CXX_FLAGS)
        target_compile_options(gaussian_tests PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${MINK_TORCH_CXX_FLAGS}>
        )
    endif()

    # ============================================================
    # Gaussian benchmarks
    # ============================================================

    if(MINK_BUILD_GAUSSIAN_BENCHMARKS AND MINKOWSKI_ENABLE_GAUSSIAN)
        add_executable(gaussian_bench
            benchmarks/gaussian_bench.cpp
        )
        target_link_libraries(gaussian_bench PRIVATE minkowski_cpp)
        target_include_directories(gaussian_bench PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine"
        )
        target_compile_definitions(gaussian_bench PRIVATE
            MINK_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        )
        if(MINK_TORCH_CXX_FLAGS)
            target_compile_options(gaussian_bench PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:${MINK_TORCH_CXX_FLAGS}>
            )
        endif()
    endif()

    if(MINK_BUILD_GAUSSIAN_BENCHMARKS AND MINKOWSKI_ENABLE_GAUSSIAN)
        add_executable(gaussian_ply_bench
            benchmarks/gaussian_ply_bench.cpp
        )
        target_link_libraries(gaussian_ply_bench PRIVATE minkowski_cpp)
        target_include_directories(gaussian_ply_bench PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine"
        )
        target_compile_definitions(gaussian_ply_bench PRIVATE
            MINK_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
            MINK_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        )
        if(MINK_TORCH_CXX_FLAGS)
            target_compile_options(gaussian_ply_bench PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:${MINK_TORCH_CXX_FLAGS}>
            )
        endif()
    endif()

    enable_testing()
    add_test(NAME gaussian_tests COMMAND gaussian_tests)
endif()

# Windows resolves DLLs next to the executable, so without this the binaries
# exit silently (0xC0000135) unless LibTorch's lib directory is on PATH.
option(MINK_COPY_TORCH_DLLS "Copy LibTorch DLLs next to the built executables (Windows)" ON)
if(WIN32 AND MINK_COPY_TORCH_DLLS AND TORCH_INSTALL_PREFIX)
    file(GLOB MINK_TORCH_RUNTIME_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    if(MINK_TORCH_RUNTIME_DLLS)
        add_custom_target(mink_copy_torch_dlls ALL
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${MINK_TORCH_RUNTIME_DLLS} "${CMAKE_BINARY_DIR}"
            COMMENT "Copying LibTorch DLLs to ${CMAKE_BINARY_DIR}"
            COMMAND_EXPAND_LISTS
        )
    endif()
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
        PATTERN "gaussian" EXCLUDE
        PATTERN "*.hpp"
        PATTERN "*.h"
        PATTERN "*.hh"
        PATTERN "*.hxx"
        PATTERN "*.inl"
        PATTERN "*.cuh"
)

if(MINKOWSKI_ENABLE_GAUSSIAN)
    install(DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}/MinkowskiEngine/gaussian/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/minkowski/gaussian"
        FILES_MATCHING
            PATTERN "gaussian_cuda.hpp" EXCLUDE
            PATTERN "*.hpp"
            PATTERN "*.h"
            PATTERN "*.hh"
            PATTERN "*.hxx"
            PATTERN "*.inl"
            PATTERN "*.cuh"
    )
endif()

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
