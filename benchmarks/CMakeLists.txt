cmake_minimum_required(VERSION 3.24)

project(SphereVolumeBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(MinkowskiEngine CONFIG REQUIRED)

separate_arguments(
    MINK_TORCH_CXX_FLAGS
    NATIVE_COMMAND
    "${TORCH_CXX_FLAGS}"
)

add_executable(sphere_volume_benchmark sphere_volume.cpp sphere_volume_cuda.cu)
target_link_libraries(sphere_volume_benchmark PRIVATE MinkowskiEngine::minkowski_cpp)
if(MSVC)
    target_compile_options(sphere_volume_benchmark
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()
# Keep device arithmetic bit-identical to the host classification.
target_compile_options(sphere_volume_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

# Place the runtime DLLs next to the executable so it runs without editing PATH.
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET sphere_volume_benchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS} $<TARGET_FILE_DIR:sphere_volume_benchmark>
        COMMAND_EXPAND_LISTS
    )
endif()

if(MINK_TORCH_CXX_FLAGS)
    target_compile_options(sphere_volume_benchmark
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${MINK_TORCH_CXX_FLAGS}>
    )
endif()
