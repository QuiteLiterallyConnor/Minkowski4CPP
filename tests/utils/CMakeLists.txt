cmake_minimum_required(VERSION 3.24)

# This target can also run without configuring LibTorch or CUDA:
# cmake -S tests/utils -B build-test-framework
# cmake --build build-test-framework && ctest --test-dir build-test-framework
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(MinkowskiTestFrameworkTests LANGUAGES CXX)
    enable_testing()
endif()

add_executable(test_framework_contract test_framework_contract.cpp)
target_compile_features(test_framework_contract PRIVATE cxx_std_20)
add_test(NAME test_framework_contract COMMAND test_framework_contract)
