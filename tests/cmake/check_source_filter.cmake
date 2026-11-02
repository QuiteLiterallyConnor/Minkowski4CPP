# Checks MINK_SOURCE_EXCLUDE_REGEX (CMakeLists.txt): test/example sources are excluded by file name, while
# directory names that merely contain "test" or "example" keep every library source.
#   cmake -DMINK_SOURCE_EXCLUDE_REGEX=<regex> -P tests/cmake/check_source_filter.cmake
if(NOT DEFINED MINK_SOURCE_EXCLUDE_REGEX)
    message(FATAL_ERROR "pass -DMINK_SOURCE_EXCLUDE_REGEX=<regex>")
endif()

set(keep
    "C:/Users/me/testing-asan/MinkowskiEngine/src/coordinate_map_cpu.cpp"
    "/home/latest/Minkowski4CPP/MinkowskiEngine/adaptive/ops.cpp"
    "/srv/examples/Minkowski4CPP/MinkowskiEngine/src/coordinate_map_gpu.cu"
    "/w/tests/MinkowskiEngine/gaussian/gaussian_render.cpp"
)
set(drop
    "/w/MinkowskiEngine/src/test_kernel.cpp"
    "/w/MinkowskiEngine/src/kernel_test.cu"
    "/w/MinkowskiEngine/utils/example_usage.cpp"
    "C:/Users/me/testing-asan/MinkowskiEngine/src/my_tests.cpp"
)

set(result ${keep} ${drop})
list(FILTER result EXCLUDE REGEX "${MINK_SOURCE_EXCLUDE_REGEX}")

set(errors "")
foreach(path IN LISTS keep)
    if(NOT path IN_LIST result)
        string(APPEND errors "\n  dropped a library source: ${path}")
    endif()
endforeach()
foreach(path IN LISTS drop)
    if(path IN_LIST result)
        string(APPEND errors "\n  kept a test/example source: ${path}")
    endif()
endforeach()
if(errors)
    message(FATAL_ERROR "MINK_SOURCE_EXCLUDE_REGEX='${MINK_SOURCE_EXCLUDE_REGEX}':${errors}")
endif()
message(STATUS "source filter ok: ${MINK_SOURCE_EXCLUDE_REGEX}")
