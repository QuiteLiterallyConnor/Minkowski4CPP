/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// On Windows MSVC with CUDA 13.1+, stub out NVTX to avoid version conflicts
#if defined(_MSC_VER) && defined(__CUDACC__)

namespace cudf {
struct libcudf_domain { 
  static constexpr char const* name{"libcudf"};
};

// Stub NVTX range for Windows compatibility
struct thread_range { 
  template<typename... Args> 
  thread_range(Args&&...) {} 
};

}  // namespace cudf

#define CUDF_FUNC_RANGE()

#else

#include "nvtx3.hpp"

namespace cudf {
/**
 * @brief Tag type for libcudf's NVTX domain.
 *
 */
struct libcudf_domain {
  static constexpr char const* name{"libcudf"};  ///< Name of the libcudf domain
};

/**
 * @brief Alias for an NVTX range in the libcudf domain.
 *
 */
using thread_range = ::nvtx3::domain_thread_range<libcudf_domain>;

}  // namespace cudf

/**
 * @brief Convenience macro for generating an NVTX range in the `libcudf` domain
 * from the lifetime of a function.
 *
 * Uses the name of the immediately enclosing function returned by `__func__` to
 * name the range.
 *
 * Example:
 * ```
 * void some_function(){
 *    CUDF_FUNC_RANGE();
 *    ...
 * }
 * ```
 *
 */
#define CUDF_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(cudf::libcudf_domain)

#endif  // Windows stub vs full NVTX
