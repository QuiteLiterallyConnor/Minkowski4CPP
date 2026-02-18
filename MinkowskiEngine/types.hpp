#ifndef MINK_CPP_TYPES_HPP
#define MINK_CPP_TYPES_HPP

#include "src/types.hpp"

#include <torch/torch.h>
#include <string>
#include <vector>

namespace minkowski {

using size_type = default_types::size_type;
using index_type = default_types::index_type;
using stride_type = default_types::stride_type;

namespace SparseTensorQuantizationMode {
enum Type {
  RANDOM_SUBSAMPLE = 0,
  UNWEIGHTED_AVERAGE = 1,
  UNWEIGHTED_SUM = 2,
  MAX_POOL = 3,
  NO_QUANTIZATION = 4
};
}

} 

#endif 
