#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_PLY_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_PLY_HPP

#include "gaussian/gaussian_tensor.hpp"

#include <string>

namespace minkowski {
namespace gaussian {

// Loads a binary_little_endian Gaussian-splat PLY file as a float32 CPU
// GaussianTensor. Supported layouts:
//  - standard 3DGS (x,y,z, f_dc_*, opacity, scale_*, rot_*), and
//  - PlayCanvas/SuperSplat compressed (chunk + packed_* elements).
// Raw parameters are converted to engine conventions: opacity = sigmoid(raw),
// scale = exp(raw), color = 0.5 + C0 * f_dc, rotations normalized (w, x, y, z).
// Spherical-harmonic rest coefficients are not loaded. Throws
// std::runtime_error on missing or malformed files.
GaussianTensor load_gaussian_ply(const std::string &path);

} // namespace gaussian
} // namespace minkowski

#endif
