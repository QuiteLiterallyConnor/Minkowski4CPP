#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_PLY_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_PLY_HPP

#include "gaussian/gaussian_model.hpp"
#include "gaussian/gaussian_tensor.hpp"

#include <string>

namespace minkowski {
namespace gaussian {

// Reads a binary_little_endian Gaussian-splat PLY file. Supported layouts:
//  - standard 3DGS (x,y,z, f_dc_*, f_rest_*, opacity, scale_*, rot_*), and
//  - PlayCanvas/SuperSplat compressed (chunk + packed_* + sh elements).
// Throws std::runtime_error on missing or malformed files.

// Loads the file as an activated float32 CPU GaussianTensor: opacity =
// sigmoid(raw), scale = exp(raw), color = 0.5 + C0 * f_dc, rotations
// normalized (w, x, y, z). Spherical-harmonic rest coefficients are skipped.
GaussianTensor load_gaussian_ply(const std::string &path);

// Loads the file as raw trainable parameters, including the spherical-harmonic
// rest coefficients (degree 0-3, inferred from the f_rest_* properties).
GaussianModel load_gaussian_model_ply(const std::string &path);

// Writes a standard 3DGS binary_little_endian PLY that other splat viewers
// and load_gaussian_model_ply() can read.
void save_gaussian_ply(const GaussianModel &model, const std::string &path);

// Writes an activated tensor by inverting the activations (SH degree 0).
void save_gaussian_ply(const GaussianTensor &gaussians,
                       const std::string &path);

} // namespace gaussian
} // namespace minkowski

#endif
