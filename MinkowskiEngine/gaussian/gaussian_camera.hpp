#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_CAMERA_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_CAMERA_HPP

#include <torch/torch.h>

#include <array>
#include <cmath>

namespace minkowski {
namespace gaussian {

// Pinhole camera using the OpenCV/COLMAP convention: +x right, +y down,
// +z forward. Pixel (i, j) has its center at (i + 0.5, j + 0.5).
struct Camera {
  int64_t width = 0;
  int64_t height = 0;
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
  // [4, 4] float32 world-to-camera transform.
  at::Tensor world_to_camera;
  double near_plane = 0.01;

  at::Tensor rotation() const {
    return world_to_camera.slice(0, 0, 3).slice(1, 0, 3);
  }
  at::Tensor translation() const {
    return world_to_camera.slice(0, 0, 3).select(1, 3);
  }
  // Camera center in world coordinates.
  at::Tensor position() const {
    return -at::matmul(rotation().transpose(0, 1), translation());
  }

  static Camera from_fov(int64_t width, int64_t height, double fov_x_radians,
                         at::Tensor world_to_camera) {
    TORCH_CHECK(width > 0 && height > 0, "camera size must be positive");
    TORCH_CHECK(world_to_camera.sizes() == at::IntArrayRef({4, 4}),
                "world_to_camera must have shape [4, 4]");
    Camera camera;
    camera.width = width;
    camera.height = height;
    camera.fx = 0.5 * static_cast<double>(width) / std::tan(0.5 * fov_x_radians);
    camera.fy = camera.fx;
    camera.cx = 0.5 * static_cast<double>(width);
    camera.cy = 0.5 * static_cast<double>(height);
    camera.world_to_camera =
        world_to_camera.to(at::kFloat).to(at::kCPU).contiguous();
    return camera;
  }

  static Camera look_at(const std::array<double, 3> &eye,
                        const std::array<double, 3> &target,
                        const std::array<double, 3> &up, int64_t width,
                        int64_t height, double fov_x_radians) {
    using V = std::array<double, 3>;
    auto sub = [](const V &a, const V &b) {
      return V{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    auto cross = [](const V &a, const V &b) {
      return V{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
               a[0] * b[1] - a[1] * b[0]};
    };
    auto normalize = [](const V &a) {
      const double n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
      TORCH_CHECK(n > 1e-12, "look_at received a degenerate direction");
      return V{a[0] / n, a[1] / n, a[2] / n};
    };
    const V forward = normalize(sub(target, eye));
    const V right = normalize(cross(V{-up[0], -up[1], -up[2]}, forward));
    const V down = cross(forward, right);
    auto matrix = at::zeros({4, 4}, at::kDouble);
    auto a = matrix.accessor<double, 2>();
    const V *rows[3] = {&right, &down, &forward};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c)
        a[r][c] = (*rows[r])[c];
      a[r][3] = -((*rows[r])[0] * eye[0] + (*rows[r])[1] * eye[1] +
                  (*rows[r])[2] * eye[2]);
    }
    a[3][3] = 1.0;
    return from_fov(width, height, fov_x_radians, matrix);
  }
};

} // namespace gaussian
} // namespace minkowski

#endif
