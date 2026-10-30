#include "gaussian/gaussian_ply.hpp"

#include "gaussian/gaussian_sh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace minkowski {
namespace gaussian {
namespace {

constexpr float kSH_C0 = static_cast<float>(kShC0);
constexpr float kProbabilityEpsilon = 1e-6f;

struct Property {
  std::string name;
  std::string type;
  size_t offset = 0;
  size_t size = 0;
};

struct Element {
  std::string name;
  int64_t count = 0;
  std::vector<Property> properties;
  size_t stride = 0;
  size_t byte_offset = 0;

  const Property *find(const std::string &property) const {
    for (const auto &p : properties)
      if (p.name == property)
        return &p;
    return nullptr;
  }
};

size_t type_size(const std::string &type) {
  if (type == "char" || type == "uchar" || type == "int8" || type == "uint8")
    return 1;
  if (type == "short" || type == "ushort" || type == "int16" ||
      type == "uint16")
    return 2;
  if (type == "int" || type == "uint" || type == "float" || type == "int32" ||
      type == "uint32" || type == "float32")
    return 4;
  if (type == "double" || type == "float64")
    return 8;
  throw std::runtime_error("PLY: unsupported property type '" + type + "'");
}

template <typename T> T read_at(const char *base, size_t offset) {
  T value;
  std::memcpy(&value, base + offset, sizeof(T));
  return value;
}

float read_float(const char *row, const Property &p) {
  if (p.type == "float" || p.type == "float32")
    return read_at<float>(row, p.offset);
  if (p.type == "double" || p.type == "float64")
    return static_cast<float>(read_at<double>(row, p.offset));
  throw std::runtime_error("PLY: property '" + p.name + "' must be a float");
}

float lerp(float a, float b, float t) { return a + (b - a) * t; }

const Property &require(const Element &e, const std::string &name) {
  const Property *p = e.find(name);
  if (!p)
    throw std::runtime_error("PLY: element '" + e.name +
                             "' is missing property '" + name + "'");
  return *p;
}

void normalize_quaternion(float *q) {
  const float n =
      std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n < 1e-12f) {
    q[0] = 1.f;
    q[1] = q[2] = q[3] = 0.f;
    return;
  }
  for (int i = 0; i < 4; ++i)
    q[i] /= n;
}

float logit(float p) {
  p = std::min(std::max(p, kProbabilityEpsilon), 1.f - kProbabilityEpsilon);
  return std::log(p / (1.f - p));
}

// Number of f_rest_* properties -> SH degree, or -1 if the layout is unknown.
int degree_from_rest_count(int64_t count) {
  switch (count) {
  case 0: return 0;
  case 9: return 1;
  case 24: return 2;
  case 45: return 3;
  default: return -1;
  }
}

int64_t count_rest_properties(const Element &e) {
  int64_t count = 0;
  while (e.find("f_rest_" + std::to_string(count)))
    ++count;
  return count;
}

// Compressed SH bytes: 0 -> -4, 255 -> 4, otherwise (v + 0.5) / 256 * 8 - 4.
float decode_sh_byte(uint8_t v) {
  const float t = v == 0 ? 0.f : (v == 255 ? 1.f : (v + 0.5f) / 256.f);
  return t * 8.f - 4.f;
}

struct RawGaussians {
  int64_t n = 0;
  int degree = 0;
  std::vector<float> pos, dc, opacity, scale, rot;
  // File order per vertex: [3][K-1] (channel-major).
  std::vector<float> rest;
};

at::Tensor make_tensor(std::vector<float> &v, at::IntArrayRef shape) {
  return torch::from_blob(v.data(), shape,
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

GaussianModel read_ply(const std::string &path, bool load_sh) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("PLY: cannot open '" + path + "'");

  std::string line;
  std::getline(file, line);
  if (!line.empty() && line.back() == '\r')
    line.pop_back();
  if (line != "ply")
    throw std::runtime_error("PLY: missing magic header in '" + path + "'");

  std::vector<Element> elements;
  bool binary_le = false, header_done = false;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    std::istringstream in(line);
    std::string keyword;
    in >> keyword;
    if (keyword == "format") {
      std::string format;
      in >> format;
      binary_le = format == "binary_little_endian";
    } else if (keyword == "element") {
      Element e;
      in >> e.name >> e.count;
      if (!in || e.count < 0)
        throw std::runtime_error("PLY: malformed element line");
      elements.push_back(std::move(e));
    } else if (keyword == "property") {
      if (elements.empty())
        throw std::runtime_error("PLY: property before element");
      Property p;
      in >> p.type >> p.name;
      if (p.type == "list")
        throw std::runtime_error("PLY: list properties are not supported");
      p.size = type_size(p.type);
      Element &e = elements.back();
      p.offset = e.stride;
      e.stride += p.size;
      e.properties.push_back(std::move(p));
    } else if (keyword == "end_header") {
      header_done = true;
      break;
    }
  }
  if (!header_done)
    throw std::runtime_error("PLY: missing end_header");
  if (!binary_le)
    throw std::runtime_error("PLY: only binary_little_endian is supported");

  size_t cursor = static_cast<size_t>(file.tellg());
  for (auto &e : elements) {
    e.byte_offset = cursor;
    cursor += e.stride * static_cast<size_t>(e.count);
  }
  file.seekg(0, std::ios::end);
  if (static_cast<size_t>(file.tellg()) < cursor)
    throw std::runtime_error("PLY: file is truncated");
  file.clear();

  const Element *vertex = nullptr, *chunk = nullptr, *sh = nullptr;
  for (const auto &e : elements) {
    if (e.name == "vertex")
      vertex = &e;
    else if (e.name == "chunk")
      chunk = &e;
    else if (e.name == "sh")
      sh = &e;
  }
  if (!vertex)
    throw std::runtime_error("PLY: no vertex element");

  auto read_block = [&](const Element &e) {
    std::vector<char> bytes(e.stride * static_cast<size_t>(e.count));
    file.seekg(static_cast<std::streamoff>(e.byte_offset));
    file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file && !bytes.empty())
      throw std::runtime_error("PLY: failed reading element '" + e.name + "'");
    return bytes;
  };

  const int64_t n = vertex->count;
  const std::vector<char> vertex_bytes = read_block(*vertex);
  RawGaussians g;
  g.n = n;
  g.pos.resize(3 * n);
  g.dc.resize(3 * n);
  g.opacity.resize(n);
  g.scale.resize(3 * n);
  g.rot.resize(4 * n);

  if (vertex->find("packed_position")) {
    if (!chunk)
      throw std::runtime_error("PLY: compressed file has no chunk element");
    const int64_t chunk_size = 256;
    if (chunk->count < (n + chunk_size - 1) / chunk_size)
      throw std::runtime_error("PLY: too few chunks for vertex count");
    const std::vector<char> chunk_bytes = read_block(*chunk);

    const char *names[18] = {"min_x",       "min_y",       "min_z",
                             "max_x",       "max_y",       "max_z",
                             "min_scale_x", "min_scale_y", "min_scale_z",
                             "max_scale_x", "max_scale_y", "max_scale_z",
                             "min_r",       "min_g",       "min_b",
                             "max_r",       "max_g",       "max_b"};
    const Property *cp[18];
    for (int i = 0; i < 18; ++i)
      cp[i] = &require(*chunk, names[i]);
    const Property &pp = require(*vertex, "packed_position");
    const Property &pr = require(*vertex, "packed_rotation");
    const Property &ps = require(*vertex, "packed_scale");
    const Property &pc = require(*vertex, "packed_color");

    for (int64_t i = 0; i < n; ++i) {
      const char *row = vertex_bytes.data() + vertex->stride * i;
      const char *ch = chunk_bytes.data() + chunk->stride * (i / chunk_size);
      float c[18];
      for (int k = 0; k < 18; ++k)
        c[k] = read_float(ch, *cp[k]);

      const uint32_t p = read_at<uint32_t>(row, pp.offset);
      g.pos[3 * i + 0] = lerp(c[0], c[3], ((p >> 21) & 2047u) / 2047.f);
      g.pos[3 * i + 1] = lerp(c[1], c[4], ((p >> 11) & 1023u) / 1023.f);
      g.pos[3 * i + 2] = lerp(c[2], c[5], (p & 2047u) / 2047.f);

      // 2-bit index of the omitted (largest) component, then three 10-bit
      // components scaled by sqrt(2).
      const uint32_t r = read_at<uint32_t>(row, pr.offset);
      const float s2 = 1.41421356237f;
      const float a = (((r >> 20) & 1023u) / 1023.f - 0.5f) * s2;
      const float b = (((r >> 10) & 1023u) / 1023.f - 0.5f) * s2;
      const float d = ((r & 1023u) / 1023.f - 0.5f) * s2;
      const float m = std::sqrt(std::max(0.f, 1.f - a * a - b * b - d * d));
      float q[4];
      switch (r >> 30) {
      case 0: q[0] = m; q[1] = a; q[2] = b; q[3] = d; break;
      case 1: q[0] = a; q[1] = m; q[2] = b; q[3] = d; break;
      case 2: q[0] = a; q[1] = b; q[2] = m; q[3] = d; break;
      default: q[0] = a; q[1] = b; q[2] = d; q[3] = m; break;
      }
      normalize_quaternion(q);
      std::copy(q, q + 4, &g.rot[4 * i]);

      // Chunk scale bounds are log-scales.
      const uint32_t s = read_at<uint32_t>(row, ps.offset);
      g.scale[3 * i + 0] = lerp(c[6], c[9], ((s >> 21) & 2047u) / 2047.f);
      g.scale[3 * i + 1] = lerp(c[7], c[10], ((s >> 11) & 1023u) / 1023.f);
      g.scale[3 * i + 2] = lerp(c[8], c[11], (s & 2047u) / 2047.f);

      // Packed color is RGB in [min, max] and alpha (already activated).
      const uint32_t k = read_at<uint32_t>(row, pc.offset);
      const float rgb[3] = {
          lerp(c[12], c[15], ((k >> 24) & 255u) / 255.f),
          lerp(c[13], c[16], ((k >> 16) & 255u) / 255.f),
          lerp(c[14], c[17], ((k >> 8) & 255u) / 255.f)};
      for (int j = 0; j < 3; ++j)
        g.dc[3 * i + j] = (rgb[j] - 0.5f) / kSH_C0;
      g.opacity[i] = logit((k & 255u) / 255.f);
    }

    if (load_sh && sh) {
      const int64_t count = count_rest_properties(*sh);
      g.degree = degree_from_rest_count(count);
      if (g.degree < 0)
        g.degree = 0;
      else if (sh->count != n)
        throw std::runtime_error("PLY: sh element count mismatch");
      else if (count > 0) {
        const std::vector<char> sh_bytes = read_block(*sh);
        std::vector<size_t> offsets(static_cast<size_t>(count));
        for (int64_t j = 0; j < count; ++j)
          offsets[static_cast<size_t>(j)] =
              require(*sh, "f_rest_" + std::to_string(j)).offset;
        g.rest.resize(static_cast<size_t>(n * count));
        for (int64_t i = 0; i < n; ++i)
          for (int64_t j = 0; j < count; ++j)
            g.rest[static_cast<size_t>(i * count + j)] = decode_sh_byte(
                static_cast<uint8_t>(sh_bytes[sh->stride * i +
                                              offsets[static_cast<size_t>(j)]]));
      }
    }
  } else {
    const Property &px = require(*vertex, "x"), &py = require(*vertex, "y"),
                   &pz = require(*vertex, "z");
    const Property &po = require(*vertex, "opacity");
    const Property *sc[3] = {&require(*vertex, "scale_0"),
                             &require(*vertex, "scale_1"),
                             &require(*vertex, "scale_2")};
    const Property *rt[4] = {&require(*vertex, "rot_0"),
                             &require(*vertex, "rot_1"),
                             &require(*vertex, "rot_2"),
                             &require(*vertex, "rot_3")};
    const Property *dc[3] = {vertex->find("f_dc_0"), vertex->find("f_dc_1"),
                             vertex->find("f_dc_2")};

    int64_t rest_count = 0;
    std::vector<const Property *> rest_props;
    if (load_sh) {
      rest_count = count_rest_properties(*vertex);
      g.degree = degree_from_rest_count(rest_count);
      if (g.degree < 0) {
        g.degree = 0;
        rest_count = 0;
      }
      for (int64_t j = 0; j < rest_count; ++j)
        rest_props.push_back(&require(*vertex, "f_rest_" + std::to_string(j)));
      g.rest.resize(static_cast<size_t>(n * rest_count));
    }

    for (int64_t i = 0; i < n; ++i) {
      const char *row = vertex_bytes.data() + vertex->stride * i;
      g.pos[3 * i + 0] = read_float(row, px);
      g.pos[3 * i + 1] = read_float(row, py);
      g.pos[3 * i + 2] = read_float(row, pz);
      for (int j = 0; j < 3; ++j) {
        g.scale[3 * i + j] = read_float(row, *sc[j]);
        g.dc[3 * i + j] = dc[j] ? read_float(row, *dc[j]) : 0.f;
      }
      float q[4];
      for (int j = 0; j < 4; ++j)
        q[j] = read_float(row, *rt[j]);
      normalize_quaternion(q);
      std::copy(q, q + 4, &g.rot[4 * i]);
      g.opacity[i] = read_float(row, po);
      for (int64_t j = 0; j < rest_count; ++j)
        g.rest[static_cast<size_t>(i * rest_count + j)] =
            read_float(row, *rest_props[static_cast<size_t>(j)]);
    }
  }

  const int64_t rest_per_channel = sh_coefficient_count(g.degree) - 1;
  at::Tensor f_rest;
  if (rest_per_channel == 0 || g.rest.empty()) {
    f_rest = at::zeros({n, rest_per_channel, 3}, at::kFloat);
  } else {
    f_rest = make_tensor(g.rest, {n, 3, rest_per_channel})
                 .transpose(1, 2)
                 .contiguous();
  }
  return GaussianModel(make_tensor(g.pos, {n, 3}), make_tensor(g.dc, {n, 3}),
                       f_rest, make_tensor(g.opacity, {n, 1}),
                       make_tensor(g.scale, {n, 3}), make_tensor(g.rot, {n, 4}));
}

} // namespace

GaussianTensor load_gaussian_ply(const std::string &path) {
  return read_ply(path, /*load_sh=*/false).to_gaussian_tensor();
}

GaussianModel load_gaussian_model_ply(const std::string &path) {
  return read_ply(path, /*load_sh=*/true);
}

void save_gaussian_ply(const GaussianModel &model, const std::string &path) {
  const auto cpu = model.to(torch::kCPU);
  const int64_t n = cpu.size();
  const int64_t rest_per_channel = cpu.f_rest().size(1);
  const int64_t rest_count = 3 * rest_per_channel;

  std::ofstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("PLY: cannot write '" + path + "'");

  std::ostringstream header;
  header << "ply\nformat binary_little_endian 1.0\nelement vertex " << n
         << "\n";
  for (const char *name : {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1",
                           "f_dc_2"})
    header << "property float " << name << "\n";
  for (int64_t i = 0; i < rest_count; ++i)
    header << "property float f_rest_" << i << "\n";
  for (const char *name : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0",
                           "rot_1", "rot_2", "rot_3"})
    header << "property float " << name << "\n";
  header << "end_header\n";
  const std::string header_text = header.str();
  file.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));

  const auto normals = at::zeros({n, 3}, at::kFloat);
  const auto rest =
      rest_count == 0
          ? at::zeros({n, 0}, at::kFloat)
          : cpu.f_rest().detach().transpose(1, 2).contiguous().reshape(
                {n, rest_count});
  const auto rows = at::cat({cpu.xyz().detach(), normals, cpu.f_dc().detach(),
                             rest, cpu.opacity_raw().detach(),
                             cpu.scale_raw().detach(),
                             cpu.rotation_raw().detach()},
                            1)
                        .contiguous();
  file.write(reinterpret_cast<const char *>(rows.data_ptr<float>()),
             static_cast<std::streamsize>(rows.numel() * sizeof(float)));
  if (!file)
    throw std::runtime_error("PLY: failed writing '" + path + "'");
}

void save_gaussian_ply(const GaussianTensor &gaussians,
                       const std::string &path) {
  save_gaussian_ply(GaussianModel::from_tensor(gaussians), path);
}

} // namespace gaussian
} // namespace minkowski
