#include "gaussian/gaussian_ply.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace minkowski {
namespace gaussian {
namespace {

constexpr float kSH_C0 = 0.28209479177387814f;

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
  const float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] +
                            q[3] * q[3]);
  if (n < 1e-12f) {
    q[0] = 1.f;
    q[1] = q[2] = q[3] = 0.f;
    return;
  }
  for (int i = 0; i < 4; ++i)
    q[i] /= n;
}

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

} // namespace

GaussianTensor load_gaussian_ply(const std::string &path) {
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

  const auto data_start = static_cast<size_t>(file.tellg());
  size_t cursor = data_start;
  for (auto &e : elements) {
    e.byte_offset = cursor;
    cursor += e.stride * static_cast<size_t>(e.count);
  }

  file.seekg(0, std::ios::end);
  const auto file_size = static_cast<size_t>(file.tellg());
  if (file_size < cursor)
    throw std::runtime_error("PLY: file is truncated");

  const Element *vertex = nullptr, *chunk = nullptr;
  for (const auto &e : elements) {
    if (e.name == "vertex")
      vertex = &e;
    else if (e.name == "chunk")
      chunk = &e;
  }
  if (!vertex)
    throw std::runtime_error("PLY: no vertex element");

  const int64_t n = vertex->count;
  std::vector<char> vertex_bytes(vertex->stride * static_cast<size_t>(n));
  file.seekg(static_cast<std::streamoff>(vertex->byte_offset));
  file.read(vertex_bytes.data(),
            static_cast<std::streamsize>(vertex_bytes.size()));
  if (!file && !vertex_bytes.empty())
    throw std::runtime_error("PLY: failed reading vertex data");

  std::vector<float> pos(3 * n), rot(4 * n), scl(3 * n), opa(n), col(3 * n);

  if (vertex->find("packed_position")) {
    if (!chunk)
      throw std::runtime_error("PLY: compressed file has no chunk element");
    const int64_t chunk_size = 256;
    if (chunk->count < (n + chunk_size - 1) / chunk_size)
      throw std::runtime_error("PLY: too few chunks for vertex count");
    std::vector<char> chunk_bytes(chunk->stride *
                                  static_cast<size_t>(chunk->count));
    file.clear();
    file.seekg(static_cast<std::streamoff>(chunk->byte_offset));
    file.read(chunk_bytes.data(),
              static_cast<std::streamsize>(chunk_bytes.size()));
    if (!file)
      throw std::runtime_error("PLY: failed reading chunk data");

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
      pos[3 * i + 0] = lerp(c[0], c[3], ((p >> 21) & 2047u) / 2047.f);
      pos[3 * i + 1] = lerp(c[1], c[4], ((p >> 11) & 1023u) / 1023.f);
      pos[3 * i + 2] = lerp(c[2], c[5], (p & 2047u) / 2047.f);

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
      std::copy(q, q + 4, &rot[4 * i]);

      const uint32_t s = read_at<uint32_t>(row, ps.offset);
      scl[3 * i + 0] = std::exp(lerp(c[6], c[9], ((s >> 21) & 2047u) / 2047.f));
      scl[3 * i + 1] = std::exp(lerp(c[7], c[10], ((s >> 11) & 1023u) / 1023.f));
      scl[3 * i + 2] = std::exp(lerp(c[8], c[11], (s & 2047u) / 2047.f));

      const uint32_t k = read_at<uint32_t>(row, pc.offset);
      const float dc[3] = {
          lerp(c[12], c[15], ((k >> 24) & 255u) / 255.f),
          lerp(c[13], c[16], ((k >> 16) & 255u) / 255.f),
          lerp(c[14], c[17], ((k >> 8) & 255u) / 255.f)};
      for (int j = 0; j < 3; ++j)
        col[3 * i + j] = 0.5f + kSH_C0 * dc[j];
      opa[i] = (k & 255u) / 255.f;
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
    for (int64_t i = 0; i < n; ++i) {
      const char *row = vertex_bytes.data() + vertex->stride * i;
      pos[3 * i + 0] = read_float(row, px);
      pos[3 * i + 1] = read_float(row, py);
      pos[3 * i + 2] = read_float(row, pz);
      for (int j = 0; j < 3; ++j) {
        scl[3 * i + j] = std::exp(read_float(row, *sc[j]));
        col[3 * i + j] =
            dc[j] ? 0.5f + kSH_C0 * read_float(row, *dc[j]) : 0.5f;
      }
      float q[4];
      for (int j = 0; j < 4; ++j)
        q[j] = read_float(row, *rt[j]);
      normalize_quaternion(q);
      std::copy(q, q + 4, &rot[4 * i]);
      opa[i] = sigmoid(read_float(row, po));
    }
  }

  const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  auto make = [&](std::vector<float> &v, int64_t cols) {
    return torch::from_blob(v.data(), {n, cols}, opts).clone();
  };
  return GaussianTensor(make(pos, 3), make(rot, 4), make(scl, 3),
                        make(opa, 1), make(col, 3));
}

} // namespace gaussian
} // namespace minkowski
