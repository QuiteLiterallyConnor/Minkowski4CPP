#pragma once

// External benchmark data, deliberately independent of LibTorch for fast parser tests.
#include "third_party/json.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace mink_bench {
struct PointSample {
  std::vector<double> points;
  int label;
};
struct PointDataset {
  nlohmann::json manifest;
  std::array<std::string, 10> class_names;
  uint32_t points_per_sample;
  std::vector<PointSample> train, test;
};

inline uint32_t read_u32(const unsigned char *p) {
  return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
inline double read_f64(const unsigned char *p) {
  static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                "Point datasets require IEEE754 binary64");
  uint64_t bits = 0;
  for (int i = 0; i < 8; ++i)
    bits |= uint64_t(p[i]) << (8 * i);
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
inline std::string fnv1a64(const std::vector<unsigned char> &bytes) {
  uint64_t h = 14695981039346656037ULL;
  for (const auto b : bytes)
    h = (h ^ b) * 1099511628211ULL;
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << h;
  return out.str();
}
inline uint32_t positive_count(const nlohmann::json &j, const char *field) {
  const auto &value = j.at(field);
  if (!value.is_number_integer() || value.get<int64_t>() <= 0 || value.get<int64_t>() > 1000000)
    throw std::invalid_argument(std::string("Invalid dataset count: ") + field);
  return value.get<uint32_t>();
}

inline PointDataset load_point_dataset(const std::filesystem::path &directory) {
  std::ifstream in(directory / "manifest.json");
  if (!in)
    throw std::invalid_argument("Cannot open dataset manifest");
  PointDataset result;
  in >> result.manifest;
  const auto &m = result.manifest;
  if (m.at("format") != "minkowski4cpp-points-v1" || !m.at("dataset").is_string() ||
      m.at("dataset").get<std::string>().empty())
    throw std::invalid_argument("Unsupported point dataset format/name");
  const auto &classes = m.at("class_names");
  if (!classes.is_array() || classes.size() != result.class_names.size())
    throw std::invalid_argument("This benchmark requires ten classes");
  std::unordered_set<std::string> names, ids;
  for (size_t i = 0; i < result.class_names.size(); ++i) {
    auto name = classes.at(i).get<std::string>();
    if (name.empty() || !names.insert(name).second)
      throw std::invalid_argument("Dataset class names must be unique and nonempty");
    result.class_names[i] = name;
  }
  result.points_per_sample = positive_count(m, "points_per_sample");
  auto load_split = [&](const char *split) {
    const auto &s = m.at("splits").at(split);
    const auto filename = std::string(split) + ".bin";
    if (s.at("file") != filename)
      throw std::invalid_argument("Dataset split file must use its fixed local filename");
    const auto count = positive_count(s, "samples");
    const auto per_class = positive_count(s, "per_class");
    if (uint64_t(per_class) * 10 != count)
      throw std::invalid_argument("Dataset split must have the declared count per class");
    const auto &sample_ids = s.at("ids");
    if (!sample_ids.is_array() || sample_ids.size() != count)
      throw std::invalid_argument("Dataset sample identities do not match split size");
    for (const auto &id : sample_ids) {
      auto text = id.get<std::string>();
      if (text.empty() || !ids.insert(text).second)
        throw std::invalid_argument("Duplicate or overlapping train/test sample identity");
    }
    std::ifstream file(directory / filename, std::ios::binary | std::ios::ate);
    const uint64_t expected = 16 + uint64_t(count) * (4 + 24ULL * result.points_per_sample);
    if (!file || file.tellg() < 0 || uint64_t(file.tellg()) != expected)
      throw std::invalid_argument("Dataset split size does not match its manifest");
    if (expected > uint64_t(std::numeric_limits<size_t>::max()) ||
        expected > uint64_t(std::numeric_limits<std::streamsize>::max()))
      throw std::invalid_argument("Dataset exceeds addressable storage");
    std::vector<unsigned char> bytes(static_cast<size_t>(expected));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(expected));
    if (!file || std::memcmp(bytes.data(), "MK4PCD1\0", 8) != 0 ||
        read_u32(bytes.data() + 8) != count || read_u32(bytes.data() + 12) != result.points_per_sample)
      throw std::invalid_argument("Dataset header is invalid or truncated");
    if (s.at("fnv1a64") != fnv1a64(bytes))
      throw std::invalid_argument("Dataset split checksum mismatch");
    std::vector<PointSample> samples;
    samples.reserve(count);
    std::array<uint32_t, 10> labels{};
    size_t cursor = 16;
    for (uint32_t i = 0; i < count; ++i) {
      const auto label = read_u32(bytes.data() + cursor);
      cursor += 4;
      if (label >= labels.size())
        throw std::invalid_argument("Dataset label is out of range");
      ++labels[label];
      PointSample sample;
      sample.label = static_cast<int>(label);
      sample.points.resize(size_t(result.points_per_sample) * 3);
      for (auto &coordinate : sample.points) {
        coordinate = read_f64(bytes.data() + cursor);
        cursor += 8;
        if (!std::isfinite(coordinate) || coordinate < -1 || coordinate >= 1)
          throw std::invalid_argument("Dataset points must be finite and within [-1, 1)");
      }
      samples.push_back(std::move(sample));
    }
    for (auto n : labels)
      if (n != per_class)
        throw std::invalid_argument("Dataset label histogram differs from its manifest");
    return samples;
  };
  result.train = load_split("train");
  result.test = load_split("test");
  return result;
}
} // namespace mink_bench
