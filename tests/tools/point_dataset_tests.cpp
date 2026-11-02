#include "benchmarks/point_dataset.hpp"
#include <chrono>
#include <functional>
#include <iostream>

using nlohmann::json;
namespace fs = std::filesystem;

struct Fixture {
  fs::path dir;
  json m;
  std::vector<unsigned char> bytes;
  Fixture() {
    dir = fs::temp_directory_path() / ("mink-point-data-" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!fs::create_directory(dir)) throw std::runtime_error("Cannot create isolated test fixture");
    bytes = {'M','K','4','P','C','D','1',0};
    add32(10); add32(2);
    for (int label = 0; label < 10; ++label) {
      add32(label);
      for (int i = 0; i < 6; ++i) {
        double value = (label * 6 + i - 30) * 0.01;
        uint64_t bits; std::memcpy(&bits, &value, 8);
        for (int j = 0; j < 8; ++j) bytes.push_back((bits >> (j * 8)) & 255);
      }
    }
    m = {{"format", "minkowski4cpp-points-v1"}, {"dataset", "fixture"},
         {"points_per_sample", 2}, {"preprocessing_seed", 1}, {"class_names", json::array()}};
    for (int c = 0; c < 10; ++c) m["class_names"].push_back("class" + std::to_string(c));
    for (const std::string split : {"train", "test"}) {
      json ids = json::array();
      for (int c = 0; c < 10; ++c) ids.push_back(split + std::to_string(c));
      m["splits"][split] = {{"file", split + ".bin"}, {"samples", 10}, {"per_class", 1},
                            {"ids", ids}, {"fnv1a64", mink_bench::fnv1a64(bytes)}};
    }
    write();
  }
  ~Fixture() { std::error_code error; fs::remove_all(dir, error); }
  void add32(uint32_t v) { for (int i = 0; i < 4; ++i) bytes.push_back((v >> (i * 8)) & 255); }
  void write(bool hash = false) {
    for (const std::string split : {"train", "test"}) {
      if (hash) m["splits"][split]["fnv1a64"] = mink_bench::fnv1a64(bytes);
      std::ofstream file(dir / (split + ".bin"), std::ios::binary);
      file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }
    std::ofstream(dir / "manifest.json") << m.dump();
  }
};

int main() {
  int passed = 0, failed = 0;
  auto test = [&](const std::string &name, const std::function<void()> &fn) {
    try { fn(); ++passed; }
    catch (const std::exception &e) { ++failed; std::cerr << name << ": " << e.what() << '\n'; }
  };
  auto require = [](bool yes) { if (!yes) throw std::runtime_error("Expectation failed"); };
  auto rejected = [&](const std::string &name, const std::function<void(Fixture &)> &mutate, bool hash = false) {
    test(name, [&] {
      Fixture f; mutate(f); f.write(hash);
      bool threw = false;
      try { (void)mink_bench::load_point_dataset(f.dir); } catch (const std::exception &) { threw = true; }
      require(threw);
    });
  };
  test("roundtrip", [&] {
    Fixture f; auto d = mink_bench::load_point_dataset(f.dir);
    require(d.train.size() == 10 && d.test.size() == 10 && d.points_per_sample == 2);
    for (int c = 0; c < 10; ++c) {
      require(d.train[c].label == c && d.test[c].label == c);
      for (int i = 0; i < 6; ++i) require(d.train[c].points[i] == (c * 6 + i - 30) * 0.01);
    }
    require(mink_bench::fnv1a64({'h','e','l','l','o'}) == "a430d84680aabd0b");
  });
  rejected("format", [](Fixture &f) { f.m["format"] = "unknown"; });
  rejected("duplicate class", [](Fixture &f) { f.m["class_names"][1] = f.m["class_names"][0]; });
  rejected("class count", [](Fixture &f) { f.m["class_names"].erase(0); });
  rejected("zero points", [](Fixture &f) { f.m["points_per_sample"] = 0; });
  rejected("negative count", [](Fixture &f) { f.m["splits"]["train"]["samples"] = -1; });
  rejected("fractional count", [](Fixture &f) { f.m["points_per_sample"] = 2.5; });
  rejected("huge count", [](Fixture &f) { f.m["points_per_sample"] = 1000001; });
  rejected("cross split leak", [](Fixture &f) { f.m["splits"]["test"]["ids"][0] = "train0"; });
  rejected("empty id", [](Fixture &f) { f.m["splits"]["train"]["ids"][0] = ""; });
  rejected("path escape", [](Fixture &f) { f.m["splits"]["train"]["file"] = "../train.bin"; });
  rejected("checksum", [](Fixture &f) { f.bytes[25] ^= 1; });
  rejected("truncation", [](Fixture &f) { f.bytes.pop_back(); });
  rejected("trailing bytes", [](Fixture &f) { f.bytes.push_back(0); });
  rejected("header count", [](Fixture &f) { f.bytes[8] = 9; }, true);
  rejected("header point count", [](Fixture &f) { f.bytes[12] = 3; }, true);
  rejected("invalid label", [](Fixture &f) { f.bytes[16] = 10; }, true);
  rejected("unbalanced labels", [](Fixture &f) { f.bytes[16] = 1; }, true);
  for (double v : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), 1.0, -1.001})
    rejected("nonfinite/out-of-grid coordinate", [v](Fixture &f) {
      uint64_t bits; std::memcpy(&bits, &v, 8);
      for (int i = 0; i < 8; ++i) f.bytes[20 + i] = (bits >> (8 * i)) & 255;
    }, true);
  std::cout << passed << " point dataset cases passed; " << failed << " failed\n";
  return failed ? 1 : 0;
}
