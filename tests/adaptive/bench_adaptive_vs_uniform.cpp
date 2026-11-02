// Adaptive vs uniform voxelization benchmark (not part of the correctness suite).
//
// Compares SPARSE uniform voxelizations (coarsest level, finest level, and a voxel size whose
// occupied-voxel count matches an adaptive tree) against adaptive trees (ROI refinement,
// refine_by_density, refine_by_variance) on several deterministic point sets, and reports
// voxel counts, bytes, kernel-map build time, convolution forward / forward+backward time
// (adaptive engine for every method, MinkowskiEngine sparse convolution for the uniform ones),
// held-out interpolation error against the reference values, peak RAM / VRAM, and the cost of
// collapsing a tree (coarsen to level 0) for dynamic-topology use. a_fnorm refines by the norm of
// a fixed random convolution response (stand-in for a learned importance criterion).
//
//   adaptive_vs_uniform_bench [--datasets uniform,clustered,shell,ply,scan]
//       [--methods u_coarse,u_fine,u_matched,a_roi,a_dens4,a_dens16,a_dens64,a_var,a_fnorm]
//       [--devices cpu,cuda] [--depth 4] [--base 8] [--points 100000] [--channels 16]
//       [--queries 20000] [--ml-queries 2000] [--warmup 2] [--reps 5] [--seed 0]
//       [--var-rel 0.05] [--data-dir DIR] [--jsonl FILE] [--no-mink] [--sphere]
//
// One JSON object per (dataset, method, device) is printed to stdout (and appended to --jsonl).
// Peak RAM is the process peak working set, so run one case per process for per-method peaks
// (tests/adaptive/run_adaptive_vs_uniform.py does that). Peak VRAM uses the CUDA caching
// allocator statistics, reset at the start of every case. Times are median / min / max over
// --reps timed runs after --warmup untimed runs; CUDA is synchronized before every clock read.
// --sphere instead analyses voxelize_sphere (reported vs geometric error).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/modules/convolution.hpp"
#include "MinkowskiEngine/sparse_tensor.hpp"
#include "MinkowskiEngine/utils/sphere.hpp"
#include "MinkowskiEngine/utils/sphere_geometry.hpp"
#ifdef MINKOWSKI_ENABLE_GAUSSIAN
#include "MinkowskiEngine/gaussian/gaussian_model.hpp"
#include "MinkowskiEngine/gaussian/gaussian_ply.hpp"
#endif
#include "third_party/json.hpp"

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#ifndef MINK_TEST_DATA_DIR
#define MINK_TEST_DATA_DIR "tests/data"
#endif

namespace ad = minkowski::adaptive;
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
  std::vector<std::string> datasets{"uniform", "clustered", "shell", "ply", "scan"};
  std::vector<std::string> methods{"u_coarse", "u_fine",   "u_matched", "a_roi",
                                   "a_dens4",  "a_dens16", "a_dens64",  "a_var"};
  std::vector<std::string> devices{"cpu"};
  uint32_t depth = 4;
  double base = 8.0;
  int64_t points = 100000;
  int64_t channels = 16;
  int64_t queries = 20000;
  int64_t ml_queries = 2000;
  int warmup = 2;
  int reps = 5;
  uint64_t seed = 0;
  double var_rel = 0.05;
  bool drop_empty = false; // also run the a_* methods with drop_empty (if supported)
  bool mink = true;
  bool sphere = false;
  bool interleave = false; // also time fwd and fwd+bwd alternately within one loop
  std::string data_dir = MINK_TEST_DATA_DIR;
  std::string jsonl;
};

std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty())
      out.push_back(item);
  return out;
}

Args parse(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << k << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--datasets") a.datasets = split(next());
    else if (k == "--methods") a.methods = split(next());
    else if (k == "--devices") a.devices = split(next());
    else if (k == "--depth") a.depth = static_cast<uint32_t>(std::stoul(next()));
    else if (k == "--base") a.base = std::stod(next());
    else if (k == "--points") a.points = std::stoll(next());
    else if (k == "--channels") a.channels = std::stoll(next());
    else if (k == "--queries") a.queries = std::stoll(next());
    else if (k == "--ml-queries") a.ml_queries = std::stoll(next());
    else if (k == "--warmup") a.warmup = std::stoi(next());
    else if (k == "--reps") a.reps = std::stoi(next());
    else if (k == "--seed") a.seed = std::stoull(next());
    else if (k == "--var-rel") a.var_rel = std::stod(next());
    else if (k == "--data-dir") a.data_dir = next();
    else if (k == "--jsonl") a.jsonl = next();
    else if (k == "--drop-empty") a.drop_empty = true;
    else if (k == "--no-mink") a.mink = false;
    else if (k == "--sphere") a.sphere = true;
    else if (k == "--interleave") a.interleave = true;
    else {
      std::cerr << "unknown argument " << k << "\n";
      std::exit(2);
    }
  }
  if (a.reps < 1)
    a.reps = 1;
  return a;
}

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void sync(bool cuda) {
#ifdef USE_CUDA
  if (cuda)
    torch::cuda::synchronize();
#else
  (void)cuda;
#endif
}

struct Stat {
  double med = 0, min = 0, max = 0;
};

json to_json(const Stat &s) { return json{{"med", s.med}, {"min", s.min}, {"max", s.max}}; }

// setup() runs untimed before every body(); `warmup` untimed rounds, then `reps` timed ones.
template <class S, class B>
Stat measure(int warmup, int reps, bool cuda, S &&setup, B &&body) {
  for (int i = 0; i < warmup; ++i) {
    setup();
    body();
    sync(cuda);
  }
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    setup();
    sync(cuda);
    const auto t0 = Clock::now();
    body();
    sync(cuda);
    t.push_back(ms_since(t0));
  }
  std::sort(t.begin(), t.end());
  const size_t n = t.size();
  Stat s;
  s.med = n % 2 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
  s.min = t.front();
  s.max = t.back();
  return s;
}

double peak_ram_mb() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    return double(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
  return -1;
#else
  rusage r{};
  getrusage(RUSAGE_SELF, &r);
  return double(r.ru_maxrss) / 1024.0;
#endif
}

// ---------------------------------------------------------------------------------------------
// Datasets: points in [0, 256)^3 (double), reference values [P, V] (double).
// ---------------------------------------------------------------------------------------------
constexpr double kDomain = 256.0;
constexpr double kPi = 3.14159265358979323846;

struct Dataset {
  std::string name;
  at::Tensor train_pts, train_vals, test_pts, test_vals;
  std::vector<double> roi_center; // dataset-specific region of interest (box centre)
  std::string note;
};

// Smooth multi-scale reference field: 4 channels, wavelengths 64, 32, 16, 8.
at::Tensor field(const at::Tensor &p) {
  const double lambda[4] = {64.0, 32.0, 16.0, 8.0};
  const double dir[4][3] = {{1, 0, 0}, {0, 1, 0}, {0.6, 0.8, 0}, {0.48, 0.6, 0.64}};
  std::vector<at::Tensor> ch;
  for (int c = 0; c < 4; ++c) {
    auto proj = p.select(1, 0) * dir[c][0] + p.select(1, 1) * dir[c][1] + p.select(1, 2) * dir[c][2];
    ch.push_back(torch::sin(proj * (2.0 * kPi / lambda[c]) + double(c)));
  }
  return torch::stack(ch, 1);
}

// Every 10th point is held out for the interpolation error.
void split_train_test(Dataset &d, const at::Tensor &pts, const at::Tensor &vals) {
  const int64_t n = pts.size(0);
  auto idx = torch::arange(n, torch::kInt64);
  auto test_mask = (idx % 10) == 9;
  d.train_pts = pts.index({~test_mask}).contiguous();
  d.train_vals = vals.index({~test_mask}).contiguous();
  d.test_pts = pts.index({test_mask}).contiguous();
  d.test_vals = vals.index({test_mask}).contiguous();
}

at::Tensor clamp_domain(const at::Tensor &p) { return p.clamp(0.0, kDomain - 1e-6); }

// Robust affine fit into the domain: per-axis 0.5 / 99.5 % quantiles, uniform scale.
at::Tensor normalize_into_domain(const at::Tensor &p, at::Tensor *keep_out) {
  auto lo = std::get<0>(p.kthvalue(std::max<int64_t>(1, p.size(0) / 200), 0));
  auto hi = std::get<0>(p.kthvalue(std::max<int64_t>(1, p.size(0) - p.size(0) / 200), 0));
  auto keep = ((p >= lo) & (p <= hi)).all(1);
  const double extent = (hi - lo).max().item<double>();
  auto q = (p - lo) * ((kDomain - 1e-3) / extent);
  if (keep_out)
    *keep_out = keep;
  return q;
}

std::vector<double> densest_cell_center(const at::Tensor &pts, double cell) {
  auto c = torch::floor(pts / cell).to(torch::kInt64);
  const int64_t g = int64_t(std::ceil(kDomain / cell));
  auto key = (c.select(1, 0) * g + c.select(1, 1)) * g + c.select(1, 2);
  auto counts = torch::bincount(key);
  const int64_t best = counts.argmax().item<int64_t>();
  return {(double(best / (g * g)) + 0.5) * cell, (double((best / g) % g) + 0.5) * cell,
          (double(best % g) + 0.5) * cell};
}

Dataset make_dataset(const std::string &name, const Args &a) {
  torch::manual_seed(a.seed);
  Dataset d;
  d.name = name;
  const int64_t n = a.points;
  const auto f64 = torch::kFloat64;
  at::Tensor pts, vals;
  if (name == "uniform") {
    pts = torch::rand({n, 3}, f64) * kDomain;
    vals = field(pts);
    d.note = "uniform in [0,256)^3, multi-scale sine field";
  } else if (name == "clustered") {
    const int64_t k = 12;
    auto centers = torch::rand({k, 3}, f64) * (kDomain - 64.0) + 32.0;
    const double sig[3] = {2.0, 4.0, 8.0};
    const int64_t nc = n * 9 / 10;
    auto which = torch::arange(nc, torch::kInt64) % k;
    auto sigma = torch::empty({nc, 1}, f64);
    for (int64_t j = 0; j < k; ++j)
      sigma.index_put_({which == j}, sig[j % 3]);
    auto cl = centers.index_select(0, which) + torch::randn({nc, 3}, f64) * sigma;
    auto bg = torch::rand({n - nc, 3}, f64) * kDomain;
    pts = clamp_domain(torch::cat({cl, bg}));
    vals = field(pts);
    d.note = "12 Gaussian clusters (sigma 2/4/8) = 90%, uniform background 10%";
  } else if (name == "shell") {
    auto dir = torch::randn({n, 3}, f64);
    dir = dir / dir.pow(2).sum(1, true).sqrt();
    auto r = 100.0 + 0.5 * torch::randn({n, 1}, f64);
    pts = clamp_domain(dir * r + 128.0);
    vals = field(pts);
    d.note = "sphere surface r=100 centre 128, radial noise sigma 0.5";
  } else if (name == "ply") {
#ifdef MINKOWSKI_ENABLE_GAUSSIAN
    const auto path = (std::filesystem::path(a.data_dir) / "test.ply").string();
    auto model = minkowski::gaussian::load_gaussian_model_ply(path);
    auto p = model.positions().to(f64).cpu();
    auto col = model.colors().to(f64).cpu();
    at::Tensor keep;
    auto q = normalize_into_domain(p, &keep);
    q = q.index({keep});
    col = col.index({keep});
    if (q.size(0) > n) {
      auto perm = torch::randperm(q.size(0), torch::kInt64).slice(0, 0, n);
      q = q.index_select(0, perm);
      col = col.index_select(0, perm);
    }
    pts = clamp_domain(q);
    vals = col;
    d.note = "tests/data/test.ply Gaussian centres (" + std::to_string(p.size(0)) +
             " total), 0.5-99.5% quantile box scaled into [0,256)^3, values = RGB from SH DC";
#else
    throw std::runtime_error("ply dataset needs MINKOWSKI_ENABLE_GAUSSIAN");
#endif
  } else if (name == "scan") {
    // tests/data/voxels/*.json: multi-camera occupancy frames (1 m voxels, intensity) with a
    // moving target. Points are jittered inside each exported voxel; all frames are stacked.
    std::vector<double> xyz, inten;
    std::vector<double> target;
    std::vector<std::string> files;
    for (const auto &e : std::filesystem::directory_iterator(std::filesystem::path(a.data_dir) / "voxels"))
      if (e.path().extension() == ".json")
        files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    for (const auto &f : files) {
      std::ifstream in(f);
      auto j = json::parse(in);
      const double vs = j["grid_info"].value("voxel_size_m", 1.0);
      for (const auto &v : j["voxels"]) {
        const auto c = v["coordinates"];
        for (int k = 0; k < 3; ++k)
          xyz.push_back((c[k].get<double>()) * vs);
        inten.push_back(v.value("intensity", 0.0));
      }
      if (target.empty() && j.contains("targets") && !j["targets"].empty()) {
        // Target position is metric; convert to grid units via origin/voxel size.
        const auto &gi = j["grid_info"];
        for (int k = 0; k < 3; ++k)
          target.push_back((j["targets"][0]["position_m"][k].get<double>() -
                            gi["origin_m"][k].get<double>()));
      }
    }
    const int64_t m = int64_t(inten.size());
    auto raw = torch::from_blob(xyz.data(), {m, 3}, f64).clone();
    auto it = torch::from_blob(inten.data(), {m, 1}, f64).clone();
    raw = raw + torch::rand({m, 3}, f64); // jitter inside the 1 m voxel
    // Scale grid extent (not quantiles: the scene is a full capture volume) into the domain.
    auto lo = std::get<0>(raw.min(0));
    auto hi = std::get<0>(raw.max(0));
    const double scale = (kDomain - 1e-3) / (hi - lo).max().item<double>();
    pts = clamp_domain((raw - lo) * scale);
    vals = it;
    if (pts.size(0) > n) {
      auto perm = torch::randperm(pts.size(0), torch::kInt64).slice(0, 0, n);
      pts = pts.index_select(0, perm);
      vals = vals.index_select(0, perm);
    }
    if (target.size() == 3) {
      for (int k = 0; k < 3; ++k)
        target[k] = (target[k] - lo[k].item<double>()) * scale;
      d.roi_center = target;
    }
    d.note = std::to_string(files.size()) + " frames of tests/data/voxels (" + std::to_string(m) +
             " occupied 1 m voxels, jittered), scale " + std::to_string(scale) +
             ", values = intensity, ROI = first target";
  } else {
    throw std::runtime_error("unknown dataset " + name);
  }
  split_train_test(d, pts, vals);
  if (d.roi_center.size() != 3)
    d.roi_center = densest_cell_center(d.train_pts, a.base);
  return d;
}

// ---------------------------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------------------------
ad::AdaptiveVoxelConfig config(double base, uint32_t depth) {
  ad::AdaptiveVoxelConfig c;
  c.base_voxel_size = {base, base, base};
  const double m = std::ldexp(base, -int(depth));
  c.min_voxel_size = {m, m, m};
  c.origin = {0.0, 0.0, 0.0};
  c.max_depth = depth;
  return c;
}

int64_t occupied_count(const at::Tensor &pts, double s) {
  auto c = torch::floor(pts / s).to(torch::kInt64);
  const int64_t g = int64_t(std::ceil(kDomain / s)) + 1;
  auto key = (c.select(1, 0) * g + c.select(1, 1)) * g + c.select(1, 2);
  return std::get<0>(at::_unique(key, false, false)).numel();
}

// Uniform voxel size whose occupied-voxel count is closest to `target` (bisection on log s).
double matched_voxel_size(const at::Tensor &pts, int64_t target, double lo, double hi) {
  double best = hi;
  int64_t best_err = INT64_MAX;
  for (int it = 0; it < 40; ++it) {
    const double mid = std::sqrt(lo * hi);
    const int64_t n = occupied_count(pts, mid);
    const int64_t err = std::llabs(n - target);
    if (err < best_err) {
      best_err = err;
      best = mid;
    }
    if (n > target)
      lo = mid; // too many voxels -> larger cells
    else
      hi = mid;
    if (hi / lo < 1.0 + 1e-6)
      break;
  }
  return best;
}

struct Built {
  std::optional<ad::AdaptiveSparseTensor> t;
  bool uniform = false;
  double voxel_size = 0; // uniform only
  double refine_threshold = 0;
  std::vector<double> roi_lo, roi_hi;
};

// Leaf row (+1) of every point, 0 when no leaf contains it.
at::Tensor leaf_of(const ad::AdaptiveSparseTensor &t, const at::Tensor &pts) {
  auto ids = torch::arange(1, t.size() + 1, torch::kFloat64).unsqueeze(1);
  return ad::interpolate(t.with_features(ids), pts, ad::InterpMode::Nearest)
      .squeeze(1)
      .round()
      .to(torch::kInt64);
}

// After refinement the children replicate the level-0 parent value; recompute leaf values
// as the mean of the source points inside each leaf, keeping the replicated value for
// leaves that contain no source point (the fairest fill for keep-empty refinement).
void revoxelize_keep_parent(ad::AdaptiveSparseTensor &t, const at::Tensor &train_pts) {
  auto replicated = t.features();
  auto counts = torch::bincount(leaf_of(t, train_pts), {}, t.size() + 1).slice(0, 1);
  t.revoxelize_from_points();
  auto has = (counts > 0).unsqueeze(1);
  t.set_features(torch::where(has, t.features(), replicated));
}

Built build_method(const std::string &method, const Dataset &d, const Args &a,
                   int64_t matched_target, bool drop_empty) {
  Built b;
  const auto &P = d.train_pts;
  const auto &V = d.train_vals;
  if (method == "u_coarse" || method == "u_fine" || method == "u_matched") {
    b.uniform = true;
    if (method == "u_coarse")
      b.voxel_size = a.base;
    else if (method == "u_fine")
      b.voxel_size = std::ldexp(a.base, -int(a.depth));
    else
      b.voxel_size = matched_voxel_size(P, matched_target, std::ldexp(a.base, -int(a.depth) - 1),
                                        a.base * 2.0);
    b.t = ad::AdaptiveSparseTensor::from_points(P, V, config(b.voxel_size, 0));
    return b;
  }
  b.t = ad::AdaptiveSparseTensor::from_points(P, V, config(a.base, a.depth));
#ifdef MINK_ADAPTIVE_HAS_DROP_EMPTY
#define MINK_DROP , drop_empty
#else
#define MINK_DROP
  if (drop_empty)
    throw std::runtime_error("this build has no drop_empty refinement");
#endif
  if (method == "a_roi") {
    const double h = 2.0 * a.base; // ROI box: 4 x 4 x 4 level-0 cells
    ad::AxisAlignedRegion r;
    for (int k = 0; k < 3; ++k) {
      r.lo.push_back(d.roi_center[k] - h);
      r.hi.push_back(d.roi_center[k] + h);
    }
    b.roi_lo = r.lo;
    b.roi_hi = r.hi;
    b.t->refine(r, a.depth, ad::RefinePolicy::Replicate MINK_DROP);
  } else if (method.rfind("a_dens", 0) == 0) {
    const int64_t k = std::stoll(method.substr(6));
    b.refine_threshold = double(k);
    b.t->refine_by_density(k MINK_DROP);
  } else if (method == "a_fnorm") {
    // Feature-response importance (stand-in for a learned criterion): a fixed random 3^3 conv
    // over the current leaf values; each pass refines the leaves whose response norm is in the
    // top 10 % by one level, then re-voxelizes, for max_depth passes.
    torch::manual_seed(a.seed + 7);
    const int64_t V = d.train_vals.size(1);
    auto w = torch::randn({27, V, V}, torch::kFloat64) / std::sqrt(27.0 * double(V));
    for (uint32_t pass = 0; pass < a.depth; ++pass) {
      revoxelize_keep_parent(*b.t, P);
      auto y = ad::adaptive_convolution(*b.t, w, ad::AdaptiveKernelConfig{});
      auto score = y.features().pow(2).sum(1).sqrt().contiguous();
      const double thr = torch::quantile(score, 0.9).item<double>();
      b.refine_threshold = thr;
      const double *sp = score.data_ptr<double>();
      const ad::AdaptiveSparseTensor *tp = &*b.t;
      const uint64_t version = b.t->topology_version();
      // Only the first refine_if pass (the topology the scores belong to) selects leaves.
      b.t->refine_if(
          [&](int64_t i, const std::array<double, 4> &, const std::array<double, 4> &, uint32_t) {
            return tp->topology_version() == version && sp[i] > thr;
          } MINK_DROP);
      ad::KernelMapCache::instance().clear();
    }
  } else if (method == "a_var") {
    const double var = d.train_vals.var(0).mean().item<double>();
    b.refine_threshold = a.var_rel * var;
    b.t->refine_by_variance(b.refine_threshold MINK_DROP);
  } else {
    throw std::runtime_error("unknown method " + method);
  }
#undef MINK_DROP
  revoxelize_keep_parent(*b.t, P);
  return b;
}

// ---------------------------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------------------------
json accuracy(const Built &b, const Dataset &d, const Args &a) {
  json j;
  const auto &q = d.test_pts;
  const auto &ref = d.test_vals;
  const double ref_std = ref.std().item<double>();
  auto t0 = Clock::now();
  auto pred = ad::interpolate(*b.t, q, ad::InterpMode::Nearest);
  const double nearest_ms = ms_since(t0);
  auto covered = leaf_of(*b.t, q) > 0;
  const double cov = covered.to(torch::kFloat64).mean().item<double>();
  auto se = (pred - ref).pow(2).mean(1);
  const double rmse_all = std::sqrt(se.mean().item<double>());
  const double rmse_cov =
      covered.any().item<bool>() ? std::sqrt(se.index({covered}).mean().item<double>()) : NAN;
  j["test_points"] = q.size(0);
  j["ref_std"] = ref_std;
  j["coverage"] = cov;
  j["rmse_nearest_all"] = rmse_all;
  j["rmse_nearest_covered"] = rmse_cov;
  j["nrmse_nearest_all"] = rmse_all / ref_std;
  j["interp_nearest_ms"] = nearest_ms;
  const int64_t mq = std::min<int64_t>(a.ml_queries, q.size(0));
  if (mq > 0) {
    auto qs = q.slice(0, 0, mq);
    t0 = Clock::now();
    auto ml = ad::interpolate(*b.t, qs, ad::InterpMode::Multilinear);
    j["interp_multilinear_ms"] = ms_since(t0);
    j["ml_queries"] = mq;
    j["rmse_multilinear_all"] = std::sqrt((ml - ref.slice(0, 0, mq)).pow(2).mean().item<double>());
  }
  return j;
}

json level_histogram(const ad::AdaptiveSparseTensor &t) {
  json h = json::array();
  for (auto c : t.counts_by_level())
    h.push_back(c);
  return h;
}

void time_adaptive(const ad::AdaptiveSparseTensor &t, const at::Tensor &feats_cpu,
                   const at::Tensor &weight_cpu, const torch::Device dev, const Args &a,
                   json &out) {
  const bool cuda = dev.is_cuda();
  ad::AdaptiveKernelConfig kc; // kernel 3, CenterSampled, no stride
  const auto coords = t.coordinates();
  const auto levels = t.levels();
  const auto cfg = t.config();
  auto f = feats_cpu.to(dev);
  auto w = weight_cpu.to(dev);
  auto tt = t.with_features(f);

  // Process-cold first call: fresh topology, empty cache, first conv in this process/case.
  ad::KernelMapCache::instance().clear();
  {
    auto fresh = ad::AdaptiveSparseTensor::from_voxels(coords, levels, f, cfg);
    sync(cuda);
    auto t0 = Clock::now();
    auto y = ad::adaptive_convolution(fresh, w, kc);
    sync(cuda);
    out["first_conv_ms"] = ms_since(t0);
  }
  ad::KernelMapCache::instance().clear();
  int64_t pairs = 0;
  {
    auto m = ad::build_adaptive_kernel_map(tt, kc);
    pairs = m.in_index.numel();
  }
  out["kmap_pairs"] = pairs;
  out["kmap_bytes"] = pairs * int64_t(3 * sizeof(int64_t) + sizeof(double));
  out["avg_neighbors"] = t.size() ? double(pairs) / double(t.size()) : 0.0;

  ad::AdaptiveSparseTensor fresh = tt;
  out["kmap_cold_ms"] = to_json(measure(
      a.warmup, a.reps, false,
      [&] {
        ad::KernelMapCache::instance().clear();
        fresh = ad::AdaptiveSparseTensor::from_voxels(coords, levels, f, cfg);
      },
      [&] { ad::build_adaptive_kernel_map(fresh, kc); }));
  ad::KernelMapCache::instance().clear();
  ad::adaptive_convolution(tt, w, kc); // populate cache
  out["fwd_ms"] = to_json(measure(a.warmup, a.reps, cuda, [] {},
                                  [&] { ad::adaptive_convolution(tt, w, kc); }));
  auto fg = f.detach().clone().set_requires_grad(true);
  auto wg = w.detach().clone().set_requires_grad(true);
  auto tg = tt.with_features(fg);
  out["fwd_bwd_ms"] = to_json(measure(
      a.warmup, a.reps, cuda,
      [&] {
        fg.mutable_grad() = at::Tensor();
        wg.mutable_grad() = at::Tensor();
      },
      [&] { ad::adaptive_convolution(tg, wg, kc).features().sum().backward(); }));
  if (a.interleave) {
    // fwd, fwd+bwd, fwd, fwd+bwd, ... in one loop: both see the same allocator / clock state.
    std::vector<double> tf, tb;
    for (int i = 0; i < a.warmup + a.reps; ++i) {
      sync(cuda);
      auto t0 = Clock::now();
      ad::adaptive_convolution(tt, w, kc);
      sync(cuda);
      const double f_ms = ms_since(t0);
      fg.mutable_grad() = at::Tensor();
      wg.mutable_grad() = at::Tensor();
      sync(cuda);
      t0 = Clock::now();
      ad::adaptive_convolution(tg, wg, kc).features().sum().backward();
      sync(cuda);
      const double b_ms = ms_since(t0);
      if (i >= a.warmup) {
        tf.push_back(f_ms);
        tb.push_back(b_ms);
      }
    }
    auto stat = [](std::vector<double> t) {
      std::sort(t.begin(), t.end());
      const size_t n = t.size();
      Stat s;
      s.med = n % 2 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
      s.min = t.front();
      s.max = t.back();
      return s;
    };
    out["interleaved_fwd_ms"] = to_json(stat(tf));
    out["interleaved_fwd_bwd_ms"] = to_json(stat(tb));
  }
  ad::KernelMapCache::instance().clear();
}

void time_minkowski(const Built &b, const at::Tensor &feats_cpu, const torch::Device dev,
                    const Args &a, json &out) {
  using minkowski::SparseTensor;
  const bool cuda = dev.is_cuda();
  // Level-0 coordinates of a uniform adaptive tensor are the integer grid coordinates.
  auto c3 = b.t->coordinates().to(torch::kInt32);
  auto coords = torch::cat({torch::zeros({c3.size(0), 1}, torch::kInt32), c3}, 1).contiguous().to(dev);
  auto f = feats_cpu.to(dev);
  const int C = int(feats_cpu.size(1));
  minkowski::MinkowskiConvolution conv(C, C, /*kernel_size=*/3, /*stride=*/1, /*dilation=*/1,
                                       /*bias=*/false, /*expand_coordinates=*/false,
                                       minkowski::ConvolutionMode::DEFAULT, /*dimension=*/3);
  conv->to(dev);
  {
    sync(cuda);
    torch::NoGradGuard ng;
    auto t0 = Clock::now();
    SparseTensor st(f, coords);
    auto y = conv->forward(st);
    sync(cuda);
    out["mink_first_conv_ms"] = ms_since(t0);
  }
  out["mink_insert_ms"] = to_json(measure(a.warmup, a.reps, cuda, [] {},
                                          [&] { SparseTensor st(f, coords); }));
  // Cold: new coordinate manager -> insert + kernel map + convolution.
  // Forward-only timings run without autograd (the conv kernel is a parameter).
  out["mink_cold_conv_ms"] = to_json(measure(a.warmup, a.reps, cuda, [] {}, [&] {
    torch::NoGradGuard ng;
    SparseTensor st(f, coords);
    conv->forward(st);
  }));
  SparseTensor st(f, coords);
  conv->forward(st); // kernel map now cached in st's manager
  out["mink_fwd_ms"] = to_json(measure(a.warmup, a.reps, cuda, [] {}, [&] {
    torch::NoGradGuard ng;
    conv->forward(st);
  }));
  // Leaf features on the existing coordinate map (the coordinates constructor puts a
  // quantization op in the graph, which a repeated backward() would traverse twice).
  auto fg = st.F().detach().clone().set_requires_grad(true);
  SparseTensor sg(fg, st.coordinate_map_key(), st.coordinate_manager());
  conv->forward(sg);
  out["mink_fwd_bwd_ms"] = to_json(measure(
      a.warmup, a.reps, cuda,
      [&] {
        fg.mutable_grad() = at::Tensor();
        conv->zero_grad();
      },
      [&] { conv->forward(sg).F().sum().backward(); }));
  out["mink_kmap_est_ms"] = out["mink_cold_conv_ms"]["med"].get<double>() -
                            out["mink_insert_ms"]["med"].get<double>() -
                            out["mink_fwd_ms"]["med"].get<double>();
}

void emit(const json &j, const Args &a) {
  const std::string line = j.dump();
  std::cout << line << std::endl;
  if (!a.jsonl.empty()) {
    std::ofstream f(a.jsonl, std::ios::app);
    f << line << "\n";
  }
}

// ---------------------------------------------------------------------------------------------
// voxelize_sphere analysis: the reported accuracy is 1 - |sum of signed volume error| / V, so
// over- and under-estimates cancel; the geometric error is the volume of the symmetric
// difference between the occupied leaves and the ball (Monte Carlo estimate here).
// ---------------------------------------------------------------------------------------------
void sphere_analysis(const Args &a) {
  const double radius = 100.0;
  const ad::sphere_detail::SphereGeometry g{radius};
  const double exact = g.exact_volume();
  torch::manual_seed(a.seed);
  const int64_t M = 2000000;
  auto pts = (torch::rand({M, 3}, torch::kFloat64) * 2.0 - 1.0) * radius;
  auto inside = pts.pow(2).sum(1) <= radius * radius;
  const double cube = std::pow(2.0 * radius, 3);

  auto mc = [&](const at::Tensor &pred_occ) {
    const double frac = (pred_occ != inside).to(torch::kFloat64).mean().item<double>();
    const double se = std::sqrt(frac * (1 - frac) / double(M));
    return std::pair<double, double>{100.0 * frac * cube / exact, 100.0 * se * cube / exact};
  };

  for (double target : {90.0, 95.0, 98.0, 99.0, 99.5, 99.9}) {
    double achieved = 0;
    auto t0 = Clock::now();
    auto s = ad::voxelize_sphere(radius, target, {0, 0, 0}, 10'000'000, 10, &achieved);
    const double ms = ms_since(t0);
    const double est = ad::sphere_volume_estimate(s);
    auto occ = ad::interpolate(s, pts, ad::InterpMode::Nearest).squeeze(1) > 0.5;
    auto [sym, sym_se] = mc(occ);
    // Boundary leaves by level (the cells that decide the geometric error).
    auto co = s.coordinates();
    auto lv = s.levels();
    std::vector<int64_t> bnd(11, 0);
    auto ca = co.accessor<int64_t, 2>();
    auto la = lv.accessor<int32_t, 1>();
    for (int64_t i = 0; i < s.size(); ++i)
      if (g.is_boundary(ca[i][0], ca[i][1], ca[i][2], la[i]))
        ++bnd[size_t(la[i])];
    int coarsest = -1;
    for (int l = 0; l <= 10 && coarsest < 0; ++l)
      if (bnd[size_t(l)] > 0)
        coarsest = l;
    json j;
    j["kind"] = "sphere_adaptive";
    j["target_accuracy"] = target;
    j["reported_accuracy"] = achieved;
    j["signed_volume_error_pct"] = 100.0 * (est - exact) / exact;
    j["symmetric_difference_pct"] = sym;
    j["symmetric_difference_se_pct"] = sym_se;
    j["leaves"] = s.size();
    j["levels"] = level_histogram(s);
    j["boundary_leaves_by_level"] = bnd;
    j["coarsest_boundary_level"] = coarsest;
    j["coarsest_boundary_size"] = coarsest >= 0 ? std::ldexp(2.0 * radius, -coarsest) : 0.0;
    j["build_ms"] = ms;
    emit(j, a);
  }

  // Replay the FIFO refinement to show how the signed estimate oscillates within a level:
  // count crossings of each target band after the first one.
  {
    struct Cell { int64_t x, y, z; int l; };
    std::vector<Cell> frontier{{0, 0, 0, 0}}, next;
    double vol = g.contribution(0, 0, 0, 0);
    int64_t count = 1;
    for (int level = 0; level < 7; ++level) {
      double lo_acc = 101, hi_acc = -1;
      int64_t sign_changes = 0;
      double prev = vol - exact;
      next.clear();
      for (const Cell &c : frontier) {
        vol -= g.contribution(c.x, c.y, c.z, c.l);
        for (int k = 0; k < 8; ++k) {
          Cell ch{2 * c.x + (k & 1), 2 * c.y + ((k >> 1) & 1), 2 * c.z + ((k >> 2) & 1), c.l + 1};
          vol += g.contribution(ch.x, ch.y, ch.z, ch.l);
          if (g.is_boundary(ch.x, ch.y, ch.z, ch.l))
            next.push_back(ch);
        }
        count += 7;
        const double acc = g.accuracy(vol);
        lo_acc = std::min(lo_acc, acc);
        hi_acc = std::max(hi_acc, acc);
        const double err = vol - exact;
        if ((err > 0) != (prev > 0))
          ++sign_changes;
        prev = err;
      }
      json j;
      j["kind"] = "sphere_fifo_level";
      j["level_split"] = level;
      j["splits"] = int64_t(frontier.size());
      j["leaves_after"] = count;
      j["accuracy_min_during_level"] = lo_acc;
      j["accuracy_max_during_level"] = hi_acc;
      j["accuracy_end_of_level"] = g.accuracy(vol);
      j["signed_error_sign_changes"] = sign_changes;
      emit(j, a);
      frontier.swap(next);
      if (count > 5'000'000)
        break;
    }
  }

  // Sparse uniform solid-sphere voxelization (cells whose centre is inside) per level.
  for (int level = 2; level <= 8; ++level) {
    const int64_t n = int64_t(1) << level;
    int64_t occupied = 0, boundary = 0;
    for (int64_t x = 0; x < n; ++x)
      for (int64_t y = 0; y < n; ++y)
        for (int64_t z = 0; z < n; ++z) {
          occupied += g.occupied(x, y, z, level);
          boundary += g.is_boundary(x, y, z, level);
        }
    const double size = g.cell_size(level);
    auto cell = torch::floor((pts + radius) / size).to(torch::kInt64);
    auto centre = (cell.to(torch::kFloat64) + 0.5) * size - radius;
    auto occ = centre.pow(2).sum(1) <= radius * radius;
    auto [sym, sym_se] = mc(occ);
    json j;
    j["kind"] = "sphere_uniform";
    j["level"] = level;
    j["cell_size"] = size;
    j["occupied_voxels"] = occupied;
    j["boundary_cells"] = boundary;
    j["dense_cells"] = n * n * n;
    j["reported_accuracy"] = g.accuracy(double(occupied) * size * size * size);
    j["signed_volume_error_pct"] = 100.0 * (double(occupied) * size * size * size - exact) / exact;
    j["symmetric_difference_pct"] = sym;
    j["symmetric_difference_se_pct"] = sym_se;
    emit(j, a);
  }
}

} // namespace

int main(int argc, char **argv) {
  const Args a = parse(argc, argv);
  if (a.sphere) {
    sphere_analysis(a);
    return 0;
  }
  const bool have_cuda = torch::cuda::is_available();

  for (const auto &ds : a.datasets) {
    auto t0 = Clock::now();
    const Dataset d = make_dataset(ds, a);
    const double load_ms = ms_since(t0);

    // Matched-count target: the voxel count of refine_by_density(16).
    int64_t matched_target = 0;
    const bool need_match =
        std::find(a.methods.begin(), a.methods.end(), "u_matched") != a.methods.end();
    if (need_match) {
      auto ref = ad::AdaptiveSparseTensor::from_points(d.train_pts, d.train_vals,
                                                       config(a.base, a.depth));
      ref.refine_by_density(16);
      matched_target = ref.size();
    }

    std::vector<std::pair<std::string, bool>> runs;
    for (const auto &m : a.methods) {
      runs.push_back({m, false});
      if (a.drop_empty && m.rfind("a_", 0) == 0)
        runs.push_back({m, true});
    }
    for (const auto &[method, drop] : runs) {
      for (const auto &devname : a.devices) {
        if (devname == "cuda" && !have_cuda) {
          std::cerr << "skipping cuda: not available\n";
          continue;
        }
        const torch::Device dev = devname == "cuda" ? torch::Device(torch::kCUDA, 0)
                                                    : torch::Device(torch::kCPU);
#ifdef USE_CUDA
        if (dev.is_cuda())
          c10::cuda::CUDACachingAllocator::resetPeakStats(0);
#endif
        json j;
        j["kind"] = "case";
        j["dataset"] = ds;
        j["method"] = method + (drop ? "+drop" : "");
        j["device"] = devname;
        j["depth"] = a.depth;
        j["base"] = a.base;
        j["channels"] = a.channels;
        j["train_points"] = d.train_pts.size(0);
        j["seed"] = a.seed;
        j["dataset_note"] = d.note;
        j["dataset_load_ms"] = load_ms;
        if (method == "none") {
          j["peak_ram_mb"] = peak_ram_mb();
          emit(j, a);
          continue;
        }
        Built b;
        j["build_ms"] = to_json(measure(std::min(a.warmup, 1), std::min(a.reps, 3), false, [] {},
                                        [&] { b = build_method(method, d, a, matched_target, drop); }));
        const auto &t = *b.t;
        j["voxels"] = t.size();
        j["levels"] = level_histogram(t);
        if (b.uniform)
          j["voxel_size"] = b.voxel_size;
        if (need_match)
          j["matched_target"] = matched_target;
        if (b.refine_threshold != 0)
          j["refine_threshold"] = b.refine_threshold;
        if (!b.roi_lo.empty())
          j["roi"] = {b.roi_lo, b.roi_hi};
        if (!b.uniform) {
          // Dynamic-topology cost: collapse the tree back to level 0 (a full re-refinement then
          // costs build_ms again, and every topology change needs a cold kernel map).
          std::optional<ad::AdaptiveSparseTensor> c;
          j["coarsen_all_ms"] = to_json(measure(
              std::min(a.warmup, 1), std::min(a.reps, 3), false, [&] { c = t; },
              [&] { c->coarsen(ad::AxisAlignedRegion::everything(3), 0, ad::Reduction::Mean); }));
        }
        // Covered physical volume (sum of leaf volumes) and bytes.
        j["covered_volume"] = t.volumes().sum().item<double>();
        j["feature_bytes"] = t.size() * a.channels * int64_t(sizeof(float));
        j["coord_bytes_adaptive"] = t.size() * int64_t(3 * sizeof(int64_t) + sizeof(int32_t));
        j["coord_bytes_mink"] = t.size() * int64_t(4 * sizeof(int32_t));
        j["accuracy"] = accuracy(b, d, a);

        torch::manual_seed(a.seed + 1);
        auto feats = torch::randn({t.size(), a.channels}, torch::kFloat32);
        auto weight = torch::randn({27, a.channels, a.channels}, torch::kFloat32) * 0.1;
        json ae;
        time_adaptive(t, feats, weight, dev, a, ae);
        j["adaptive_engine"] = ae;
        if (b.uniform && a.mink) {
          json me;
          try {
            time_minkowski(b, feats, dev, a, me);
          } catch (const std::exception &e) {
            me["error"] = e.what();
          }
          j["minkowski_engine"] = me;
        }
        j["peak_ram_mb"] = peak_ram_mb();
#ifdef USE_CUDA
        if (dev.is_cuda()) {
          const auto st = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
          j["peak_vram_alloc_mb"] =
              double(st.allocated_bytes[0].peak) / (1024.0 * 1024.0);
          j["peak_vram_reserved_mb"] =
              double(st.reserved_bytes[0].peak) / (1024.0 * 1024.0);
        }
#endif
        emit(j, a);
      }
    }
  }
  return 0;
}
