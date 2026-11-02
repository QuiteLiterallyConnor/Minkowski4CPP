// Interpolation evaluation (not part of the correctness suite): Nearest vs Multilinear vs
// ContinuousMultilinear.
// Run: adaptive_interp_eval [--section all|continuity|accuracy|latency] [--seed 1]
//                           [--lines 300] [--queries 50000] [--interp 20000] [--n 20000]
//                           [--depth 4] [--warmup 2] [--reps 11] [--only all|nearest|multilinear|continuous|none]
// Prints "key  value" lines.
//  continuity: leaf-boundary crossings along random lines through adaptive trees; at each
//    crossing the result is evaluated 1e-9 before and after the boundary. Reports the max and
//    the count of jumps > 1e-6 (relative to the field range) for level-change and equal-level
//    crossings, per mode.
//  accuracy: smooth analytic fields sampled at leaf centres of random and field-refined trees;
//    nRMSE = RMSE / (max - min of the field) over random query points inside the leaves, also
//    split by whether the point's tent neighbourhood mixes levels.
//  latency: the adaptive_kmap_bench workload (half the points uniform in [0,256)^3, half in
//    [0,16)^3, [0,16)^3 refined to --depth); --interp queries with the same distribution. All
//    modes interleaved in one process (rotating order per repeat): median / min / max over
//    --reps after --warmup rounds. --only <mode> times just that mode (for peak memory runs).
// Always: peak_working_set_mb (process peak, Windows) at exit.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "interp_eval_util.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

using namespace minkowski::adaptive;
using namespace interp_eval;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
    std::string section = "all";
    uint64_t seed = 1;
    int lines = 300;
    int64_t queries = 50000;
    int64_t interp = 20000;
    int64_t n = 20000;
    uint32_t depth = 4;
    int warmup = 2;
    int reps = 11;
    std::string only = "all";
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--section") a.section = v;
        else if (k == "--seed") a.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "--lines") a.lines = std::atoi(v.c_str());
        else if (k == "--queries") a.queries = std::atoll(v.c_str());
        else if (k == "--interp") a.interp = std::atoll(v.c_str());
        else if (k == "--n") a.n = std::atoll(v.c_str());
        else if (k == "--depth") a.depth = uint32_t(std::atoi(v.c_str()));
        else if (k == "--warmup") a.warmup = std::atoi(v.c_str());
        else if (k == "--reps") a.reps = std::atoi(v.c_str());
        else if (k == "--only") a.only = v;
        else {
            std::cerr << "unknown argument " << k << "\n";
            std::exit(2);
        }
    }
    return a;
}

void print(const std::string& key, double v) {
    std::cout << std::left << std::setw(48) << key << "  " << v << "\n";
}

const std::vector<std::pair<std::string, InterpMode>> kModes = {
    {"nearest", InterpMode::Nearest},
    {"multilinear", InterpMode::Multilinear},
    {"continuous", InterpMode::ContinuousMultilinear}};

// Analytic fields on R^D (D <= 3 used here).
struct Field {
    std::string name;
    std::function<double(const DVec&, int)> f;
};

std::vector<Field> fields() {
    const double pi = 3.14159265358979323846;
    return {
        {"sine", [pi](const DVec& x, int D) {
             double v = 1.0;
             for (int d = 0; d < D; ++d) v *= std::sin(2.0 * pi * x[d] / 5.0 + 0.3 * d);
             return v;
         }},
        {"gauss", [](const DVec& x, int D) {
             double r2 = 0.0;
             for (int d = 0; d < D; ++d) r2 += (x[d] - 0.7) * (x[d] - 0.7);
             return std::exp(-r2 / (2.0 * 0.8 * 0.8));
         }},
        {"linear", [](const DVec& x, int D) {
             double v = 0.5;
             for (int d = 0; d < D; ++d) v += (1.0 + 0.5 * d) * x[d];
             return v;
         }},
        {"front", [](const DVec& x, int D) {
             double r2 = 0.0;
             for (int d = 0; d < D; ++d) r2 += x[d] * x[d];
             return std::tanh((std::sqrt(r2) - 2.0) / 0.3);
         }},
    };
}

// Iso tree over [-4, 4)^D (base 1, roots [-4,4)), refined where the field varies: a cell is
// split while the spread of f over its centre and corners exceeds `tol` (max level `depth`).
std::shared_ptr<const Topology> field_tree(int D, uint32_t depth, const Field& fld, double tol) {
    AdaptiveVoxelConfig cfg;
    for (int d = 0; d < D; ++d) {
        cfg.base_voxel_size.push_back(1.0);
        cfg.origin.push_back(0.0);
    }
    cfg.max_depth = depth;
    std::vector<Key> leaves, stack, children;
    std::vector<int64_t> c(D, -4);
    while (true) {
        Key root{};
        for (int d = 0; d < D; ++d) root.coordinates[d] = c[d];
        stack.push_back(root);
        int d = D - 1;
        for (; d >= 0; --d) {
            if (++c[d] < 4) break;
            c[d] = -4;
        }
        if (d < 0) break;
    }
    while (!stack.empty()) {
        Key k = stack.back();
        stack.pop_back();
        DVec lo{}, hi{};
        const double size = std::ldexp(1.0, -static_cast<int>(k.level));
        for (int d = 0; d < D; ++d) {
            lo[d] = static_cast<double>(k.coordinates[d]) * size;
            hi[d] = lo[d] + size;
        }
        DVec mid{};
        for (int d = 0; d < D; ++d) mid[d] = 0.5 * (lo[d] + hi[d]);
        double fmin = fld.f(mid, D), fmax = fmin;
        for (int corner = 0; corner < (1 << D); ++corner) {
            DVec x{};
            for (int d = 0; d < D; ++d) x[d] = (corner >> d) & 1 ? hi[d] : lo[d];
            const double v = fld.f(x, D);
            fmin = std::min(fmin, v);
            fmax = std::max(fmax, v);
        }
        if (k.level < depth && fmax - fmin > tol) {
            children.clear();
            adt::child_keys(k, D, children);
            stack.insert(stack.end(), children.begin(), children.end());
        } else {
            leaves.push_back(k);
        }
    }
    std::sort(leaves.begin(), leaves.end(), adt::key_less);
    return Topology::make(cfg, std::move(leaves));
}

at::Tensor sample_centres(const Topology& t, const Field& fld) {
    auto f = torch::empty({t.size(), 1}, torch::kFloat64);
    for (int64_t i = 0; i < t.size(); ++i) f[i][0] = fld.f(leaf_center(t, i), t.D);
    return f;
}

// Points uniform in the bounding box, kept when inside a leaf.
std::vector<DVec> inside_points(const Topology& t, int64_t n, std::mt19937_64& rng) {
    std::vector<DVec> out;
    while (static_cast<int64_t>(out.size()) < n)
        for (const DVec& q : random_points(t, n, rng))
            if (t.find_containing(q) >= 0 && static_cast<int64_t>(out.size()) < n) out.push_back(q);
    return out;
}

// True when the leaves with a positive tent at q do not all share one level.
bool mixed_neighbourhood(const Topology& t, const DVec& q) {
    const DVec s = t.size_at(t.present_levels.front());
    DVec lo{}, hi{};
    for (int d = 0; d < t.D; ++d) {
        lo[d] = q[d] - 0.5 * s[d];
        hi[d] = q[d] + 0.5 * s[d];
    }
    int64_t level = -1;
    bool mixed = false;
    t.for_each_overlap(lo, hi, [&](int64_t i, double) {
        if (mixed || tent(t, i, q) <= 0.0) return;
        const int64_t l = t.keys[i].level;
        if (level < 0) level = l;
        else if (l != level) mixed = true;
    });
    return mixed;
}

void continuity(const Args& a) {
    std::mt19937_64 rng(a.seed);
    const auto flds = fields();
    struct Tree {
        std::string name;
        std::shared_ptr<const Topology> t;
    };
    std::vector<Tree> trees;
    for (int D : {2, 3}) {
        trees.push_back({"random_d" + std::to_string(D), random_tree(D, 4, 2, 0.4, 0.05, rng, true)});
        trees.push_back({"front_d" + std::to_string(D), field_tree(D, D == 2 ? 6 : 4, flds[3], 0.1)});
    }
    for (const auto& tr : trees) {
        const Topology& t = *tr.t;
        std::vector<DVec> before, after;
        std::vector<char> change;
        auto ends = random_points(t, 2 * a.lines, rng);
        for (int i = 0; i < a.lines; ++i)
            for (const auto& c : line_crossings(t, ends[2 * i], ends[2 * i + 1], 1e-9)) {
                before.push_back(c.before);
                after.push_back(c.after);
                change.push_back(c.level_change);
            }
        int64_t nchange = 0;
        for (char c : change) nchange += c;
        const std::string p = "cont_" + tr.name + "_";
        print(p + "leaves", double(t.size()));
        print(p + "crossings_level_change", double(nchange));
        print(p + "crossings_equal_level", double(change.size() - nchange));
        for (const Field& fld : {flds[0], flds[3]}) {
            auto x = AdaptiveSparseTensor::from_topology(tr.t, sample_centres(t, fld));
            const double range = (x.features().max() - x.features().min()).item<double>();
            for (const auto& [mname, mode] : kModes) {
                auto j = (interpolate(x, to_tensor(before, t.D), mode) -
                          interpolate(x, to_tensor(after, t.D), mode))
                             .abs()
                             .select(1, 0) /
                         range;
                const auto* jp = j.data_ptr<double>();
                double max_change = 0.0, max_equal = 0.0;
                int64_t big_change = 0, big_equal = 0;
                for (size_t i = 0; i < change.size(); ++i) {
                    if (change[i]) {
                        max_change = std::max(max_change, jp[i]);
                        big_change += jp[i] > 1e-6;
                    } else {
                        max_equal = std::max(max_equal, jp[i]);
                        big_equal += jp[i] > 1e-6;
                    }
                }
                const std::string q = p + fld.name + "_" + mname + "_";
                print(q + "maxjump_level_change", max_change);
                print(q + "jumps_gt1e-6_level_change", double(big_change));
                print(q + "maxjump_equal_level", max_equal);
                print(q + "jumps_gt1e-6_equal_level", double(big_equal));
            }
        }
    }
}

void accuracy(const Args& a) {
    std::mt19937_64 rng(a.seed + 100);
    const auto flds = fields();
    for (int D : {2, 3}) {
        for (const Field& fld : flds) {
            struct Tree {
                std::string name;
                std::shared_ptr<const Topology> t;
            };
            std::vector<Tree> trees = {
                {"random", random_tree(D, D == 2 ? 5 : 4, 4, 0.35, 0.0, rng, true)},
                {"refined", field_tree(D, D == 2 ? 6 : 4, fld, D == 2 ? 0.05 : 0.1)},
                {"uniform", random_tree(D, 2, 4, 1.0, 0.0, rng, true)}};
            for (const auto& tr : trees) {
                const Topology& t = *tr.t;
                auto x = AdaptiveSparseTensor::from_topology(tr.t, sample_centres(t, fld));
                auto pts = inside_points(t, a.queries, rng);
                std::vector<double> truth(pts.size());
                double fmin = 1e300, fmax = -1e300;
                for (size_t i = 0; i < pts.size(); ++i) {
                    truth[i] = fld.f(pts[i], D);
                    fmin = std::min(fmin, truth[i]);
                    fmax = std::max(fmax, truth[i]);
                }
                const int64_t nclass = std::min<int64_t>(static_cast<int64_t>(pts.size()), 20000);
                std::vector<char> mixed(static_cast<size_t>(nclass));
                int64_t nmixed = 0;
                for (int64_t i = 0; i < nclass; ++i) nmixed += mixed[i] = mixed_neighbourhood(t, pts[i]);
                const std::string p = "acc_d" + std::to_string(D) + "_" + fld.name + "_" + tr.name + "_";
                print(p + "leaves", double(t.size()));
                print(p + "levels", double(t.present_levels.size()));
                print(p + "mixed_fraction", double(nmixed) / double(nclass));
                auto q = to_tensor(pts, D);
                for (const auto& [mname, mode] : kModes) {
                    auto y = interpolate(x, q, mode);
                    const double* yp = y.data_ptr<double>();
                    double se = 0.0, se_m = 0.0, se_u = 0.0, emax = 0.0;
                    for (size_t i = 0; i < pts.size(); ++i) {
                        const double e = yp[i] - truth[i];
                        se += e * e;
                        emax = std::max(emax, std::fabs(e));
                        if (static_cast<int64_t>(i) < nclass) (mixed[i] ? se_m : se_u) += e * e;
                    }
                    const double range = fmax - fmin;
                    const std::string k = p + mname + "_";
                    print(k + "nrmse", std::sqrt(se / pts.size()) / range);
                    print(k + "maxerr_rel", emax / range);
                    if (nmixed > 0) print(k + "nrmse_mixed", std::sqrt(se_m / nmixed) / range);
                    if (nclass > nmixed)
                        print(k + "nrmse_uniform", std::sqrt(se_u / (nclass - nmixed)) / range);
                }
            }
        }
    }
}

void latency(const Args& a) {
    torch::manual_seed(0);
    auto pts = torch::cat({torch::rand({a.n / 2, 3}, torch::kFloat64) * 256.0,
                           torch::rand({a.n - a.n / 2, 3}, torch::kFloat64) * 16.0});
    auto feats = torch::randn({a.n, 8}, torch::kFloat64);
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {8.0, 8.0, 8.0};
    const double m = 8.0 / double(1u << a.depth);
    cfg.min_voxel_size = {m, m, m};
    cfg.max_depth = a.depth;
    auto t = AdaptiveSparseTensor::from_points(pts, feats, cfg);
    t.refine(AxisAlignedRegion{{0, 0, 0}, {16.0, 16.0, 16.0}}, a.depth);
    t = t.with_features(t.features().to(torch::kFloat32));
    auto q = torch::cat({torch::rand({a.interp / 2, 3}, torch::kFloat64) * 256.0,
                         torch::rand({a.interp - a.interp / 2, 3}, torch::kFloat64) * 16.0});
    print("lat_leaves", double(t.size()));
    print("lat_queries", double(a.interp));
    std::vector<std::pair<std::string, InterpMode>> modes;
    for (const auto& md : kModes)
        if (a.only == "all" || a.only == md.first) modes.push_back(md);
    if (modes.empty()) return;  // --only none: workload only (peak memory baseline)
    std::map<std::string, std::vector<double>> times;
    for (int r = 0; r < a.warmup + a.reps; ++r) {
        for (size_t i = 0; i < modes.size(); ++i) {
            const auto& md = modes[(i + r) % modes.size()];
            auto t0 = Clock::now();
            auto y = interpolate(t, q, md.second);
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            if (r >= a.warmup) times[md.first].push_back(ms);
        }
    }
    for (auto& [name, v] : times) {
        std::sort(v.begin(), v.end());
        const double med = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
        print("lat_" + name + "_ms", med);
        print("lat_" + name + "_min_ms", v.front());
        print("lat_" + name + "_max_ms", v.back());
    }
}

void print_peak_ws() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        print("peak_working_set_mb", pmc.PeakWorkingSetSize / 1048576.0);
#endif
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    std::cout << std::setprecision(6);
    if (a.section == "all" || a.section == "continuity") continuity(a);
    if (a.section == "all" || a.section == "accuracy") accuracy(a);
    if (a.section == "all" || a.section == "latency") latency(a);
    print_peak_ws();
    return 0;
}
