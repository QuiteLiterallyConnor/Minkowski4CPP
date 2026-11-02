// Kernel-map part reuse (KernelMapCache::set_part_reuse_bytes): batches made of samples placed
// apart along the first axis must get exactly the map a full build gives - same pairs in the
// same order, same weights, same output topology - whether the parts' maps are cut from a full
// build, built one by one or reused from earlier batches.
#include <algorithm>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/adaptive/adaptive_topology.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

// Turns part reuse on for one test and restores the default (off) afterwards.
struct PartReuse {
    explicit PartReuse(size_t bytes) {
        KernelMapCache::instance().clear();
        KernelMapCache::instance().set_part_reuse_bytes(bytes);
    }
    ~PartReuse() {
        KernelMapCache::instance().set_part_reuse_bytes(0);
        KernelMapCache::instance().clear();
    }
};

AdaptiveVoxelConfig make_config(int D, double base, double origin) {
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size.assign(D, base);
    cfg.origin.assign(D, origin);
    cfg.max_depth = 3;
    return cfg;
}

// A sample spanning about 4 level-0 cells per axis, refined to level 3 in one corner, 2 in
// another, so its leaves sit at levels 0-3.
AdaptiveSparseTensor make_sample(int seed, int D, const AdaptiveVoxelConfig &cfg) {
    torch::manual_seed(seed);
    const double b = cfg.base_voxel_size[0];
    const double o = cfg.origin[0];
    auto pts = torch::cat({torch::rand({150, D}, torch::kFloat64) * (4 * b),
                           torch::rand({100, D}, torch::kFloat64) * b}) + o;
    auto t = AdaptiveSparseTensor::from_points(pts, torch::ones({250, 1}, torch::kFloat64), cfg);
    std::vector<double> lo(D, o), hi(D, o + b), lo2(D, o + 2 * b), hi2(D, o + 3 * b);
    t.refine(AxisAlignedRegion{lo, hi}, 3);
    t.refine(AxisAlignedRegion{lo2, hi2}, 2);
    return t;
}

// Samples side by side along the first axis: sample k moved by `stride` * k level-0 cells.
AdaptiveSparseTensor make_batch(const std::vector<AdaptiveSparseTensor> &samples,
                                const std::vector<int> &order, int64_t stride,
                                const AdaptiveVoxelConfig &cfg) {
    std::vector<at::Tensor> cs, ls, fs;
    for (size_t k = 0; k < order.size(); ++k) {
        const auto &s = samples[order[k]];
        auto c = s.coordinates().clone();
        auto shift = (torch::pow(2.0, s.levels().to(torch::kFloat64)) *
                      static_cast<double>(stride * static_cast<int64_t>(k)))
                         .to(torch::kInt64);
        c.select(1, 0).add_(shift);
        cs.push_back(c);
        ls.push_back(s.levels());
        fs.push_back(torch::randn({s.size(), 3}, torch::kFloat64));
    }
    return AdaptiveSparseTensor::from_voxels(torch::cat(cs, 0), torch::cat(ls, 0),
                                             torch::cat(fs, 0), cfg);
}

KernelMapEntry full_build(const AdaptiveSparseTensor &t, const AdaptiveKernelConfig &kc) {
    KernelMapCache::instance().set_part_reuse_bytes(0);
    KernelMapCache::instance().clear();
    return KernelMapCache::instance().get_or_build(t, kc);
}

void check_same_entry(const KernelMapEntry &a, const KernelMapEntry &b) {
    CHECK_EQ(a.map.num_kernels, b.map.num_kernels);
    CHECK(a.map.kernel_offsets == b.map.kernel_offsets);
    CHECK(torch::equal(a.map.in_index, b.map.in_index));
    CHECK(torch::equal(a.map.out_index, b.map.out_index));
    CHECK(torch::equal(a.map.kernel_index, b.map.kernel_index));
    CHECK(torch::equal(a.map.weight, b.map.weight));
    CHECK(a.out_topology->keys == b.out_topology->keys);
}

} // namespace

// Exact equality against a full build for every reuse path: all parts new (cut from a full
// build), some new (built alone, then composed), all stored (composed only).
TEST(part_reuse_matches_full_build_modes_shifts_kernels) {
    for (int D : {1, 2, 3}) {
        const auto cfg = make_config(D, 2.0, -3.0);
        std::vector<AdaptiveSparseTensor> samples;
        for (int s = 0; s < 5; ++s)
            samples.push_back(make_sample(10 + s, D, cfg));
        // Samples of b2 equal to one in b1 (in 1-2 D the random samples can coincide).
        size_t b2_hits = 0;
        for (int i : {2, 3, 0, 4}) {
            bool seen = false;
            for (int j : {0, 1, 2})
                seen = seen || (torch::equal(samples[i].coordinates(), samples[j].coordinates()) &&
                                torch::equal(samples[i].levels(), samples[j].levels()));
            b2_hits += seen;
        }
        for (auto mode : {AdaptiveConvMode::CenterSampled, AdaptiveConvMode::VolumeWeighted}) {
            for (int K : {1, 3, 5}) {
                for (int shift : {0, -1, -3}) {
                    AdaptiveKernelConfig kc;
                    kc.mode = mode;
                    kc.kernel_size = K;
                    kc.output_level_shift = shift;
                    const int64_t stride = 4 + (K - 1) / 2 + 1; // span 4 cells, gap r + 1
                    // full_build clears the store, so build each batch's expectation first.
                    auto b1 = make_batch(samples, {0, 1, 2}, stride, cfg);
                    auto b2 = make_batch(samples, {2, 3, 0, 4}, stride, cfg);
                    auto b3 = make_batch(samples, {4, 2, 3, 0, 1}, stride, cfg);
                    const auto e1 = full_build(b1, kc), e2 = full_build(b2, kc),
                               e3 = full_build(b3, kc);
                    PartReuse on(size_t(1) << 30);
                    auto &cache = KernelMapCache::instance();
                    check_same_entry(cache.get_or_build(b1, kc), e1); // 3 misses: cut
                    if (K == 1) { // radius 0: every occupied level-0 cell is a part of its own
                        check_same_entry(cache.get_or_build(b2, kc), e2);
                        check_same_entry(cache.get_or_build(b3, kc), e3);
                        CHECK(cache.stats().part_hits > 0);
                        continue;
                    }
                    auto st = cache.stats();
                    CHECK_EQ(st.part_hits, size_t(0));
                    CHECK_EQ(st.part_misses, size_t(3));
                    check_same_entry(cache.get_or_build(b2, kc), e2); // hits, rest built alone
                    st = cache.stats();
                    CHECK_EQ(st.part_hits, b2_hits);
                    CHECK_EQ(st.part_misses, size_t(7) - b2_hits);
                    check_same_entry(cache.get_or_build(b3, kc), e3); // all 5 stored
                    st = cache.stats();
                    CHECK_EQ(st.part_hits, b2_hits + 5);
                    CHECK_EQ(st.part_misses, size_t(7) - b2_hits);
                    CHECK(st.part_bytes > 0);
                }
            }
        }
    }
}

// Too small a gap (a tap could reach the next sample) or a single sample: no split.
TEST(part_reuse_needs_a_gap_wider_than_the_kernel_radius) {
    const auto cfg = make_config(3, 2.0, 0.0);
    std::vector<AdaptiveSparseTensor> samples{make_sample(1, 3, cfg), make_sample(2, 3, cfg)};
    for (int K : {3, 5}) {
        AdaptiveKernelConfig kc;
        kc.kernel_size = K;
        const int r = (K - 1) / 2;
        // Samples occupy level-0 x cells 0..3, so a stride of 3 + r leaves r empty cells.
        auto tight = make_batch(samples, {0, 1}, 3 + r, cfg);
        auto single = make_batch(samples, {0}, 4, cfg);
        const auto et = full_build(tight, kc), es = full_build(single, kc);
        PartReuse on(size_t(1) << 30);
        auto &cache = KernelMapCache::instance();
        check_same_entry(cache.get_or_build(tight, kc), et);
        check_same_entry(cache.get_or_build(single, kc), es);
        const auto st = cache.stats();
        CHECK_EQ(st.part_hits + st.part_misses, size_t(0));
    }
}

// Modes other than CenterSampled / VolumeWeighted ignore part reuse.
TEST(part_reuse_skips_overlap_weighted) {
    const auto cfg = make_config(3, 2.0, 0.0);
    std::vector<AdaptiveSparseTensor> samples{make_sample(1, 3, cfg), make_sample(2, 3, cfg)};
    AdaptiveKernelConfig kc;
    kc.mode = AdaptiveConvMode::OverlapWeighted;
    auto b = make_batch(samples, {0, 1}, 8, cfg);
    const auto e = full_build(b, kc);
    PartReuse on(size_t(1) << 30);
    check_same_entry(KernelMapCache::instance().get_or_build(b, kc), e);
    CHECK_EQ(KernelMapCache::instance().stats().part_misses, size_t(0));
}

// A small budget keeps the store within it (least recently used parts go first); results stay
// exact and a budget of 0 frees everything.
TEST(part_reuse_budget_and_eviction) {
    const auto cfg = make_config(3, 2.0, 0.0);
    std::vector<AdaptiveSparseTensor> samples;
    for (int s = 0; s < 6; ++s)
        samples.push_back(make_sample(30 + s, 3, cfg));
    AdaptiveKernelConfig kc;
    auto b1 = make_batch(samples, {0, 1, 2, 3, 4, 5}, 6, cfg);
    auto b2 = make_batch(samples, {5, 4, 3, 2, 1, 0}, 6, cfg);
    const auto e1 = full_build(b1, kc), e2 = full_build(b2, kc);
    // Measure one batch's part bytes, then allow about half of them.
    size_t all = 0;
    {
        PartReuse on(size_t(1) << 30);
        KernelMapCache::instance().get_or_build(b1, kc);
        all = KernelMapCache::instance().stats().part_bytes;
    }
    CHECK(all > 0);
    PartReuse on(all / 2);
    auto &cache = KernelMapCache::instance();
    check_same_entry(cache.get_or_build(b1, kc), e1);
    CHECK(cache.stats().part_bytes <= all / 2);
    CHECK(cache.stats().part_bytes > 0);
    check_same_entry(cache.get_or_build(b2, kc), e2); // some parts were evicted: rebuilt alone
    const auto st = cache.stats();
    CHECK(st.part_hits > 0);
    CHECK(st.part_misses > size_t(6));
    CHECK(st.part_bytes <= all / 2);
    cache.set_part_reuse_bytes(0);
    CHECK_EQ(cache.stats().part_bytes, size_t(0));
}

// Convolution outputs and gradients with a composed map equal those with a full build. On CUDA
// with deterministic algorithms: the scatter-adds otherwise use float atomics, whose order (and
// so the last bits) varies between any two runs.
void check_conv(torch::Device dev) {
    auto &ctx = at::globalContext();
    const bool was = ctx.deterministicAlgorithms(), warn = ctx.deterministicAlgorithmsWarnOnly();
    if (dev.is_cuda()) ctx.setDeterministicAlgorithms(true, false);
    struct Restore {
        bool was, warn;
        ~Restore() { at::globalContext().setDeterministicAlgorithms(was, warn); }
    } restore{was, warn};
    const auto cfg = make_config(3, 2.0, 0.0);
    std::vector<AdaptiveSparseTensor> samples;
    for (int s = 0; s < 4; ++s)
        samples.push_back(make_sample(50 + s, 3, cfg));
    for (auto mode : {AdaptiveConvMode::CenterSampled, AdaptiveConvMode::VolumeWeighted}) {
        AdaptiveKernelConfig kc;
        kc.mode = mode;
        auto batch = make_batch(samples, {3, 1, 0, 2}, 6, cfg).to(dev);
        torch::manual_seed(7);
        auto w = torch::randn({27, 3, 4}, torch::kFloat64).to(dev);
        auto g = torch::randn({batch.size(), 4}, torch::kFloat64).to(dev);
        auto run = [&](bool reuse) {
            KernelMapCache::instance().clear();
            KernelMapCache::instance().set_part_reuse_bytes(reuse ? size_t(1) << 30 : 0);
            if (reuse) // store the parts from another batch first so this one is composed
                KernelMapCache::instance().get_or_build(make_batch(samples, {0, 1, 2, 3}, 6, cfg),
                                                        kc);
            auto x = batch.features().detach().clone().requires_grad_(true);
            auto wr = w.detach().clone().requires_grad_(true);
            auto y = adaptive_convolution(batch.with_features(x), wr, kc).features();
            (y * g).sum().backward();
            return std::vector<at::Tensor>{y.detach(), x.grad(), wr.grad()};
        };
        const auto a = run(false);
        PartReuse on(size_t(1) << 30);
        const auto b = run(true);
        CHECK(KernelMapCache::instance().stats().part_hits == size_t(4));
        for (size_t i = 0; i < a.size(); ++i)
            CHECK(torch::equal(a[i], b[i]));
    }
}

TEST(part_reuse_convolution_cpu_forward_backward_identical) { check_conv(torch::kCPU); }

TEST(part_reuse_convolution_cuda_forward_backward_identical) {
    if (!torch::cuda::is_available()) return;
    check_conv(torch::kCUDA);
}
