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

// Samples side by side along the first axis: sample k moved by start + stride * k level-0 cells.
AdaptiveSparseTensor make_batch(const std::vector<AdaptiveSparseTensor> &samples,
                                const std::vector<int> &order, int64_t stride,
                                const AdaptiveVoxelConfig &cfg, int64_t start = 0) {
    std::vector<at::Tensor> cs, ls, fs;
    for (size_t k = 0; k < order.size(); ++k) {
        const auto &s = samples[order[k]];
        auto c = s.coordinates().clone();
        auto shift = (torch::pow(2.0, s.levels().to(torch::kFloat64)) *
                      static_cast<double>(start + stride * static_cast<int64_t>(k)))
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
// build), many new (full build, new ones cut), all stored (composed only), few new (built
// alone, then composed).
TEST(part_reuse_matches_full_build_modes_shifts_kernels) {
    const std::vector<std::vector<int>> batches{
        {0, 1, 2}, {2, 3, 0, 4}, {4, 2, 3, 0, 1}, {1, 5, 0, 2, 3, 4, 0, 1}};
    for (int D : {1, 2, 3}) {
        const auto cfg = make_config(D, 2.0, -3.0);
        std::vector<AdaptiveSparseTensor> samples;
        for (int s = 0; s < 6; ++s)
            samples.push_back(make_sample(10 + s, D, cfg));
        // Expected hits per batch: samples equal (in content) to one of an earlier batch.
        // In 1-2 D some random samples coincide, which correctly hits.
        auto same = [&](int i, int j) {
            return torch::equal(samples[i].coordinates(), samples[j].coordinates()) &&
                   torch::equal(samples[i].levels(), samples[j].levels());
        };
        std::vector<size_t> expect_hits;
        std::vector<int> seen;
        for (const auto &b : batches) {
            size_t h = 0;
            for (int i : b)
                for (int j : seen)
                    if (same(i, j)) {
                        ++h;
                        break;
                    }
            expect_hits.push_back(h);
            seen.insert(seen.end(), b.begin(), b.end());
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
                    std::vector<AdaptiveSparseTensor> bt;
                    std::vector<KernelMapEntry> expected;
                    for (const auto &b : batches) {
                        bt.push_back(make_batch(samples, b, stride, cfg));
                        expected.push_back(full_build(bt.back(), kc));
                    }
                    PartReuse on(size_t(1) << 30);
                    auto &cache = KernelMapCache::instance();
                    size_t hits = 0, lookups = 0;
                    for (size_t i = 0; i < batches.size(); ++i) {
                        check_same_entry(cache.get_or_build(bt[i], kc), expected[i]);
                        if (K == 1) // radius 0: every occupied level-0 cell is a part of its own
                            continue;
                        hits += expect_hits[i];
                        lookups += batches[i].size();
                        const auto st = cache.stats();
                        CHECK_EQ(st.part_hits, hits);
                        CHECK_EQ(st.part_misses, lookups - hits);
                    }
                    CHECK(cache.stats().part_hits > 0);
                    CHECK(cache.stats().part_bytes > 0);
                }
            }
        }
    }
}

// Too small a gap (a tap could reach the next sample): the two samples stay one part. A single
// sample is one part as well, and is matched again wherever it reappears.
TEST(part_reuse_needs_a_gap_wider_than_the_kernel_radius) {
    const auto cfg = make_config(3, 2.0, 0.0);
    std::vector<AdaptiveSparseTensor> samples{make_sample(1, 3, cfg), make_sample(2, 3, cfg)};
    for (int K : {3, 5}) {
        AdaptiveKernelConfig kc;
        kc.kernel_size = K;
        const int r = (K - 1) / 2;
        // Samples occupy level-0 x cells 0..3, so a stride of 3 + r leaves r - 1 empty cells.
        auto tight = make_batch(samples, {0, 1}, 3 + r, cfg);
        auto single = make_batch(samples, {0}, 4, cfg);
        auto moved = make_batch(samples, {0}, 4, cfg, 1000);
        const auto et = full_build(tight, kc), es = full_build(single, kc),
                   em = full_build(moved, kc);
        PartReuse on(size_t(1) << 30);
        auto &cache = KernelMapCache::instance();
        check_same_entry(cache.get_or_build(tight, kc), et); // one part, new
        auto st = cache.stats();
        CHECK_EQ(st.part_hits, size_t(0));
        CHECK_EQ(st.part_misses, size_t(1));
        check_same_entry(cache.get_or_build(single, kc), es); // one part, new
        check_same_entry(cache.get_or_build(moved, kc), em);  // the same sample, moved: reused
        st = cache.stats();
        CHECK_EQ(st.part_hits, size_t(1));
        CHECK_EQ(st.part_misses, size_t(2));
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

// Device composition (CUDA batches, detail::cuda_part_compose on): the batch map is written on
// the device and its host arrays stay undefined until asked for. Both the device slot and the
// host arrays filled on request equal a full build pair for pair; the off switch composes on
// the host as before.
TEST(part_reuse_device_compose_matches_full_build) {
    if (!torch::cuda::is_available()) throw mink_test::Skip("CUDA not available");
    const std::vector<std::vector<int>> batches{
        {0, 1, 2}, {2, 3, 0, 4}, {4, 2, 3, 0, 1}, {1, 5, 0, 2, 3, 4, 0, 1}};
    const auto cfg = make_config(3, 2.0, -3.0);
    std::vector<AdaptiveSparseTensor> samples;
    for (int s = 0; s < 6; ++s)
        samples.push_back(make_sample(70 + s, 3, cfg));
    const torch::Device cuda(torch::kCUDA);
    // 16-bit device indices (default), 32-bit device indices (limit 0), host composition.
    const std::vector<std::pair<bool, int64_t>> variants{{true, 65536}, {true, 0}, {false, 65536}};
    for (const auto &[device_compose, narrow_limit] : variants) {
        minkowski::adaptive::detail::cuda_part_compose() = device_compose;
        minkowski::adaptive::detail::cuda_part_compose_narrow_limit() = narrow_limit;
        for (auto mode : {AdaptiveConvMode::CenterSampled, AdaptiveConvMode::VolumeWeighted}) {
            for (int K : {3, 5}) {
                for (int shift : {0, -1}) {
                    AdaptiveKernelConfig kc;
                    kc.mode = mode;
                    kc.kernel_size = K;
                    kc.output_level_shift = shift;
                    const int64_t stride = 4 + (K - 1) / 2 + 1;
                    std::vector<AdaptiveSparseTensor> bt;
                    std::vector<KernelMapEntry> expected;
                    for (const auto &b : batches) {
                        bt.push_back(make_batch(samples, b, stride, cfg).to(cuda));
                        expected.push_back(full_build(bt.back(), kc));
                    }
                    PartReuse on(size_t(1) << 30);
                    auto &cache = KernelMapCache::instance();
                    int composed_on_device = 0;
                    for (size_t i = 0; i < batches.size(); ++i) {
                        const auto e = cache.get_or_build(bt[i], kc, /*host_arrays=*/false);
                        if (!e.map.in_index.defined()) {
                            ++composed_on_device;
                            CHECK(e.map.kernel_offsets == expected[i].map.kernel_offsets);
                            const auto a = e.device->get(e.map, cuda, torch::kFloat64, true);
                            CHECK(a.in_index.is_cuda());
                            CHECK(torch::equal(a.in_index.cpu(), expected[i].map.in_index));
                            CHECK(torch::equal(a.out_index.cpu(), expected[i].map.out_index));
                            CHECK(torch::equal(a.weight.cpu(), expected[i].map.weight));
                            const auto f = e.device->get(e.map, cuda, torch::kFloat32, true);
                            CHECK(torch::equal(f.weight.cpu(), expected[i].map.weight.to(torch::kFloat32)));
                        }
                        check_same_entry(cache.get_or_build(bt[i], kc), expected[i]);
                    }
                    // Batches 3 and 4 are composed when their new parts are few enough.
                    if (device_compose)
                        CHECK(composed_on_device >= 1);
                    else
                        CHECK_EQ(composed_on_device, 0);
                }
            }
        }
    }
    minkowski::adaptive::detail::cuda_part_compose() = true;
    minkowski::adaptive::detail::cuda_part_compose_narrow_limit() = 65536;
}

// A map composed on the device and then used for the same topology on the CPU (no host arrays
// yet) is copied from the device slot; the CPU convolution matches the one without part reuse.
TEST(part_reuse_device_compose_then_cpu_convolution) {
    if (!torch::cuda::is_available()) throw mink_test::Skip("CUDA not available");
    const auto cfg = make_config(3, 2.0, 0.0);
    std::vector<AdaptiveSparseTensor> samples;
    for (int s = 0; s < 4; ++s)
        samples.push_back(make_sample(90 + s, 3, cfg));
    for (auto mode : {AdaptiveConvMode::CenterSampled, AdaptiveConvMode::VolumeWeighted}) {
        AdaptiveKernelConfig kc;
        kc.mode = mode;
        const auto batch = make_batch(samples, {2, 0, 3, 1}, 6, cfg);
        torch::manual_seed(3);
        const auto w = torch::randn({27, 3, 5}, torch::kFloat64);
        KernelMapCache::instance().clear();
        const auto want = adaptive_convolution(batch, w, kc).features();
        PartReuse on(size_t(1) << 30);
        auto &cache = KernelMapCache::instance();
        cache.get_or_build(make_batch(samples, {0, 1, 2, 3}, 6, cfg), kc); // store the parts
        const auto on_cuda = batch.to(torch::kCUDA);
        CHECK(!cache.get_or_build(on_cuda, kc, false).map.in_index.defined());
        const auto got = adaptive_convolution(on_cuda.to(torch::kCPU), w, kc).features();
        CHECK(torch::equal(got, want));
        const auto y = adaptive_convolution(on_cuda, w.to(torch::kCUDA), kc).features().cpu();
        CHECK_ALLCLOSE(y, want, 1e-12, 1e-12);
    }
}
