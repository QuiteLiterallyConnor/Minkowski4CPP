// detail::sort_keys / from_voxels: every input order gives the keys and permutation of a full
// (key, index) sort, including the O(n) path for keys already sorted within each level (a
// batch of per-sample sorted pieces placed along x).
#include <algorithm>
#include <random>
#include <set>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/adaptive/adaptive_topology.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;
namespace adt = minkowski::adaptive::detail;
using adt::Key;

namespace {

constexpr uint32_t kDepth = 3;

AdaptiveVoxelConfig config() {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {1.0, 1.0, 1.0};
    const double fine = 1.0 / double(1u << kDepth);
    c.min_voxel_size = {fine, fine, fine};
    c.max_depth = kDepth;
    return c;
}

Key key(uint32_t level, int64_t x, int64_t y, int64_t z) {
    Key k{};
    k.level = level;
    k.coordinates[0] = x;
    k.coordinates[1] = y;
    k.coordinates[2] = z;
    return k;
}

// One sample: distinct non-overlapping leaves inside [0, cells)^3 level-0 cells, sorted.
std::vector<Key> sample(std::mt19937_64 &rng, int n, int cells) {
    std::uniform_int_distribution<int64_t> pos(0, (int64_t(cells) << kDepth) - 1);
    std::uniform_int_distribution<uint32_t> lvl(0, kDepth);
    std::set<std::vector<int64_t>> blocked; // (level, x, y, z) of leaves and their ancestors
    std::vector<Key> out;
    for (int i = 0; i < n; ++i) {
        const uint32_t l = lvl(rng);
        const int s = int(kDepth - l);
        const int64_t x = pos(rng) >> s, y = pos(rng) >> s, z = pos(rng) >> s;
        // Leaves are stored as (level, x, y, z), their ancestors as (-1 - level, ...): k may not
        // be a leaf, an ancestor of a leaf, or under a leaf.
        bool ok = !blocked.count({l, x, y, z}) && !blocked.count({-1 - int64_t(l), x, y, z});
        for (uint32_t a = 0; a < l && ok; ++a)
            ok = !blocked.count({a, x >> (l - a), y >> (l - a), z >> (l - a)});
        if (!ok)
            continue;
        blocked.insert({l, x, y, z});
        for (uint32_t a = 0; a < l; ++a)
            blocked.insert({-1 - int64_t(a), x >> (l - a), y >> (l - a), z >> (l - a)});
        out.push_back(key(l, x, y, z));
    }
    std::sort(out.begin(), out.end(), adt::key_less);
    return out;
}

// Samples placed along x (stride level-0 cells apart), concatenated in sample order.
std::vector<Key> batch(std::mt19937_64 &rng, int samples, int n, int cells, int stride) {
    std::vector<Key> all;
    for (int s = 0; s < samples; ++s)
        for (Key k : sample(rng, n, cells)) {
            k.coordinates[0] += int64_t(s) * stride << k.level;
            all.push_back(k);
        }
    return all;
}

void reference(const std::vector<Key> &keys, std::vector<Key> &sorted, std::vector<int64_t> &order) {
    std::vector<int64_t> idx(keys.size());
    for (size_t i = 0; i < idx.size(); ++i)
        idx[i] = static_cast<int64_t>(i);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int64_t a, int64_t b) { return adt::key_less(keys[a], keys[b]); });
    sorted.clear();
    for (int64_t i : idx)
        sorted.push_back(keys[i]);
    order = idx;
}

void check_sort(std::vector<Key> keys) {
    std::vector<Key> want;
    std::vector<int64_t> want_order;
    reference(keys, want, want_order);
    const bool was_sorted = std::is_sorted(keys.begin(), keys.end(), adt::key_less);
    std::vector<int64_t> order;
    adt::sort_keys(keys, order);
    CHECK(keys == want);
    if (was_sorted) {
        CHECK(order.empty()); // nothing moved
    } else {
        CHECK(order == want_order);
    }
}

} // namespace

TEST(sort_keys_batches_sorted_within_levels_match_full_sort) {
    std::mt19937_64 rng(20261102);
    for (int trial = 0; trial < 20; ++trial) {
        const int cells = 2 + trial % 3;
        const auto b = batch(rng, 1 + trial % 9, 20 + 15 * trial, cells, cells + 1 + trial % 3);
        check_sort(b);
    }
}

TEST(sort_keys_large_batch_matches_full_sort) {
    // Above the size where CUDA builds sort on the device (16384 keys).
    std::mt19937_64 rng(5);
    const auto b = batch(rng, 24, 1500, 6, 7);
    CHECK(b.size() > 16384);
    check_sort(b);
}

TEST(sort_keys_unsorted_levels_sorted_and_empty_inputs) {
    std::mt19937_64 rng(9);
    auto b = batch(rng, 4, 200, 3, 4);
    std::vector<Key> sorted = b;
    std::sort(sorted.begin(), sorted.end(), adt::key_less);
    check_sort(sorted);  // already sorted
    std::shuffle(b.begin(), b.end(), rng);
    check_sort(b);       // levels out of order: general path
    std::vector<Key> one_swap = sorted;
    std::swap(one_swap[1], one_swap[one_swap.size() - 2]);
    check_sort(one_swap);
    check_sort({});
    check_sort({key(2, 1, 1, 1)});
}

TEST(sort_keys_from_voxels_batch_matches_presorted_input) {
    std::mt19937_64 rng(13);
    const auto b = batch(rng, 8, 300, 3, 5);
    const int64_t n = static_cast<int64_t>(b.size());
    auto coords = torch::empty({n, 3}, torch::kInt64);
    auto levels = torch::empty({n}, torch::kInt32);
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d)
            coords[i][d] = b[i].coordinates[d];
        levels[i] = static_cast<int32_t>(b[i].level);
    }
    torch::manual_seed(1);
    auto feats = torch::randn({n, 4}, torch::kFloat32);
    const auto t = AdaptiveSparseTensor::from_voxels(coords, levels, feats, config());
    std::vector<Key> want;
    std::vector<int64_t> perm;
    reference(b, want, perm);
    const auto p = torch::tensor(perm, torch::kInt64);
    const auto u = AdaptiveSparseTensor::from_voxels(coords.index_select(0, p),
                                                     levels.index_select(0, p),
                                                     feats.index_select(0, p), config());
    CHECK(torch::equal(t.coordinates(), u.coordinates()));
    CHECK(torch::equal(t.levels(), u.levels()));
    CHECK(torch::equal(t.features(), u.features()));
}

TEST(sort_keys_duplicates_still_rejected) {
    auto coords = torch::tensor({{1, 0, 0}, {0, 0, 0}, {1, 0, 0}}, torch::kInt64);
    auto levels = torch::tensor({1, 0, 1}, torch::kInt32);
    CHECK_THROWS(AdaptiveSparseTensor::from_voxels(coords, levels, torch::zeros({3, 1}), config()));
}
