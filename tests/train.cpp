// ============================================================================
// train.cpp — C++ Sparse 4D CNN Trainer
//
//
// Dependencies:
//   - nlohmann/json (single-header JSON library)
//     Download json.hpp from:
//       https://github.com/nlohmann/json/releases/latest/download/json.hpp
//     Place in: third_party/json.hpp
//
// Usage:
//   ./train --data_dir /path/to/data [options]
//
// Options:
//   --data_dir DIR          Path to training data directory (required)
//   --epochs N              Number of training epochs (default: 10)
//   --batch_size N          Batch size (default: 1)
//   --radius N              Labeling radius in voxels (default: 2)
//   --pos_weight F          Positive class weight for BCE loss (default: 12.0)
//   --lr F                  Learning rate (default: 1e-3)
//   --weight_decay F        Weight decay (default: 1e-4)
//   --save_dir DIR          Checkpoint save directory (default: checkpoints)
//   --clip_len N            Number of frames per clip (default: 10)
//   --c_mid N               Mid-level channels (default: 32)
//   --help                  Show usage
// ============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <regex>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <cassert>
#include <filesystem>
#include <functional>
#include <optional>
#include <random>
#include <iomanip>

#include <torch/torch.h>

#include "third_party/json.hpp"

#include "MinkowskiEngine/minkowski.hpp"

using namespace minkowski;
using json = nlohmann::json;
namespace fs = std::filesystem;


constexpr int MAX_REMOVAL_R = 8;

struct TrainConfig {
    std::string data_dir;
    int epochs = 10;
    int batch_size = 1;
    int radius = 2;
    float pos_weight = 12.0f;
    float lr = 1e-3f;
    float weight_decay = 1e-4f;
    std::string save_dir = "checkpoints";
    int clip_len = 10;
    int c_mid = 32;
};

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " --data_dir DIR [options]\n\n"
        << "Sparse 4D CNN training (C++ MinkowskiEngine)\n\n"
        << "Required:\n"
        << "  --data_dir DIR          Training data directory\n\n"
        << "Options:\n"
        << "  --epochs N              Number of epochs (default: 10)\n"
        << "  --batch_size N          Batch size (default: 1)\n"
        << "  --radius N              Labeling radius in voxels (default: 2)\n"
        << "  --pos_weight F          BCE positive weight (default: 12.0)\n"
        << "  --lr F                  Learning rate (default: 1e-3)\n"
        << "  --weight_decay F        Weight decay (default: 1e-4)\n"
        << "  --save_dir DIR          Checkpoint directory (default: checkpoints)\n"
        << "  --clip_len N            Frames per clip (default: 10)\n"
        << "  --c_mid N               Mid-level channels (default: 32)\n"
        << "  --help                  Show this message\n";
}

TrainConfig parse_args(int argc, char** argv) {
    TrainConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << key << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
        std::string val = argv[++i];

        if      (key == "--data_dir")        cfg.data_dir = val;
        else if (key == "--epochs")          cfg.epochs = std::stoi(val);
        else if (key == "--batch_size")      cfg.batch_size = std::stoi(val);
        else if (key == "--radius")          cfg.radius = std::stoi(val);
        else if (key == "--pos_weight")      cfg.pos_weight = std::stof(val);
        else if (key == "--lr")              cfg.lr = std::stof(val);
        else if (key == "--weight_decay")    cfg.weight_decay = std::stof(val);
        else if (key == "--save_dir")        cfg.save_dir = val;
        else if (key == "--clip_len")        cfg.clip_len = std::stoi(val);
        else if (key == "--c_mid")           cfg.c_mid = std::stoi(val);
        else {
            std::cerr << "Unknown argument: " << key << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    if (cfg.data_dir.empty()) {
        std::cerr << "Error: --data_dir is required\n";
        print_usage(argv[0]);
        std::exit(1);
    }
    return cfg;
}

// ============================================================================
// Data Structures for Parsed JSON Frames
// ============================================================================

struct FrameVoxel {
    std::array<int32_t, 3> coordinates;
    float intensity;
    int num_cameras;
};

struct FrameTarget {
    std::array<float, 3> position_m;
};

struct GridInfo {
    std::array<float, 3> origin_m = {0.0f, 0.0f, 0.0f};
    float voxel_size_m = 1.0f;
};

struct FrameData {
    std::vector<FrameVoxel> voxels;
    std::vector<FrameTarget> targets;
    GridInfo grid_info;
};

// ============================================================================
// Data Loading Utilities
// ============================================================================

FrameData parse_json_frame(const std::string& path) {
    FrameData frame;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    json j = json::parse(ifs);

    // Parse grid_info
    if (j.contains("grid_info")) {
        auto& gi = j["grid_info"];
        if (gi.contains("origin_m")) {
            auto orig = gi["origin_m"].get<std::vector<float>>();
            for (int i = 0; i < 3 && i < static_cast<int>(orig.size()); ++i)
                frame.grid_info.origin_m[i] = orig[i];
        }
        frame.grid_info.voxel_size_m = gi.value("voxel_size_m", 1.0f);
    }

    // Parse voxels
    if (j.contains("voxels")) {
        for (auto& v : j["voxels"]) {
            FrameVoxel fv;
            auto coords = v["coordinates"].get<std::vector<int>>();
            for (int i = 0; i < 3; ++i) fv.coordinates[i] = coords[i];
            fv.intensity = v.value("intensity", 0.0f);
            fv.num_cameras = v.value("num_cameras", 1);
            frame.voxels.push_back(fv);
        }
    }

    // Parse targets
    if (j.contains("targets")) {
        for (auto& t : j["targets"]) {
            FrameTarget ft;
            auto pos = t["position_m"].get<std::vector<float>>();
            for (int i = 0; i < 3; ++i) ft.position_m[i] = pos[i];
            frame.targets.push_back(ft);
        }
    }

    return frame;
}

float normalize_num_cameras(int num_cameras, int max_cameras = 10) {
    return static_cast<float>(std::min(num_cameras, max_cameras))
         / static_cast<float>(max_cameras);
}

std::array<int32_t, 3> meters_to_grid(
    const std::array<float, 3>& position_m,
    const std::array<float, 3>& origin_m,
    float voxel_size_m)
{
    std::array<int32_t, 3> result;
    for (int i = 0; i < 3; ++i) {
        result[i] = static_cast<int32_t>(
            std::floor((position_m[i] - origin_m[i]) / voxel_size_m));
    }
    return result;
}

/// Return maximum r >= 1 such that kept length with pattern (r+1) >= clip_len.
/// Uses binary search to avoid scanning many r values for long runs.
int max_r_for_L(int L, int clip_len) {
    if (L < clip_len) return 0;
    int low = 1, high = L, best = 0;
    while (low <= high) {
        int mid = (low + high) / 2;
        int kept = (L + mid) / (mid + 1);
        if (kept >= clip_len) {
            best = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return best;
}

/// Return all contiguous clip_len-frame segments (sliding window) found within
/// a single directory. Optionally generates block-removal subsampled segments
/// (remove r frames then keep 1, repeat).
std::vector<std::vector<std::string>> find_segments_in_dir(
    const std::string& frame_dir,
    int clip_len = 10,
    bool subsample = true,
    bool debug = false)
{
    std::vector<std::vector<std::string>> segments;
    if (!fs::exists(frame_dir) || !fs::is_directory(frame_dir)) return segments;

    // Find all JSON files and extract numeric indices
    std::map<int, std::string> idx_to_path;
    std::regex num_re(R"((\d+))");

    int json_count = 0;
    for (auto& entry : fs::directory_iterator(frame_dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext != ".json") continue;
        
        json_count++;
        auto stem = entry.path().stem().string();
        std::smatch match;
        if (std::regex_search(stem, match, num_re)) {
            int idx = std::stoi(match[1].str());
            if (idx_to_path.find(idx) == idx_to_path.end()) {
                idx_to_path[idx] = entry.path().string();
                if (debug && json_count <= 5) {
                    std::cout << "  [DEBUG] Found: " << stem << " -> index " << idx << "\n";
                }
            }
        } else if (debug && json_count <= 5) {
            std::cout << "  [DEBUG] No numeric index in: " << stem << "\n";
        }
    }

    if (debug) {
        std::cout << "  [DEBUG] Dir: " << frame_dir << " - " << json_count 
                  << " JSON files, " << idx_to_path.size() << " with indices\n";
    }

    if (idx_to_path.empty()) return segments;

    // Extract sorted indices
    std::vector<int> matched_indices;
    for (auto& [idx, _path] : idx_to_path) matched_indices.push_back(idx);
    std::sort(matched_indices.begin(), matched_indices.end());

    // Split into runs of consecutive indices
    std::vector<std::vector<int>> runs;
    std::vector<int> current_run = {matched_indices[0]};
    for (size_t i = 1; i < matched_indices.size(); ++i) {
        if (matched_indices[i] == current_run.back() + 1) {
            current_run.push_back(matched_indices[i]);
        } else {
            runs.push_back(current_run);
            current_run = {matched_indices[i]};
        }
    }
    runs.push_back(current_run);

    if (debug) {
        std::cout << "  [DEBUG] Found " << runs.size() << " consecutive runs:\n";
        for (size_t r = 0; r < runs.size() && r < 10; ++r) {
            std::cout << "  [DEBUG]   Run " << r << ": " << runs[r].size() 
                      << " frames [" << runs[r].front() << ".." << runs[r].back() << "]\n";
        }
        std::cout << "  [DEBUG] Need >= " << clip_len << " consecutive frames for a clip\n";
    }

    // Use set of index tuples for deduplication
    std::set<std::vector<int>> segments_idx_set;

    for (auto& run : runs) {
        int L = static_cast<int>(run.size());

        if (debug && L >= clip_len) {
            std::cout << "  [DEBUG] Run of length " << L << " can produce segments\n";
        }

        // Contiguous sliding windows
        for (int i = 0; i <= L - clip_len; ++i) {
            std::vector<int> seg_idx(run.begin() + i, run.begin() + i + clip_len);
            if (segments_idx_set.insert(seg_idx).second) {
                std::vector<std::string> seg_paths;
                seg_paths.reserve(clip_len);
                for (int idx : seg_idx) seg_paths.push_back(idx_to_path[idx]);
                segments.push_back(std::move(seg_paths));
            }
        }

        // Block-removal subsampling: consider r in [1, r_max]
        if (subsample) {
            int r_max = std::min(max_r_for_L(L, clip_len), MAX_REMOVAL_R);
            for (int r = 1; r <= r_max; ++r) {
                std::vector<int> positions;
                for (int p = 0; p < L; p += r + 1) positions.push_back(p);
                int kept_len = static_cast<int>(positions.size());
                if (kept_len < clip_len) continue;

                for (int i = 0; i <= kept_len - clip_len; ++i) {
                    std::vector<int> seg_idx;
                    seg_idx.reserve(clip_len);
                    for (int j = i; j < i + clip_len; ++j) {
                        seg_idx.push_back(run[positions[j]]);
                    }
                    if (segments_idx_set.insert(seg_idx).second) {
                        std::vector<std::string> seg_paths;
                        seg_paths.reserve(clip_len);
                        for (int idx : seg_idx) seg_paths.push_back(idx_to_path[idx]);
                        segments.push_back(std::move(seg_paths));
                    }
                }
            }
        }
    }

    return segments;
}

/// Recursively search root_dir and its subdirectories for all contiguous
/// clip_len-frame segments, including optional block-removal subsampled
/// variants.
std::vector<std::vector<std::string>> find_all_segments(
    const std::string& root_dir,
    int clip_len = 10,
    bool subsample = true,
    bool debug = false)
{
    // Collect all directories (including root)
    std::vector<std::string> dirs;
    dirs.push_back(root_dir);
    try {
        for (auto& entry : fs::recursive_directory_iterator(root_dir, 
                fs::directory_options::skip_permission_denied)) {
            if (entry.is_directory()) {
                dirs.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[WARNING] Filesystem error during scan: " << e.what() << "\n";
    }

    std::cout << "[INFO] Scanning " << dirs.size() << " directories...\n";
    if (debug) {
        std::cout << "[DEBUG] Root: " << root_dir << "\n";
        for (size_t i = 0; i < std::min(size_t(5), dirs.size()); ++i) {
            std::cout << "[DEBUG]   Dir " << i << ": " << dirs[i] << "\n";
        }
    }
    std::vector<std::vector<std::string>> all_segments;

    for (size_t i = 0; i < dirs.size(); ++i) {
        auto segs = find_segments_in_dir(dirs[i], clip_len, subsample, debug);
        if (!segs.empty()) {
            all_segments.insert(all_segments.end(), segs.begin(), segs.end());
        }
        if ((i + 1) % 100 == 0 || !segs.empty() || i + 1 == dirs.size()) {
            std::cout << "\r  Scanned " << i + 1 << "/" << dirs.size()
                      << " dirs, " << all_segments.size() << " segments found"
                      << std::flush;
        }
    }
    std::cout << "\n";
    return all_segments;
}

/// Generate binary labels ONLY at a single reference frame (t_ref),
/// producing a spatial sphere rather than a motion-smeared tube.
void compute_labels_for_points(
    const std::vector<std::array<int32_t, 4>>& coords_xyzt,
    const std::map<int, std::vector<std::array<int32_t, 3>>>& targets_per_t,
    int radius_vox,
    int clip_len,
    std::vector<float>& labels)
{
    labels.assign(coords_xyzt.size(), 0.0f);
    int t_ref = clip_len / 2;

    auto it = targets_per_t.find(t_ref);
    if (it == targets_per_t.end()) return;

    const auto& centers = it->second;
    int r2 = radius_vox * radius_vox;

    for (size_t i = 0; i < coords_xyzt.size(); ++i) {
        if (coords_xyzt[i][3] != t_ref) continue;

        for (const auto& center : centers) {
            int dx = coords_xyzt[i][0] - center[0];
            int dy = coords_xyzt[i][1] - center[1];
            int dz = coords_xyzt[i][2] - center[2];
            int d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= r2) {
                labels[i] = 1.0f;
                break;
            }
        }
    }
}

// ============================================================================
// Clip Sample
// ============================================================================

struct ClipSample {
    at::Tensor coordinates; // [N, 4] int32 (ix, iy, iz, it) — no batch index yet
    at::Tensor features;    // [N, 3] float32 (intensity_norm, num_cameras_norm, t_norm)
    at::Tensor labels;      // [N, 1] float32 (binary)
};

/// Build a single clip sample from parsed frames.
///   coords_xyzt: [N,4] int32  => (ix, iy, iz, it)
///   feats:       [N,3] float32 => (intensity_norm, num_cameras_norm, t_norm)
///   labels:      [N,1] float32 => (0/1)
std::optional<ClipSample> build_clip_arrays(
    const std::vector<FrameData>& frames,
    const GridInfo& grid_info,
    int radius_vox,
    int clip_len)
{
    std::vector<std::array<int32_t, 4>> coords_list;
    std::vector<float> intensity_list;
    std::vector<float> num_cameras_list;
    std::vector<float> tnorm_list;
    std::map<int, std::vector<std::array<int32_t, 3>>> targets_per_t;

    // Collect target centers per frame
    for (int it = 0; it < static_cast<int>(frames.size()); ++it) {
        if (!frames[it].targets.empty()) {
            std::vector<std::array<int32_t, 3>> centers;
            for (const auto& tgt : frames[it].targets) {
                centers.push_back(meters_to_grid(
                    tgt.position_m, grid_info.origin_m, grid_info.voxel_size_m));
            }
            targets_per_t[it] = centers;
        }
    }

    // Collect voxels across all frames
    for (int it = 0; it < static_cast<int>(frames.size()); ++it) {
        float t_norm = static_cast<float>(it)
                     / static_cast<float>(std::max(clip_len - 1, 1));
        for (const auto& v : frames[it].voxels) {
            coords_list.push_back(
                {v.coordinates[0], v.coordinates[1], v.coordinates[2], it});
            intensity_list.push_back(v.intensity);
            num_cameras_list.push_back(normalize_num_cameras(v.num_cameras));
            tnorm_list.push_back(t_norm);
        }
    }

    if (coords_list.empty()) return std::nullopt;

    int N = static_cast<int>(coords_list.size());

    // Normalize intensity across the clip
    float mu = 0.0f, sigma = 1e-6f;
    if (N > 1) {
        mu = std::accumulate(intensity_list.begin(), intensity_list.end(), 0.0f)
           / static_cast<float>(N);
        float var = 0.0f;
        for (float v : intensity_list) var += (v - mu) * (v - mu);
        sigma = std::sqrt(var / static_cast<float>(N)) + 1e-6f;
    }

    // Build coordinate tensor [N, 4] int32
    auto coords_tensor = torch::zeros({N, 4}, torch::kInt32);
    auto coords_acc = coords_tensor.accessor<int32_t, 2>();

    // Build feature tensor [N, 3] float32
    auto feats_tensor = torch::zeros({N, 3}, torch::kFloat32);
    auto feats_acc = feats_tensor.accessor<float, 2>();

    for (int i = 0; i < N; ++i) {
        coords_acc[i][0] = coords_list[i][0]; // ix
        coords_acc[i][1] = coords_list[i][1]; // iy
        coords_acc[i][2] = coords_list[i][2]; // iz
        coords_acc[i][3] = coords_list[i][3]; // it

        feats_acc[i][0] = (intensity_list[i] - mu) / sigma; // normalized intensity
        feats_acc[i][1] = num_cameras_list[i];               // num_cameras norm
        feats_acc[i][2] = tnorm_list[i];                     // t_norm
    }

    // Compute labels [N, 1]
    std::vector<float> labels_vec;
    compute_labels_for_points(coords_list, targets_per_t, radius_vox,
                              clip_len, labels_vec);
    auto labels_tensor = torch::from_blob(
        labels_vec.data(), {N, 1}, torch::kFloat32).clone();

    return ClipSample{coords_tensor, feats_tensor, labels_tensor};
}

/// Load all clip samples from a data directory.
std::vector<ClipSample> load_dataset(
    const std::string& data_dir,
    int radius_vox,
    int clip_len)
{
    auto segments = find_all_segments(data_dir, clip_len, /*subsample=*/true, /*debug=*/true);
    std::cout << "[INFO] Found " << segments.size()
              << " segments (including subsampled)\n";

    if (segments.empty()) {
        throw std::runtime_error("No segments found in " + data_dir);
    }

    std::vector<ClipSample> samples;
    int skipped = 0;

    for (size_t s = 0; s < segments.size(); ++s) {
        if (s % 100 == 0 || s + 1 == segments.size()) {
            std::cout << "\r  Processing segment " << s + 1 << "/"
                      << segments.size() << " (" << samples.size()
                      << " valid)" << std::flush;
        }

        try {
            std::vector<FrameData> frames;
            frames.reserve(segments[s].size());
            for (const auto& path : segments[s]) {
                frames.push_back(parse_json_frame(path));
            }
            GridInfo gi = frames[0].grid_info;

            auto sample = build_clip_arrays(frames, gi, radius_vox, clip_len);
            if (sample.has_value()) {
                samples.push_back(std::move(sample.value()));
            } else {
                ++skipped;
            }
        } catch (const std::exception& e) {
            std::cerr << "\n[WARNING] Failed segment " << s << ": "
                      << e.what() << "\n";
            ++skipped;
        }
    }

    std::cout << "\n[INFO] Built " << samples.size() << " samples (skipped "
              << skipped << ")\n";
    return samples;
}

// ============================================================================
// Collation
// ============================================================================

struct CollatedBatch {
    at::Tensor coordinates; // [sum_N, 5] int32 (batch_idx, ix, iy, iz, it)
    at::Tensor features;    // [sum_N, 3] float32
    at::Tensor labels;      // [sum_N, 1] float32
};

/// Collate a vector of ClipSamples into a single batch by prepending batch
/// indices to coordinates and concatenating everything.
CollatedBatch sparse_collate_fn(const std::vector<ClipSample>& samples) {
    std::vector<at::Tensor> coords_list, feats_list, labels_list;
    coords_list.reserve(samples.size());
    feats_list.reserve(samples.size());
    labels_list.reserve(samples.size());

    for (int64_t b = 0; b < static_cast<int64_t>(samples.size()); ++b) {
        const auto& s = samples[b];
        int64_t N = s.coordinates.size(0);

        // Prepend batch index column => [N, 1+4] = [N, 5]
        auto batch_col = torch::full({N, 1}, b, torch::kInt32);
        coords_list.push_back(torch::cat({batch_col, s.coordinates}, /*dim=*/1));
        feats_list.push_back(s.features);
        labels_list.push_back(s.labels);
    }

    return CollatedBatch{
        torch::cat(coords_list, /*dim=*/0),
        torch::cat(feats_list, /*dim=*/0),
        torch::cat(labels_list, /*dim=*/0)
    };
}

// ============================================================================
// Model Blocks
// ============================================================================

// Conv-BN-ReLU block for sparse 4D tensors
struct ConvBNReLUImpl : torch::nn::Module {
    MinkowskiConvolution conv_{nullptr};
    MinkowskiBatchNorm bn_{nullptr};
    MinkowskiReLU relu_;

    ConvBNReLUImpl(int c_in, int c_out, int dim = 4,
                   int ks = 3, int stride = 1, int dilation = 1)
    {
        conv_ = register_module("conv", MinkowskiConvolution(
            c_in, c_out, ks, stride, dilation,
            /*bias=*/false, /*expand_coordinates=*/false,
            ConvolutionMode::DEFAULT, dim));
        bn_ = register_module("bn", MinkowskiBatchNorm(c_out));
        relu_ = register_module("relu", MinkowskiReLU());
    }

    SparseTensor forward(SparseTensor x) {
        return relu_->forward(bn_->forward(conv_->forward(x)));
    }
};
TORCH_MODULE(ConvBNReLU);

// Residual block with 2x conv3x3 + BN + skip connection
struct ResidualBlockImpl : torch::nn::Module {
    ConvBNReLU conv1_{nullptr};
    MinkowskiConvolution conv2_{nullptr};
    MinkowskiBatchNorm bn2_{nullptr};
    MinkowskiReLU relu_;

    ResidualBlockImpl(int c, int dim = 4) {
        conv1_ = register_module("conv1", ConvBNReLU(c, c, dim));
        conv2_ = register_module("conv2", MinkowskiConvolution(
            c, c, /*ks=*/3, /*stride=*/1, /*dilation=*/1,
            /*bias=*/false, /*expand_coordinates=*/false,
            ConvolutionMode::DEFAULT, dim));
        bn2_ = register_module("bn2", MinkowskiBatchNorm(c));
        relu_ = register_module("relu", MinkowskiReLU());
    }

    SparseTensor forward(SparseTensor x) {
        auto out = conv1_->forward(x);
        out = bn2_->forward(conv2_->forward(out));
        out = out + x; // residual
        return relu_->forward(out);
    }
};
TORCH_MODULE(ResidualBlock);

// ============================================================================
// Simple 4D UNet (CNN only)
// ============================================================================

struct Simple4DUNetImpl : torch::nn::Module {
    int dim_;

    // Encoder
    ConvBNReLU stem_{nullptr};
    ResidualBlock enc1_{nullptr};
    MinkowskiConvolution down1_{nullptr};
    ResidualBlock enc2_{nullptr};
    MinkowskiConvolution down2_{nullptr};
    ResidualBlock bottleneck_{nullptr};

    // Decoder
    MinkowskiConvolutionTranspose up1_{nullptr};
    ResidualBlock dec1_{nullptr};
    MinkowskiConvolutionTranspose up2_{nullptr};
    ResidualBlock dec2_{nullptr};

    // Head
    MinkowskiConvolution head_{nullptr};

    Simple4DUNetImpl(
        int c_in = 3,
        int c_mid = 24,
        int dim = 4)
        : dim_(dim)
    {
        int Cb = c_mid * 4; // bottleneck channels

        // --- Encoder ---
        stem_ = register_module("stem", ConvBNReLU(c_in, c_mid, dim));
        enc1_ = register_module("enc1", ResidualBlock(c_mid, dim));
        down1_ = register_module("down1", MinkowskiConvolution(
            c_mid, c_mid * 2, /*ks=*/2, /*stride=*/2, /*dilation=*/1,
            /*bias=*/false, /*expand_coordinates=*/false,
            ConvolutionMode::DEFAULT, dim));
        enc2_ = register_module("enc2", ResidualBlock(c_mid * 2, dim));
        down2_ = register_module("down2", MinkowskiConvolution(
            c_mid * 2, Cb, /*ks=*/2, /*stride=*/2, /*dilation=*/1,
            /*bias=*/false, /*expand_coordinates=*/false,
            ConvolutionMode::DEFAULT, dim));
        bottleneck_ = register_module("bottleneck", ResidualBlock(Cb, dim));

        // --- Decoder ---
        up1_ = register_module("up1", MinkowskiConvolutionTranspose(
            Cb, c_mid * 2, /*ks=*/2, /*stride=*/2, /*dilation=*/1,
            /*bias=*/false, /*expand_coordinates=*/false,
            ConvolutionMode::DEFAULT, dim));
        dec1_ = register_module("dec1", ResidualBlock(c_mid * 2, dim));
        up2_ = register_module("up2", MinkowskiConvolutionTranspose(
            c_mid * 2, c_mid, /*ks=*/2, /*stride=*/2, /*dilation=*/1,
            /*bias=*/false, /*expand_coordinates=*/false,
            ConvolutionMode::DEFAULT, dim));
        dec2_ = register_module("dec2", ResidualBlock(c_mid, dim));

        // --- Head ---
        head_ = register_module("head", MinkowskiConvolution(
            c_mid, 1, /*ks=*/1, /*stride=*/1, /*dilation=*/1,
            /*bias=*/true, /*expand_coordinates=*/false,
            ConvolutionMode::DEFAULT, dim));
    }

    SparseTensor forward(SparseTensor x) {
        // --- Encoder ---
        auto x0 = stem_->forward(x);
        auto x1 = enc1_->forward(x0);
        auto x2 = enc2_->forward(down1_->forward(x1));
        auto x3 = bottleneck_->forward(down2_->forward(x2));

        // --- Decoder with skip connections ---
        auto u1 = up1_->forward(x3);
        auto d1 = dec1_->forward(u1 + x2);  // skip from encoder stride-2
        auto u2 = up2_->forward(d1);
        auto d2 = dec2_->forward(u2 + x1);  // skip from encoder stride-1

        // --- Head ---
        return head_->forward(d2); // SparseTensor with .F() [N, 1]
    }
};
TORCH_MODULE(Simple4DUNet);

// ============================================================================
// Training
// ============================================================================

double train_one_epoch(
    Simple4DUNet& model,
    const std::vector<ClipSample>& samples,
    torch::optim::Optimizer& optimizer,
    torch::Device device,
    CoordinateMapBackend::Type backend,
    float pos_weight,
    int batch_size)
{
    model->train();
    double total_loss = 0.0;
    int num_batches = 0;

    // Shuffle sample indices
    std::vector<int> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0);
    static std::mt19937 rng(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);

    // Loss function with positive-class weighting
    auto pw = torch::tensor({pos_weight},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto criterion = torch::nn::BCEWithLogitsLoss(
        torch::nn::BCEWithLogitsLossOptions().pos_weight(pw));

    int total_batches =
        (static_cast<int>(samples.size()) + batch_size - 1) / batch_size;

    for (int i = 0; i < static_cast<int>(samples.size()); i += batch_size) {
        int end = std::min(i + batch_size,
                           static_cast<int>(samples.size()));

        // Gather batch samples
        std::vector<ClipSample> batch_samples;
        batch_samples.reserve(end - i);
        for (int j = i; j < end; ++j) {
            batch_samples.push_back(samples[indices[j]]);
        }
        auto batch = sparse_collate_fn(batch_samples);

        // Move to device
        auto coords = batch.coordinates.to(device);
        auto feats  = batch.features.to(device);
        auto labels = batch.labels.to(device);

        // Fresh coordinate manager per forward pass
        auto mgr = std::make_shared<CoordinateManager>(4, backend);
        SparseTensor input(feats, coords, mgr);

        // Forward
        auto output = model->forward(input);
        auto logits = output.F(); // [N, 1]

        // Loss
        auto loss = criterion(logits, labels);

        // Backward
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        total_loss += loss.item<double>();
        ++num_batches;

        std::cout << "\r  Batch " << num_batches << "/" << total_batches
                  << " | loss=" << std::fixed << std::setprecision(4)
                  << (total_loss / num_batches) << std::flush;
    }
    std::cout << "\n";

    return total_loss / std::max(num_batches, 1);
}

/// Evaluate model on samples, returning positive-class recall (matching the
/// Python evaluate() behaviour).
double evaluate(
    Simple4DUNet& model,
    const std::vector<ClipSample>& samples,
    torch::Device device,
    CoordinateMapBackend::Type backend,
    int batch_size)
{
    // Keep model in train mode so that BatchNorm uses per-sample statistics
    // rather than running statistics.  With batch_size=1 the running stats
    // diverge significantly from the per-sample stats the model was trained
    // against, causing predictions to collapse toward all-negative.
    model->train();
    torch::NoGradGuard no_grad;

    int64_t total_pos = 0, correct_pos = 0;

    for (int i = 0; i < static_cast<int>(samples.size()); i += batch_size) {
        int end = std::min(i + batch_size,
                           static_cast<int>(samples.size()));

        std::vector<ClipSample> batch_samples(
            samples.begin() + i, samples.begin() + end);
        auto batch = sparse_collate_fn(batch_samples);

        auto coords = batch.coordinates.to(device);
        auto feats  = batch.features.to(device);
        auto labels = batch.labels.to(device);

        auto mgr = std::make_shared<CoordinateManager>(4, backend);
        SparseTensor input(feats, coords, mgr);

        auto output = model->forward(input);
        auto logits = output.F();
        auto probs  = torch::sigmoid(logits);
        auto preds  = (probs >= 0.5f).to(torch::kFloat32);

        total_pos   += labels.sum().item<int64_t>();
        correct_pos += (preds * labels).sum().item<int64_t>();
    }

    return total_pos > 0
        ? static_cast<double>(correct_pos) / static_cast<double>(total_pos)
        : 0.0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    auto cfg = parse_args(argc, argv);

    // Create save directory
    fs::create_directories(cfg.save_dir);

    // Device selection
#ifdef CPU_ONLY
    auto device  = torch::kCPU;
    auto backend = CoordinateMapBackend::CPU;
    std::cout << "Using device: cpu (CPU_ONLY build)\n";
#else
    auto device  = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    auto backend = torch::cuda::is_available()
        ? CoordinateMapBackend::CUDA
        : CoordinateMapBackend::CPU;
    std::cout << "Using device: "
              << (device == torch::kCUDA ? "cuda" : "cpu") << "\n";
#endif

    // ---- Load dataset ----
    std::cout << "[INFO] Loading dataset from: " << cfg.data_dir << "\n";
    auto samples = load_dataset(cfg.data_dir, cfg.radius, cfg.clip_len);
    std::cout << "[INFO] Dataset: " << samples.size() << " samples\n";

    // Log class balance
    {
        int64_t total_pos = 0, total_vox = 0;
        for (const auto& s : samples) {
            total_pos += s.labels.sum().item<int64_t>();
            total_vox += s.labels.numel();
        }
        double pct = total_vox > 0
            ? 100.0 * total_pos / total_vox : 0.0;
        std::cout << "[INFO] Class balance: " << total_pos << " positive / "
                  << (total_vox - total_pos) << " negative voxels ("
                  << std::fixed << std::setprecision(2) << pct
                  << "% positive)\n";
    }

    // ---- Create model ----
    std::cout << "[INFO] Creating 4D UNet...\n";
    Simple4DUNet model(
        /*c_in=*/3,
        /*c_mid=*/cfg.c_mid,
        /*dim=*/4);
    model->to(device);

    // Count parameters
    int64_t total_params = 0;
    for (auto& p : model->parameters()) total_params += p.numel();
    std::cout << "[INFO] Model parameters: " << total_params << "\n";

    // ---- Create optimizer ----
    auto optimizer = torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(cfg.lr)
            .weight_decay(cfg.weight_decay));

    // ---- Training loop ----
    std::cout << "\nStarting training...\n";
    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        auto t0 = std::chrono::high_resolution_clock::now();

        std::cout << "\n--- Epoch " << epoch << "/" << cfg.epochs << " ---\n";
        double loss = train_one_epoch(
            model, samples, optimizer, device, backend,
            cfg.pos_weight, cfg.batch_size);

        double prec = evaluate(
            model, samples, device, backend, cfg.batch_size);

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Epoch " << std::setw(2) << std::setfill('0') << epoch
                  << std::setfill(' ')
                  << " | loss=" << std::fixed << std::setprecision(4) << loss
                  << " | pos-precision=" << std::setprecision(4) << prec
                  << " | " << std::setprecision(1) << elapsed << "s\n";

        // Save checkpoint
        auto ckpt_path = (fs::path(cfg.save_dir)
            / ("model_epoch_"
               + std::string(epoch < 10 ? "0" : "")
               + std::to_string(epoch) + ".pt")).string();
        save_model(model, ckpt_path);
    }

    // Save final checkpoint
    auto final_path = (fs::path(cfg.save_dir) / "model_final.pt").string();
    save_model(model, final_path);
    std::cout << "\nSaved final checkpoint to: " << final_path << "\n";

    return 0;
}
