import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import json
import glob
import argparse
from typing import List, Tuple, Dict, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import MinkowskiEngine as ME

# Maximum number of frames removed in block-removal subsampling (r cap)
MAX_REMOVAL_R = 8


# -----------------------------
# Utilities
# -----------------------------

def parse_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def normalize_num_cameras(num_cameras: int, max_cameras: int = 10) -> float:
    """
    Normalize num_cameras to [0, 1] range.
    """
    return float(min(num_cameras, max_cameras)) / float(max_cameras)


def meters_to_grid(position_m: np.ndarray, origin_m: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """
    Convert continuous meters to integer grid indices using floor.
    """
    return np.floor((position_m - origin_m) / voxel_size_m).astype(np.int32)


def _max_r_for_L(L: int, clip_len: int) -> int:
    """Return maximum r >= 1 such that kept length with pattern (r+1) is >= clip_len.
    Uses binary search to avoid scanning many r values for long runs.
    """
    if L < clip_len:
        return 0
    low = 1
    high = L
    best = 0
    while low <= high:
        mid = (low + high) // 2
        kept = (L + mid) // (mid + 1)  # integer count of kept positions
        if kept >= clip_len:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def find_segments_in_dir(frame_dir: str, clip_len: int = 10, subsample: bool = True) -> List[List[str]]:
    """
    Return all contiguous clip_len-frame segments (sliding window) found within a single directory.
    Also optionally generate block-removal subsampled segments (remove r frames then keep 1, repeat).

    Implementation notes:
      - Operates on runs of consecutive numeric indices to avoid cross-run mixing.
      - Uses tuples of frame indices for fast deduping (integers are faster and smaller than long path tuples).
      - Uses binary search to limit r values considered for each run.
    """
    import re
    all_jsons = sorted(glob.glob(os.path.join(frame_dir, "*.json")))
    if len(all_jsons) == 0:
        return []

    idx_to_path: Dict[int, str] = {}
    for p in all_jsons:
        name = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(\d+)", name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in idx_to_path:
            idx_to_path[idx] = p

    matched_indices = sorted(idx_to_path.keys())
    if len(matched_indices) == 0:
        return []

    # Split into runs of consecutive indices (so we only subsample within contiguous ranges)
    runs: List[List[int]] = []
    current_run = [matched_indices[0]]
    for idx in matched_indices[1:]:
        if idx == current_run[-1] + 1:
            current_run.append(idx)
        else:
            runs.append(current_run)
            current_run = [idx]
    runs.append(current_run)

    segments_idx_set = set()  # store tuples of integer indices for dedup
    segments: List[List[str]] = []

    for run in runs:
        L = len(run)

        # contiguous windows over the run (by index)
        for i in range(0, L - clip_len + 1):
            seg_idx = tuple(run[i:i + clip_len])
            if seg_idx not in segments_idx_set:
                segments_idx_set.add(seg_idx)
                segments.append([idx_to_path[n] for n in seg_idx])

        # block-removal subsampling: consider r in [1, r_max]
        if subsample:
            r_max = _max_r_for_L(L, clip_len)
            # Cap r_max to avoid excessive subsampling patterns
            r_max = min(r_max, MAX_REMOVAL_R)
            if r_max <= 0:
                continue
            for r in range(1, r_max + 1):
                positions = list(range(0, L, r + 1))
                kept_len = len(positions)
                if kept_len < clip_len:
                    continue
                # For each sliding window over the kept positions, build the corresponding index tuple
                for i in range(0, kept_len - clip_len + 1):
                    seg_idx = tuple(run[positions[j]] for j in range(i, i + clip_len))
                    if seg_idx not in segments_idx_set:
                        segments_idx_set.add(seg_idx)
                        segments.append([idx_to_path[n] for n in seg_idx])

    return segments


def find_all_segments(root_dir: str, clip_len: int = 10, subsample: bool = True) -> List[List[str]]:
    """
    Recursively search root_dir and its subdirectories for all contiguous clip_len-frame segments,
    including optional block-removal subsampled variants.

    Uses a tqdm progress bar across directories and reports the number of segments found so far.
    """
    all_segments: List[List[str]] = []
    dirs = [d for d, _, _ in os.walk(root_dir)]
    total_dirs = len(dirs)
    total_found = 0
    pbar = tqdm(dirs, desc="Scanning dirs", unit="dir", total=total_dirs)
    for dpath in pbar:
        segs = find_segments_in_dir(dpath, clip_len=clip_len, subsample=subsample)
        if segs:
            all_segments.extend(segs)
            total_found += len(segs)
            pbar.set_postfix(dict(segments=total_found))
    pbar.close()
    return all_segments


def compute_labels_for_points(
    coords_xyz_it: np.ndarray,
    targets_per_t: Dict[int, List[np.ndarray]],
    radius_vox: int,
    clip_len: int
) -> np.ndarray:
    """
    Generate binary labels ONLY at a single reference frame (t_ref),
    producing a spatial sphere rather than a motion-smeared tube.

    Args:
        coords_xyz_it: [N,4] int32 array -> (ix, iy, iz, it)
        targets_per_t: dict mapping timestep -> list of target centers (grid coords)
        radius_vox: labeling radius in voxel units
        clip_len: number of frames in the clip

    Returns:
        labels: [N] float32 binary array
    """

    labels = np.zeros((coords_xyz_it.shape[0],), dtype=np.float32)

    # Reference time = center frame of the clip
    t_ref = clip_len // 2

    # If no target exists at reference frame, return all zeros
    if t_ref not in targets_per_t:
        return labels

    centers = targets_per_t[t_ref]
    r2 = radius_vox * radius_vox

    # Only consider voxels at reference timestep
    for i in range(coords_xyz_it.shape[0]):
        ix, iy, iz, it = coords_xyz_it[i]

        if it != t_ref:
            continue

        # Check voxel against all targets at reference frame
        for center in centers:
            dx = ix - center[0]
            dy = iy - center[1]
            dz = iz - center[2]
            d2 = dx * dx + dy * dy + dz * dz

            if d2 <= r2:
                labels[i] = 1.0
                break  # no need to check other targets

    return labels


def build_clip_arrays_standalone(
    frames: List[dict],
    grid_info: dict,
    radius_vox: int,
    clip_len: int = 10
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build:
      coords_xyz_it: [N,4] int32 (ix, iy, iz, it)
      feats:         [N,3] float32 (intensity_norm, num_cameras_norm, t_norm)
      labels:        [N]   float32 (0/1) positive around target center
    """
    origin_m = np.array(grid_info.get("origin_m", [0.0, 0.0, 0.0]), dtype=np.float32)
    voxel_size_m = float(grid_info.get("voxel_size_m", 1.0))

    coords_list = []
    intens_list = []
    num_cameras_list = []
    tnorm_list = []
    targets_per_t: Dict[int, np.ndarray] = {}

    # Targets per frame - NOW STORES ALL TARGETS
    for it, fr in enumerate(frames):
        tlist = fr.get("targets", [])
        if len(tlist) > 0:
            centers = []
            for tgt in tlist:
                pos_m = np.array(tgt.get("position_m", [0, 0, 0]), dtype=np.float32)
                center_idx = meters_to_grid(pos_m, origin_m, voxel_size_m)
                centers.append(center_idx)
            targets_per_t[it] = centers  # List of all target centers

    # Sparse voxels across frames
    for it, fr in enumerate(frames):
        vox = fr.get("voxels", [])
        if len(vox) == 0:
            continue
        c = np.array([v["coordinates"] for v in vox], dtype=np.int32)  # [M,3]
        tcol = np.full((c.shape[0], 1), it, dtype=np.int32)
        coords_list.append(np.hstack([c, tcol]))

        intens = np.array([v.get("intensity", 0.0) for v in vox], dtype=np.float32)
        intens_list.append(intens)

        num_cams = np.array([normalize_num_cameras(v.get("num_cameras", 1)) for v in vox], dtype=np.float32)
        num_cameras_list.append(num_cams)

        t_norm = np.full((c.shape[0], 1), float(it) / float(max(clip_len - 1, 1)), dtype=np.float32)
        tnorm_list.append(t_norm)

    if len(coords_list) == 0:
        return None

    coords_xyz_it = np.vstack(coords_list)             # [N,4]
    intens = np.concatenate(intens_list, axis=0)       # [N]
    num_cams = np.concatenate(num_cameras_list, axis=0)  # [N]
    t_norm = np.vstack(tnorm_list)                     # [N,1]

    # Normalize intensity across the clip
    if len(intens) > 1:
        mu, sigma = float(np.mean(intens)), float(np.std(intens) + 1e-6)
        intens_norm = (intens - mu) / sigma
    else:
        intens_norm = intens

    feats = np.concatenate([
        intens_norm.reshape(-1, 1).astype(np.float32),
        num_cams.reshape(-1, 1).astype(np.float32),
        t_norm.astype(np.float32)
    ], axis=1)  # [N,3]

    labels = compute_labels_for_points(
        coords_xyz_it,
        targets_per_t,
        radius_vox=radius_vox,
        clip_len=clip_len
    )
    return coords_xyz_it.astype(np.int32), feats.astype(np.float32), labels.astype(np.float32)


def process_segment_worker(args_tuple):
    seg_files, radius_vox, clip_len = args_tuple
    try:
        frames = [parse_json(p) for p in seg_files]
        grid_info = frames[0].get("grid_info", {})
        result = build_clip_arrays_standalone(frames, grid_info, radius_vox, clip_len=clip_len)
        if result is None:
            return None
        coords, feats, labels = result
        return (coords, feats, labels, grid_info)
    except Exception as e:
        print(f"[WARNING] Failed to process segment: {e}")
        return None


# -----------------------------
# Dataset
# -----------------------------

class Sparse4DClipDataset(torch.utils.data.Dataset):
    """
    Each sample is a 10-frame clip merged into one 4D sparse point cloud.
    Features per point: [intensity_norm, num_cameras_norm, t_norm] => 3 dims.
    Labels: binary for objectness at active points, radius-based around target center per frame.

    This dataset also generates additional clips by block-removal subsampling of each contiguous
    run of frames (remove r frames then keep 1, for r=1,2,...), producing extra temporal
    resolutions for training.
    """
    def __init__(self, dir_path: str, radius_vox: int = 2, clip_len: int = 10):
        super().__init__()
        self.dir_path = dir_path
        self.radius_vox = radius_vox
        self.clip_len = clip_len

        print(f"[DEBUG] Searching for {clip_len}-frame segments in: {dir_path}")
        self.clip_segments = find_all_segments(dir_path, clip_len=clip_len, subsample=True)
        print(f"[DEBUG] Found {len(self.clip_segments)} segments (including subsampled)")
        if len(self.clip_segments) == 0:
            raise RuntimeError(f"No contiguous segments found under {dir_path}")

        self._samples = []
        cores = cpu_count()
        print(f"[DEBUG] Building samples from {len(self.clip_segments)} segments using {cores} cores...")

        worker_args = [(seg_files, self.radius_vox, self.clip_len) for seg_files in self.clip_segments]
        with Pool(processes=cores) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_segment_worker, worker_args, chunksize=10),
                               total=len(worker_args), desc="Processing segments", unit="seg"):
                if result is not None:
                    results.append(result)
        self._samples = results

        print(f"[DEBUG] Built {len(self._samples)} samples (skipped {len(self.clip_segments) - len(self._samples)} empty segments)")
        self.grid_info = self._samples[0][3]
        self.voxel_size_m = float(self.grid_info.get("voxel_size_m", 1.0))
        self.origin_m = np.array(self.grid_info.get("origin_m", [0.0, 0.0, 0.0]), dtype=np.float32)

    def _find_all_segments_len(self, root_dir: str, clip_len: int) -> List[List[str]]:
        # Delegate to the generic segment finder that supports subsampled block-removal sequences
        return find_all_segments(root_dir, clip_len=clip_len, subsample=True)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        coords, feats, labels, grid_info = self._samples[idx]
        return coords.copy(), feats.copy(), labels.copy(), grid_info


def sparse_collate_fn(batch):
    """
    Batch is a list of (coords_xyz_it, feats, labels, grid_info).
    Produces MinkowskiEngine-friendly coordinates/features via ME.utils.sparse_collate,
    and concatenates labels in the same per-sample order.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None, None, []

    coords_list = []
    feats_list = []
    labels_list = []
    infos = []

    for (coords_xyz_it, feats, labels, info) in batch:
        if coords_xyz_it.ndim != 2 or coords_xyz_it.shape[1] != 4:
            raise ValueError(f"Expected coords_xyz_it [N,4], got {coords_xyz_it.shape}")
        coords_list.append(coords_xyz_it.astype(np.int32))
        feats_list.append(feats.astype(np.float32))
        labels_list.append(torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1))
        infos.append(info)

    # ME will prepend batch index automatically, yielding coords [sumN, 1+4] = [sumN, 5]
    coords_b, feats_b = ME.utils.sparse_collate(coords_list, feats_list)
    labels_b = torch.cat(labels_list, dim=0)  # aligns with concatenation order used by sparse_collate

    return coords_b, feats_b, labels_b, infos


# -----------------------------
# Model blocks
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, dim=4, ks=3, stride=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=ks,
                stride=stride,
                dilation=dilation,
                dimension=dim,
            ),
            ME.MinkowskiBatchNorm(c_out),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x: ME.SparseTensor):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, c, dim=4):
        super().__init__()
        self.conv1 = ConvBNReLU(c, c, dim=dim)
        self.conv2 = ME.MinkowskiConvolution(c, c, kernel_size=3, stride=1, dimension=dim)
        self.bn2 = ME.MinkowskiBatchNorm(c)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor):
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = out + x
        return self.relu(out)


class TemporalTransformer(nn.Module):
    """
    Transformer over per-frame tokens (tiny T, typically 10).
    """
    def __init__(self, d_model: int, nhead: int = 4, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, tokens_bt: torch.Tensor) -> torch.Tensor:
        # tokens_bt: [B, T, C]
        return self.encoder(tokens_bt)


# -----------------------------
# Hybrid CNN + Transformer UNet
# -----------------------------

class Simple4DUNetWithTemporalTransformer(nn.Module):
    def __init__(
        self,
        c_in: int = 3,
        c_mid: int = 24,
        dim: int = 4,
        clip_len: int = 10,
        tx_layers: int = 2,
        tx_heads: int = 4,
        tx_dropout: float = 0.0,
        inject_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.clip_len = clip_len
        self.inject_scale = inject_scale

        # Encoder
        self.stem = ConvBNReLU(c_in, c_mid, dim=dim)
        self.enc1 = ResidualBlock(c_mid, dim=dim)
        self.down1 = ME.MinkowskiConvolution(c_mid, c_mid * 2, kernel_size=2, stride=2, dimension=dim)
        self.enc2 = ResidualBlock(c_mid * 2, dim=dim)
        self.down2 = ME.MinkowskiConvolution(c_mid * 2, c_mid * 4, kernel_size=2, stride=2, dimension=dim)
        self.bottleneck = ResidualBlock(c_mid * 4, dim=dim)

        # Temporal transformer on bottleneck channels
        Cb = c_mid * 4
        self.pos_embed = nn.Parameter(torch.randn(clip_len, Cb) * 0.02)
        self.temporal_tx = TemporalTransformer(d_model=Cb, nhead=tx_heads, layers=tx_layers, dropout=tx_dropout)

        # Optional projection/gating (keeps things stable early in training)
        self.tx_to_feat = nn.Sequential(
            nn.LayerNorm(Cb),
            nn.Linear(Cb, Cb),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))  # starts near 0, learns to use transformer

        # Decoder
        self.up1 = ME.MinkowskiConvolutionTranspose(c_mid * 4, c_mid * 2, kernel_size=2, stride=2, dimension=dim)
        self.dec1 = ResidualBlock(c_mid * 2, dim=dim)
        self.up2 = ME.MinkowskiConvolutionTranspose(c_mid * 2, c_mid, kernel_size=2, stride=2, dimension=dim)
        self.dec2 = ResidualBlock(c_mid, dim=dim)

        # Head
        self.head = ME.MinkowskiConvolution(c_mid, 1, kernel_size=1, stride=1, dimension=dim)

    @staticmethod
    def _mean_token(feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # feats: [N,C], mask: [N] bool
        if mask.any():
            return feats[mask].mean(dim=0)
        return torch.zeros((feats.shape[1],), device=feats.device, dtype=feats.dtype)

    def _apply_temporal_transformer(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        x is bottleneck sparse tensor with coords [N, 1+4] = [N,5] (b, x, y, z, t).
        We build per-frame tokens per batch, run transformer over time, then broadcast
        back into the sparse features via residual injection.
        """
        coords = x.C  # int tensor [N,5]
        feats = x.F   # float tensor [N,C]
        device = feats.device
        dtype = feats.dtype

        bcol = coords[:, 0].long()
        tcol = coords[:, -1].long()
        B = int(bcol.max().item()) + 1 if coords.shape[0] > 0 else 1
        T = self.clip_len
        C = feats.shape[1]

        # Build tokens [B,T,C] via mean pooling per frame
        tokens = torch.zeros((B, T, C), device=device, dtype=dtype)
        for b in range(B):
            bm = (bcol == b)
            if not bm.any():
                continue
            for t in range(T):
                m = bm & (tcol == t)
                if m.any():
                    tokens[b, t] = feats[m].mean(dim=0)

        # Add positional embedding
        tokens = tokens + self.pos_embed.to(device=device, dtype=dtype).unsqueeze(0)  # [B,T,C]

        # Transformer
        tokens2 = self.temporal_tx(tokens)  # [B,T,C]
        tokens2 = self.tx_to_feat(tokens2)

        # Broadcast back: enhanced_feats[i] += gate * tokens2[b,t]
        enhanced = feats.clone()
        g = torch.tanh(self.gate) * self.inject_scale
        if coords.shape[0] > 0:
            # vectorized gather
            bt = tokens2[bcol.clamp(min=0, max=B-1), tcol.clamp(min=0, max=T-1)]  # [N,C]
            enhanced = enhanced + g * bt

        return ME.SparseTensor(
            features=enhanced,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

    def forward(self, x: ME.SparseTensor):
        # Encoder
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        # Temporal transformer injection at bottleneck
        x3 = self._apply_temporal_transformer(x3)

        # Decoder
        u1 = self.up1(x3)
        d1 = self.dec1(u1)
        u2 = self.up2(d1)
        d2 = self.dec2(u2)

        logits = self.head(d2)  # SparseTensor with .F [N,1]
        return logits


# -----------------------------
# Training
# -----------------------------

def train_one_epoch(model, loader, optimizer, device, pos_weight: float, show_progress: bool = True):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    iterator = tqdm(loader, desc="Batches", unit="batch", leave=False) if show_progress else loader

    for coords, feats, labels, _ in iterator:
        if coords is None:
            continue

        coords = coords.to(device)
        feats = feats.to(device)
        labels = labels.to(device)  # [N,1]

        stensor = ME.SparseTensor(features=feats, coordinates=coords, device=device)
        logits = model(stensor).F  # [N,1]

        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        if show_progress:
            n = max(int(iterator.n), 1) if hasattr(iterator, "n") else 1
            iterator.set_postfix(loss=total_loss / n)

    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    model.eval()
    total_pos, correct_pos = 0, 0
    with torch.no_grad():
        for coords, feats, labels, _ in loader:
            if coords is None:
                continue
            coords = coords.to(device)
            feats = feats.to(device)
            labels = labels.to(device)

            stensor = ME.SparseTensor(features=feats, coordinates=coords, device=device)
            logits = model(stensor).F
            probs = torch.sigmoid(logits)
            
            preds = (probs >= 0.5).float()

            total_pos += int(labels.sum().item())
            correct_pos += int((preds * labels).sum().item())

    return (correct_pos / total_pos) if total_pos > 0 else 0.0


def main():
    ap = argparse.ArgumentParser("Sparse 4D CNN + Temporal Transformer training (MinkowskiEngine)")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--pos_weight", type=float, default=12.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--clip_len", type=int, default=10)

    # Transformer knobs
    ap.add_argument("--tx_layers", type=int, default=2)
    ap.add_argument("--tx_heads", type=int, default=4)
    ap.add_argument("--tx_dropout", type=float, default=0.0)
    ap.add_argument("--tx_inject_scale", type=float, default=1.0)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"[DEBUG] Creating dataset from: {args.data_dir}")
    dataset = Sparse4DClipDataset(args.data_dir, radius_vox=args.radius, clip_len=args.clip_len)
    print(f"[DEBUG] Dataset created with {len(dataset)} samples")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=sparse_collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[DEBUG] Creating hybrid model...")
    model = Simple4DUNetWithTemporalTransformer(
        c_in=3,
        c_mid=32,
        dim=4,
        clip_len=args.clip_len,
        tx_layers=args.tx_layers,
        tx_heads=args.tx_heads,
        tx_dropout=args.tx_dropout,
        inject_scale=args.tx_inject_scale,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Starting training...")
    epoch_iter = tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        loss = train_one_epoch(model, loader, optimizer, device, pos_weight=args.pos_weight, show_progress=True)
        prec = evaluate(model, loader, device)
        tqdm.write(f"Epoch {epoch:02d} | loss={loss:.4f} | pos-precision={prec:.4f} | gate(tanh)={float(torch.tanh(model.gate).item()):+.4f}")

        ckpt_path = os.path.join(args.save_dir, f"model_epoch_{epoch:02d}.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "args": vars(args)}, ckpt_path)

    final_path = os.path.join(args.save_dir, "model_final.pt")
    torch.save({"epoch": args.epochs, "model_state": model.state_dict(), "args": vars(args)}, final_path)
    print(f"Saved final checkpoint to: {final_path}")


if __name__ == "__main__":
    main()
