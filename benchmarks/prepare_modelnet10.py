"""Prepare deterministic, area-weighted ModelNet10 points with original split membership.

Python standard library only. Data is downloaded only with --download; model files are
read directly from the ZIP, never extracted. CAD data stays outside the Git repository.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import urllib.request
import zipfile

URL = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
CLASSES = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
MAGIC = b"MK4PCD1\0"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    partial = path.with_suffix(".download")
    with urllib.request.urlopen(URL, timeout=60) as source, partial.open("wb") as out:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            out.write(chunk)
    # Verify archive integrity before moving to its final name.
    with zipfile.ZipFile(partial) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt archive member: {bad}")
    partial.replace(path)


def parse_off(data):
    rows = [line.split("#", 1)[0].strip() for line in data.decode("utf-8-sig").splitlines()]
    rows = [line for line in rows if line]
    if not rows or not rows[0].startswith("OFF"):
        raise ValueError("Expected OFF header")
    remainder = rows[0][3:].strip()
    rows = ([remainder] if remainder else []) + rows[1:]
    if not rows or len(rows[0].split()) != 3:
        raise ValueError("Expected OFF vertex/face/edge counts")
    nv, nf, _ = map(int, rows[0].split())
    if nv < 3 or nf < 1 or len(rows) < 1 + nv + nf:
        raise ValueError("Empty or truncated OFF mesh")
    vertices = []
    for row in rows[1:1 + nv]:
        values = list(map(float, row.split()[:3]))
        if len(values) != 3 or not all(map(math.isfinite, values)):
            raise ValueError("Nonfinite or malformed vertex")
        vertices.append(values)
    triangles, cumulative = [], []
    total = 0.0
    for row in rows[1 + nv:1 + nv + nf]:
        values = row.split()
        n = int(values[0])
        if n < 3 or len(values) < n + 1:
            raise ValueError("Malformed face")
        indices = list(map(int, values[1:n + 1]))
        if any(i < 0 or i >= nv for i in indices):
            raise ValueError("Face index out of range")
        # Fan triangulation matches the usual OFF mesh preprocessing convention.
        for j in range(1, n - 1):
            a, b, c = (vertices[i] for i in (indices[0], indices[j], indices[j + 1]))
            u, v = [b[k] - a[k] for k in range(3)], [c[k] - a[k] for k in range(3)]
            cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
            area = 0.5 * math.sqrt(sum(x*x for x in cross))
            if not math.isfinite(area):
                raise ValueError("Nonfinite face area")
            if area > 0:
                triangles.append((a, b, c))
                total += area
                cumulative.append(total)
    if not triangles or not math.isfinite(total):
        raise ValueError("Mesh has no finite positive-area faces")
    lo = [min(p[k] for p in vertices) for k in range(3)]
    hi = [max(p[k] for p in vertices) for k in range(3)]
    center = [(a + b) * 0.5 for a, b in zip(lo, hi)]
    extent = max(b - a for a, b in zip(lo, hi))
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("Invalid bounding box")
    return triangles, cumulative, center, 1.9 / extent


def sample_points(data, count, seed):
    triangles, cumulative, center, scale = parse_off(data)
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        index = bisect.bisect_right(cumulative, rng.random() * cumulative[-1])
        a, b, c = triangles[min(index, len(triangles) - 1)]
        root, v = math.sqrt(rng.random()), rng.random()
        weights = (1 - root, root * (1 - v), root * v)
        result.append([(sum(w * p[k] for w, p in zip(weights, (a, b, c))) - center[k]) * scale for k in range(3)])
    return result


def stable_seed(seed, text):
    return int.from_bytes(hashlib.sha256(f"{seed}:{text}".encode()).digest()[:8], "little")


def fnv1a64(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def prepare(archive_path, output, train_per_class, test_per_class, points, seed):
    if min(train_per_class, test_per_class, points) <= 0:
        raise ValueError("Sample/point counts must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be a new or empty directory; preserve existing experiments")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(format="minkowski4cpp-points-v1", dataset="ModelNet10", source_url=URL,
                    archive_sha256=sha256(archive_path), class_names=CLASSES,
                    points_per_sample=points, preprocessing_seed=seed,
                    sampling="triangle-area weighted; fan triangulation; no augmentation",
                    normalization="mesh bounding-box center; longest side 1.9", splits={})
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for split, per_class in (("train", train_per_class), ("test", test_per_class)):
            selected = []
            for label, cls in enumerate(CLASSES):
                prefix = f"ModelNet10/{cls}/{split}/"
                candidates = sorted(n for n in names if n.startswith(prefix) and n.endswith(".off"))
                if len(candidates) < per_class:
                    raise ValueError(f"Insufficient {split} examples for {cls}")
                random.Random(stable_seed(seed, prefix)).shuffle(candidates)
                selected.append([(label, n) for n in candidates[:per_class]])
            # Interleave classes, matching the synthetic harness's sample order.
            rows = [selected[c][i] for i in range(per_class) for c in range(len(CLASSES))]
            blob = bytearray(MAGIC + struct.pack("<II", len(rows), points))
            ids, source_hashes = [], []
            for label, name in rows:
                source = archive.read(name)
                cloud = sample_points(source, points, stable_seed(seed, name))
                blob.extend(struct.pack("<I", label))
                blob.extend(struct.pack(f"<{points * 3}d", *(x for point in cloud for x in point)))
                ids.append(name)
                source_hashes.append(hashlib.sha256(source).hexdigest())
            filename = f"{split}.bin"
            (output / filename).write_bytes(blob)
            manifest["splits"][split] = dict(file=filename, samples=len(rows), per_class=per_class,
                sha256=hashlib.sha256(blob).hexdigest(), fnv1a64=fnv1a64(blob),
                ids=ids, source_sha256=source_hashes)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--test-per-class", type=int, default=20)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.download:
        download(args.archive)
    if args.download_only:
        print(json.dumps(dict(archive=str(args.archive), sha256=sha256(args.archive), url=URL)))
    else:
        if args.output is None:
            parser.error("--output is required unless --download-only")
        result = prepare(args.archive, args.output, args.train_per_class, args.test_per_class, args.points, args.seed)
        print(json.dumps(dict(output=str(args.output), archive_sha256=result["archive_sha256"],
                              samples={k: v["samples"] for k, v in result["splits"].items()})))
