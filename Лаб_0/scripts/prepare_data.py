from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_PATH = DATA_DIR / "split.json"
MANIFEST_PATH = DATA_DIR / "manifest.json"

SPLIT_SEED = 0
N_CLASSES = 10
EXPECTED_SIZES = {"train": 1077, "val": 360, "test": 360}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_digits(return_X_y=True)
    idx = np.arange(len(y))

    rest, test = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    train, val = train_test_split(
        rest, test_size=0.25, stratify=y[rest], random_state=SPLIT_SEED
    )

    sizes = {"train": len(train), "val": len(val), "test": len(test)}
    assert sizes == EXPECTED_SIZES, f"划分大小不符: {sizes}"

    all_indices = np.concatenate([train, val, test])
    assert len(all_indices) == len(y) == 1797
    assert len(np.unique(all_indices)) == 1797, "存在重叠索引"

    class_counts = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        counts = np.bincount(y[part], minlength=N_CLASSES)
        assert counts.shape[0] == N_CLASSES
        assert np.all(counts > 0), f"{name} 中缺少某些类别: {counts.tolist()}"
        class_counts[name] = counts.tolist()

    split_data = {
        "train": sorted(int(i) for i in train),
        "val": sorted(int(i) for i in val),
        "test": sorted(int(i) for i in test),
    }
    with SPLIT_PATH.open("w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    split_hash = sha256_file(SPLIT_PATH)

    manifest = {
        "dataset": "sklearn.datasets.load_digits",
        "source": {"library": "scikit-learn", "version": sklearn.__version__},
        "split_rule": {
            "split_seed": SPLIT_SEED,
            "test_size": 0.2,
            "val_size_of_rest": 0.25,
            "stratify": True,
        },
        "sizes": sizes,
        "n_samples_total": int(len(y)),
        "n_features": int(X.shape[1]),
        "classes": list(range(N_CLASSES)),
        "class_counts": class_counts,
        "split_file": "data/split.json",
        "split_sha256": split_hash,
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print("数据划分完成")
    print(f"train/val/test = {sizes['train']}/{sizes['val']}/{sizes['test']}")
    print(f"split.json SHA-256 = {split_hash}")
    print(f"已写入: {SPLIT_PATH}")
    print(f"已写入: {MANIFEST_PATH}")

if __name__ == "__main__":
    main()