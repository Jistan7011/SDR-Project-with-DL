from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import sample_metadata
from src.signal.processing import complex64_from_channels


def higher_order_moment(x: np.ndarray, p: int, q: int) -> complex:
    return complex(np.mean((x ** p) * (np.conj(x) ** q)))


def feature_vector(channels: np.ndarray) -> np.ndarray:
    x = complex64_from_channels(channels)
    x = x - np.mean(x)
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-8)
    amp = np.abs(x)
    phase_step = np.angle(x[1:] * np.conj(x[:-1]))
    moments = []
    for p, q in [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2)]:
        m = higher_order_moment(x, p, q)
        moments.extend([m.real, m.imag, abs(m)])
    c40 = higher_order_moment(x, 4, 0) - 3.0 * higher_order_moment(x, 2, 0) ** 2
    c42 = higher_order_moment(x, 4, 2) - abs(higher_order_moment(x, 2, 0)) ** 2 - 2.0 * higher_order_moment(x, 2, 1) ** 2
    stats = [
        np.mean(amp),
        np.std(amp),
        np.mean((amp - np.mean(amp)) ** 3),
        np.mean((amp - np.mean(amp)) ** 4),
        np.mean(phase_step),
        np.std(phase_step),
        np.mean(np.abs(phase_step)),
        np.std(np.abs(phase_step)),
        c40.real,
        c40.imag,
        abs(c40),
        c42.real,
        c42.imag,
        abs(c42),
    ]
    return np.asarray(moments + stats, dtype=np.float32)


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((root / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {root / split}")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for path in files:
        data = np.load(path, allow_pickle=False)
        xs.append(feature_vector(data["iq"].astype(np.float32)))
        meta = sample_metadata(path)
        ys.append(CLASS_NAMES.index(str(meta["modulation"])))
    return np.vstack(xs), np.asarray(ys, dtype=np.int64)


def make_classifier(seed: int):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
        ), "xgboost"
    except Exception:
        return GradientBoostingClassifier(random_state=seed), "sklearn_gradient_boosting"


def run_hos_baseline(config_path: str, data_root: str | None = None, output_dir: str = "../results/hos_xgboost") -> dict[str, object]:
    cfg = load_config(config_path)
    root = Path(data_root or cfg["dataset"]["root"])
    seed = int(cfg["project"]["seed"])
    x_train, y_train = load_split(root, "train")
    x_val, y_val = load_split(root, "val")
    x_test, y_test = load_split(root, "test")
    x_fit = np.vstack([x_train, x_val])
    y_fit = np.concatenate([y_train, y_val])
    clf, backend = make_classifier(seed)
    clf.fit(x_fit, y_fit)
    pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASS_NAMES))))
    result = {
        "backend": backend,
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(y_test, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }
    out = ensure_dir(output_dir)
    write_json(out / "logs" / "hos_baseline_eval.json", result)
    with (out / "hos_baseline_model.pkl").open("wb") as f:
        pickle.dump(clf, f)
    print(f"backend={backend} accuracy={result['accuracy']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="../results/hos_xgboost")
    args = parser.parse_args()
    run_hos_baseline(args.config, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
