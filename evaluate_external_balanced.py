from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


WORKSPACE_ROOT = Path(r"C:\Users\Chan\Desktop\a")
DEFAULT_SOURCE_ROOT = Path(r"C:\Users\Chan\Desktop\AI_Human") / "\uac80\uc99d\uc9c0\ud45c"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "external_eval_outputs"
DEFAULT_TRAIN_DIR = Path(r"C:\Users\Chan\Desktop\AI_Human_TeamProject_first\transfer_learning")
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "backend" / "checkpoints" / "best_beats_fine.pt"
DEFAULT_BASE_CHECKPOINT = WORKSPACE_ROOT / "backend" / "checkpoints" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
DEFAULT_BEATS_DIR = WORKSPACE_ROOT / "backend" / "beats"

SAMPLE_RATE = 16000
CLIP_SECONDS = 5
TARGET_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

if str(DEFAULT_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(DEFAULT_TRAIN_DIR))

import train_beats_fine as train_fine  # noqa: E402

LABELS = train_fine.PROJECT_TASK_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BEATs on a balanced external folder test set.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-class", type=int, default=73, help="0 means use the smallest available class count.")
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--beats-dir", type=Path, default=DEFAULT_BEATS_DIR)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only sample and cache audio; skip model evaluation.")
    return parser.parse_args()


def normalized_group_key(path: Path) -> str:
    stem = path.stem
    patterns = [
        r"-(N|S)$",
        r"_(SD|SN|NS)$",
        r"_1$",
        r"_label$",
    ]
    for pattern in patterns:
        new_stem = re.sub(pattern, "", stem, flags=re.IGNORECASE)
        if new_stem != stem:
            return new_stem
    return stem


def collect_audio_groups(source_root: Path) -> dict[str, dict[str, list[Path]]]:
    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")

    groups_by_label: dict[str, dict[str, list[Path]]] = {}
    for label in LABELS:
        label_dir = source_root / label
        if not label_dir.exists():
            raise FileNotFoundError(f"class folder not found: {label_dir}")

        groups: dict[str, list[Path]] = defaultdict(list)
        for path in sorted(label_dir.rglob("*.wav")):
            groups[normalized_group_key(path)].append(path)

        if not groups:
            raise RuntimeError(f"no wav files found in class folder: {label_dir}")
        groups_by_label[label] = groups

    return groups_by_label


def select_balanced_items(
    groups_by_label: dict[str, dict[str, list[Path]]],
    per_class: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    available = {label: len(groups) for label, groups in groups_by_label.items()}
    target = min(available.values()) if per_class <= 0 else min(per_class, min(available.values()))

    if target <= 0:
        raise RuntimeError("balanced sample target is zero.")
    if per_class > min(available.values()):
        print(f"[WARN] requested per-class={per_class}, but smallest class has {min(available.values())}; using {target}.")

    selected: list[dict] = []
    for label in LABELS:
        group_keys = list(groups_by_label[label].keys())
        rng.shuffle(group_keys)
        for group_key in group_keys[:target]:
            variants = list(groups_by_label[label][group_key])
            chosen = rng.choice(variants)
            selected.append(
                {
                    "task_label": label,
                    "group_key": group_key,
                    "source_path": str(chosen),
                    "variant_count": len(variants),
                }
            )

    rng.shuffle(selected)
    return selected


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)

    from scipy.signal import resample_poly

    divisor = math.gcd(int(sr), SAMPLE_RATE)
    up = SAMPLE_RATE // divisor
    down = int(sr) // divisor
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def read_first_clip(path: Path) -> np.ndarray:
    info = sf.info(str(path))
    frames = min(info.frames, int(math.ceil(info.samplerate * CLIP_SECONDS)))
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True, frames=frames)
    audio = audio.mean(axis=1)
    audio = resample_to_16k(audio, sr)

    if audio.size >= TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    else:
        padded = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        padded[: audio.size] = audio
        audio = padded

    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def cache_items(items: list[dict], output_dir: Path, rebuild_cache: bool) -> None:
    cache_root = output_dir / "cache_16k_mono_5s"
    for index, item in enumerate(items, start=1):
        source_path = Path(item["source_path"])
        cache_path = cache_root / item["task_label"] / source_path.name
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if rebuild_cache or not cache_path.exists():
            audio = read_first_clip(source_path)
            sf.write(str(cache_path), audio, SAMPLE_RATE, subtype="PCM_16")

        item["cache_path"] = str(cache_path)

        if index % 50 == 0 or index == len(items):
            print(f"[cache] {index}/{len(items)}", flush=True)


def write_manifest(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_label", "group_key", "variant_count", "source_path", "cache_path"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({key: item.get(key, "") for key in fieldnames})


class ExternalEvalDataset(Dataset):
    def __init__(self, items: list[dict], label_to_idx: dict[str, int]) -> None:
        self.items = items
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        audio = train_fine.read_cached_wav(item["cache_path"])
        target = self.label_to_idx[item["task_label"]]
        return torch.from_numpy(audio), torch.tensor(target, dtype=torch.long), torch.tensor(index, dtype=torch.long)


@torch.no_grad()
def evaluate_external(model: nn.Module, items: list[dict], args: argparse.Namespace) -> tuple[dict, list[dict]]:
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    dataset = ExternalEvalDataset(items, label_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = select_device(args.device)
    model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    total_loss = 0.0
    seen = 0
    predictions: list[dict] = []
    started = time.time()

    for step, (audio, target, indices) in enumerate(loader, start=1):
        audio = audio.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(audio)
        loss = loss_fn(logits, target)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)

        batch_size = int(target.numel())
        total_loss += float(loss.item()) * batch_size
        seen += batch_size

        true_np = target.cpu().numpy()
        pred_np = pred.cpu().numpy()
        conf_np = confidence.cpu().numpy()
        index_np = indices.cpu().numpy()
        for true_idx, pred_idx, conf, item_idx in zip(true_np, pred_np, conf_np, index_np):
            confusion[int(true_idx), int(pred_idx)] += 1
            item = items[int(item_idx)]
            predictions.append(
                {
                    "task_label": item["task_label"],
                    "pred_label": LABELS[int(pred_idx)],
                    "confidence": f"{float(conf):.6f}",
                    "correct": str(int(true_idx) == int(pred_idx)),
                    "source_path": item["source_path"],
                    "cache_path": item["cache_path"],
                }
            )

        if step % 10 == 0 or step == len(loader):
            print(f"[eval] {step}/{len(loader)}")

    acc, macro_f1, weighted_f1, per_class = train_fine.confusion_to_metrics(confusion, LABELS)
    elapsed = time.time() - started
    metrics = {
        "loss": total_loss / max(seen, 1),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "elapsed_sec": elapsed,
        "items": seen,
    }
    return metrics, predictions


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def load_model(args: argparse.Namespace) -> nn.Module:
    model_args = argparse.Namespace(
        beats_dir=args.beats_dir,
        base_checkpoint=args.base_checkpoint,
        resume_checkpoint=args.checkpoint,
    )
    model = train_fine.load_beats_model(model_args, num_classes=len(LABELS))
    train_fine.load_full_checkpoint_if_requested(model, model_args)
    return model


def write_predictions(path: Path, predictions: list[dict]) -> None:
    fieldnames = ["task_label", "pred_label", "confidence", "correct", "source_path", "cache_path"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def write_per_class_csv(path: Path, metrics: dict) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1", "support"])
        for label in LABELS:
            item = metrics["per_class"][label]
            writer.writerow([label, item["precision"], item["recall"], item["f1"], item["support"]])


def write_confusion_csv(path: Path, metrics: dict) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, row in zip(LABELS, metrics["confusion_matrix"]):
            writer.writerow([label, *row])


def write_png_plots(output_dir: Path, metrics: dict, items: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib import failed; PNG plots skipped: {exc}")
        return

    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, np.maximum(row_sums, 1), out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("External Balanced Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            count = int(matrix[row, col])
            value = normalized[row, col]
            color = "white" if value > 0.55 else "black"
            ax.text(col, row, f"{count}\n{value:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "external_balanced_confusion_matrix.png", dpi=160)
    plt.close(fig)

    f1_values = [metrics["per_class"][label]["f1"] for label in LABELS]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(LABELS, f1_values, color=["#4b5563", "#2563eb", "#16a34a", "#f59e0b", "#dc2626"])
    ax.set_title("External Balanced Class F1")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, f1_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "external_balanced_class_f1.png", dpi=160)
    plt.close(fig)

    counts = Counter(item["task_label"] for item in items)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    count_values = [counts[label] for label in LABELS]
    bars = ax.bar(LABELS, count_values, color="#64748b")
    ax.set_title("External Balanced Sample Counts")
    ax.set_ylabel("Samples")
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, count_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "external_balanced_sample_counts.png", dpi=160)
    plt.close(fig)


def write_report(path: Path, args: argparse.Namespace, items: list[dict], metrics: dict) -> None:
    counts = Counter(item["task_label"] for item in items)
    lines = [
        "# External Balanced Test Report",
        "",
        f"- source_root: `{args.source_root}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- items: {metrics['items']}",
        f"- loss: {metrics['loss']:.4f}",
        f"- accuracy: {metrics['accuracy']:.4f}",
        f"- macro_f1: {metrics['macro_f1']:.4f}",
        f"- weighted_f1: {metrics['weighted_f1']:.4f}",
        f"- elapsed_sec: {metrics['elapsed_sec']:.2f}",
        "",
        "## Counts",
        "",
    ]
    for label in LABELS:
        lines.append(f"- {label}: {counts[label]}")
    lines.extend(["", "## Per Class", "", "| label | precision | recall | f1 | support |", "|---|---:|---:|---:|---:|"])
    for label in LABELS:
        item = metrics["per_class"][label]
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = collect_audio_groups(args.source_root)
    for label in LABELS:
        wav_count = sum(len(paths) for paths in groups[label].values())
        print(f"[source] {label:13s} wav={wav_count:,} unique_base={len(groups[label]):,}")

    items = select_balanced_items(groups, args.per_class, args.seed)
    print(f"[sample] selected={len(items):,} per_class={Counter(item['task_label'] for item in items)}")

    cache_items(items, args.output_dir, args.rebuild_cache)
    manifest_path = args.output_dir / "external_balanced_manifest.csv"
    write_manifest(manifest_path, items)
    print(f"[write] manifest={manifest_path}")

    if args.prepare_only:
        print("[done] prepare-only mode; evaluation skipped.")
        return

    model = load_model(args)
    metrics, predictions = evaluate_external(model, items, args)

    metrics_path = args.output_dir / "external_balanced_metrics.json"
    predictions_path = args.output_dir / "external_balanced_predictions.csv"
    per_class_path = args.output_dir / "external_balanced_per_class_metrics.csv"
    confusion_path = args.output_dir / "external_balanced_confusion_matrix.csv"
    report_path = args.output_dir / "external_balanced_report.md"

    train_fine.save_json(metrics_path, metrics)
    write_predictions(predictions_path, predictions)
    write_per_class_csv(per_class_path, metrics)
    write_confusion_csv(confusion_path, metrics)
    write_png_plots(args.output_dir, metrics, items)
    write_report(report_path, args, items, metrics)

    print(f"[metrics] accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"[write] metrics={metrics_path}")
    print(f"[write] report={report_path}")


if __name__ == "__main__":
    main()
