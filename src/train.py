"""Çok görevli bilişsel tahmin modelinin eğitimi.

Akış:
  1. ``data/processed/image_embeddings.pt`` yüklenir.
  2. Seed=42 ile deterministik %80/%20 eğitim-doğrulama ayrımı yapılır.
  3. ``CognitiveGatingModel`` AdamW + CosineAnnealingLR ile eğitilir;
     kayıp = MSE(D) + 0.5 * KLDiv(parmak izi). Erken durdurma uygulanır.
  4. En düşük doğrulama kayıplı ağırlıklar ``models/cognitive_gating_head.pt``
     olarak, geçmiş ve son metrikler ``reports/training_metrics.json``
     olarak yazılır.

Son val metrikleri: D için MAE ve R^2; parmak izi için ortalama kosinüs
benzerliği ve argmax (Top-1) doğruluğu.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.model import CognitiveGatingModel

DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data" / "processed" / "image_embeddings.pt"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "models" / "cognitive_gating_head.pt"
DEFAULT_METRICS_OUT = PROJECT_ROOT / "reports" / "training_metrics.json"

BATCH_SIZE = 32
SEED = 42
VAL_FRACTION = 0.2
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FP_LOSS_WEIGHT = 0.5
EARLY_STOP_PATIENCE = 8
N_FINGERPRINT = 4
_FP_AXES = ("anatomy", "optics", "typography", "render")


def resolve_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def load_payload(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embeddings: torch.Tensor = payload["embeddings"].float()
    target_d: torch.Tensor = payload["targets"]["D"].float()
    target_fp: torch.Tensor = payload["targets"]["fingerprint"].float()
    return embeddings, target_d, target_fp


def split_indices(n: int, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_count = int(round(n * val_fraction))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    return train_idx, val_idx


def make_loaders(
    embeddings: torch.Tensor,
    target_d: torch.Tensor,
    target_fp: torch.Tensor,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    def dataset(idx) -> TensorDataset:
        return TensorDataset(embeddings[idx], target_d[idx], target_fp[idx])

    train_loader = DataLoader(dataset(torch.as_tensor(train_idx)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset(torch.as_tensor(val_idx)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def val_metrics(model: nn.Module, loader: DataLoader, device: torch.device, mse: nn.MSELoss, kld: nn.KLDivLoss) -> dict:
    was_training = model.training
    model.eval()
    pred_d_list, target_d_list, pred_fp_list, target_fp_list = [], [], [], []
    total_d_loss = 0.0
    total_fp_loss = 0.0
    total_units = 0

    with torch.no_grad():
        for x, d, fp in loader:
            x = x.to(device)
            d = d.to(device)
            fp = fp.to(device)
            out = model(x)
            pred_d = out["D"]
            pred_fp = out["fingerprint"]
            batch = int(x.shape[0])
            loss_d = mse(pred_d, d)
            loss_fp = kld(torch.log(pred_fp + 1e-8), fp)
            total_d_loss += float(loss_d) * batch
            total_fp_loss += float(loss_fp) * batch
            total_units += batch
            pred_d_list.append(pred_d.cpu())
            target_d_list.append(d.cpu())
            pred_fp_list.append(pred_fp.cpu())
            target_fp_list.append(fp.cpu())

    pred_d = torch.cat(pred_d_list)
    target_d = torch.cat(target_d_list)
    pred_fp = torch.cat(pred_fp_list)
    target_fp = torch.cat(target_fp_list)

    model.train(mode=was_training)

    mae = float((pred_d - target_d).abs().mean())
    ss_res = float(((target_d - pred_d) ** 2).sum())
    ss_tot = float(((target_d - target_d.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    cos_sim = nn.functional.cosine_similarity(pred_fp, target_fp, dim=-1)
    cosine = float(cos_sim.mean())
    argmax_acc = float((pred_fp.argmax(dim=-1) == target_fp.argmax(dim=-1)).float().mean())

    return {
        "loss_d": total_d_loss / max(total_units, 1),
        "loss_fp": total_fp_loss / max(total_units, 1),
        "loss": total_d_loss / max(total_units, 1) + FP_LOSS_WEIGHT * total_fp_loss / max(total_units, 1),
        "mae": mae,
        "r2": r2,
        "cosine": cosine,
        "argmax": argmax_acc,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mse: nn.MSELoss,
    kld: nn.KLDivLoss,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    total_d = 0.0
    total_fp = 0.0
    total_units = 0

    for x, d, fp in loader:
        x = x.to(device)
        d = d.to(device)
        fp = fp.to(device)
        out = model(x)
        pred_d = out["D"]
        pred_fp = out["fingerprint"]
        loss_d = mse(pred_d, d)
        loss_fp = kld(torch.log(pred_fp + 1e-8), fp)
        loss = loss_d + FP_LOSS_WEIGHT * loss_fp
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch = int(x.shape[0])
        total_loss += float(loss.detach()) * batch
        total_d += float(loss_d.detach()) * batch
        total_fp += float(loss_fp.detach()) * batch
        total_units += batch

    if training and scheduler is not None:
        scheduler.step()

    return total_loss / max(total_units, 1), total_d / max(total_units, 1), total_fp / max(total_units, 1)


def print_table(rows: list[dict]) -> None:
    cols = ["epoch", "train_loss", "val_loss", "val_mae", "val_r2", "val_cos", "val_acc"]
    header = " | ".join(f"{c:>10}" for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        vals = [
            f"{int(row['epoch']):>10}",
            f"{row['train_loss']:>10.4f}",
            f"{row['val_loss']:>10.4f}",
            f"{row['val_mae']:>10.4f}",
            f"{row['val_r2']:>10.4f}",
            f"{row['val_cos']:>10.4f}",
            f"{row['val_acc']:>10.4f}",
        ]
        print(" | ".join(vals))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="çok görevli bilişsel tahmin modeli eğitimi")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else resolve_device()
    print(f"[ortam] cihaz: {device}")

    embeddings, target_d, target_fp = load_payload(args.embeddings)
    n = int(embeddings.shape[0])
    train_idx, val_idx = split_indices(n, args.seed, args.val_fraction)
    print(f"[veri] {n:,} gönderi | eğitim: {len(train_idx):,} | doğrulama: {len(val_idx):,}")

    train_loader, val_loader = make_loaders(
        embeddings, target_d, target_fp, train_idx, val_idx, args.batch_size
    )

    model = CognitiveGatingModel(embedding_dim=int(embeddings.shape[1])).to(device)
    mse = nn.MSELoss()
    kld = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    epochs_since_improvement = 0
    history: list[dict] = []

    progress = tqdm(range(1, args.max_epochs + 1), desc="epoch", leave=True)
    for epoch in progress:
        train_loss, train_d, train_fp = run_epoch(
            model, train_loader, device, mse, kld, optimizer, scheduler
        )
        val = val_metrics(model, val_loader, device, mse, kld)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_d": train_d,
            "train_loss_fp": train_fp,
            "val_loss": val["loss"],
            "val_loss_d": val["loss_d"],
            "val_loss_fp": val["loss_fp"],
            "val_mae": val["mae"],
            "val_r2": val["r2"],
            "val_cos": val["cosine"],
            "val_acc": val["argmax"],
        }
        history.append(row)
        progress.set_postfix(val=val["loss"], d_mae=val["mae"], acc=val["argmax"])

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= EARLY_STOP_PATIENCE:
                print(f"[erken-durdurma] {EARLY_STOP_PATIENCE} epoch iyileşme yok; epoch {epoch} sonlandı.")
                break

    if best_state is None:
        raise SystemExit("Eğitim sırasında geçerli doğrulama kaybı üretilemedi.")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.model_out)

    model.load_state_dict(best_state)
    final = val_metrics(model, val_loader, device, mse, kld)

    strict = history[:best_epoch]
    print("\nEğitim geçmişi (epoch bazında):")
    print_table(strict)

    print("\nEn iyi epoch özeti:")
    best_row = history[best_epoch - 1]
    print(f"  epoch          : {best_epoch}")
    print(f"  val_loss       : {best_row['val_loss']:.4f}")
    print(f"  val MAE (D)    : {final['mae']:.4f}")
    print(f"  val R^2 (D)    : {final['r2']:.4f}")
    print(f"  val kosinüs    : {final['cosine']:.4f}")
    print(f"  val argmax aşd : {final['argmax']:.4f}")

    metrics_payload = {
        "config": {
            "embedding_dim": int(embeddings.shape[1]),
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "fp_loss_weight": FP_LOSS_WEIGHT,
            "loss_d": "MSE",
            "loss_fp": "KLDiv(batchmean)",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "device": str(device),
        },
        "data": {"n": n, "n_train": int(len(train_idx)), "n_val": int(len(val_idx))},
        "history": history,
        "best": {"epoch": best_epoch, "val_loss": float(best_val_loss)},
        "final_val": {
            "mae_d": float(final["mae"]),
            "r2_d": float(final["r2"]),
            "cosine_fp": float(final["cosine"]),
            "argmax_acc_fp": float(final["argmax"]),
        },
        "model_path": str(args.model_out),
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[kayıt] {args.model_out}")
    print(f"[kayıt] {args.metrics_out}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    raise SystemExit(main())