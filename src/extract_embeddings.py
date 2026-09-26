"""Puanlanmış gönderilerin görsellerinden frozen DINOv2 ile embedding çıkarır.

Akış:
  1. ``scored_posts.jsonl`` satır satır okunur; D=null ya da parmak izi
     toplamı 0 olan gönderiler hariç tutulur.
  2. Her post_id için ``data/raw/images/`` altında ``{post_id}.jpg/.png``
     aranır; kayıp/bozuk görseller loglanıp atlanır.
  3. Görseller ImageNet normalizasyonuyla ön işlenip 32'lik batch'ler halinde
     dondurulmuş ``dinov2_vits14`` modeline verilir; [CLS] embedding'leri
     birleştirilir.
  4. Sonuç ``data/processed/image_embeddings.pt`` sözlüğüne yazılır.

Çıktı şeması::
    {
        "post_ids": List[str],
        "embeddings": torch.Tensor (N, 384),
        "targets": {
            "D": torch.Tensor (N,),
            "H": torch.Tensor (N,),
            "S_T": torch.Tensor (N,),
            "fingerprint": torch.Tensor (N, 4)   # anatomy, optics, typography, render
        }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.config import PROJECT_ROOT

DEFAULT_SCORED = PROJECT_ROOT / "data" / "processed" / "scored_posts.jsonl"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "image_embeddings.pt"

BATCH_SIZE = 32
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
_FINGERPRINT_ORDER = ("anatomy", "optics", "typography", "render")
_IMAGE_MEAN = [0.485, 0.456, 0.406]
_IMAGE_STD = [0.229, 0.224, 0.225]


def resolve_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGE_MEAN, std=_IMAGE_STD),
        ]
    )


def load_model(device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model.to(device)


def find_image(images_dir: Path, post_id: str) -> Path | None:
    for ext in _IMAGE_EXTS:
        candidate = images_dir / f"{post_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_targets(scored_path: Path) -> list[dict]:
    """Puanlanmış gönderilerden eğitim kümesine girecek hedef satırlarını döner."""
    targets: list[dict] = []
    for line in Path(scored_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("D") is None:
            continue
        fingerprint = record.get("fingerprint") or {}
        fp_vec = [float(fingerprint.get(axis, 0.0)) for axis in _FINGERPRINT_ORDER]
        if sum(fp_vec) <= 0:
            continue
        d = float(record["D"])
        h = record.get("H")
        s_t = record.get("S_T")
        if h is None or s_t is None:
            continue
        targets.append(
            {
                "post_id": record["post_id"],
                "D": d,
                "H": float(h),
                "S_T": float(s_t),
                "fingerprint": fp_vec,
            }
        )
    return targets


def extract_cls(output) -> torch.Tensor:
    """Model çıktısından [CLS] embedding vektörlerini ayıklar (B, 384)."""
    tensors = output if isinstance(output, dict) else {"out": output}
    if "x_norm_clstoken" in tensors:
        emb = tensors["x_norm_clstoken"]
    else:
        emb = output
    if emb.ndim == 3:
        emb = emb[:, 0]
    return emb.float()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DINOv2 görsel embedding çıkarımı")
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device) if args.device else resolve_device()
    print(f"[model] cihaz: {device}")
    model = load_model(device)

    transform = build_transform()
    targets = read_targets(args.scored)
    print(f"[veri] hedef gönderi: {len(targets):,}")

    post_ids: list[str] = []
    embeddings: list[torch.Tensor] = []
    d_list: list[float] = []
    h_list: list[float] = []
    st_list: list[float] = []
    fp_list: list[list[float]] = []

    missing = 0
    corrupt = 0
    batch_tensors: list[torch.Tensor] = []
    batch_targets: list[dict] = []

    def flush_batch() -> None:
        nonlocal batch_tensors, batch_targets
        if not batch_tensors:
            return
        images = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            output = model(images)
        embeddings.append(extract_cls(output).cpu())
        for target in batch_targets:
            post_ids.append(target["post_id"])
            d_list.append(target["D"])
            h_list.append(target["H"])
            st_list.append(target["S_T"])
            fp_list.append(target["fingerprint"])
        batch_tensors = []
        batch_targets = []

    progress = tqdm(targets, desc="embedding", unit="görsel")
    for target in progress:
        image_path = find_image(args.images_dir, target["post_id"])
        if image_path is None:
            missing += 1
            continue
        try:
            with Image.open(image_path) as img:
                tensor = transform(img.convert("RGB"))
        except Exception:
            corrupt += 1
            continue
        batch_tensors.append(tensor)
        batch_targets.append(target)
        if len(batch_tensors) >= args.batch_size:
            flush_batch()
    flush_batch()

    if not post_ids:
        raise SystemExit("İşlenebilir görsel bulunamadı; çıktı üretilmedi.")

    payload = {
        "post_ids": post_ids,
        "embeddings": torch.cat(embeddings, dim=0),
        "targets": {
            "D": torch.tensor(d_list, dtype=torch.float32),
            "H": torch.tensor(h_list, dtype=torch.float32),
            "S_T": torch.tensor(st_list, dtype=torch.float32),
            "fingerprint": torch.tensor(fp_list, dtype=torch.float32),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    n, dim = int(payload["embeddings"].shape[0]), int(payload["embeddings"].shape[1])
    print(f"[kayıt] {args.output}")
    print(f"[özet] işlenen görsel: {n:,} | embedding boyutu: {n} x {dim}")
    print(f"[özet] diskte bulunamayan görsel: {missing:,} | bozuk/okunamayan: {corrupt:,}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    raise SystemExit(main())