"""Dışarıdan verilen tek bir görsel için bilişsel değerlendirme CLI'ı.

Akış:
  1. PIL ile görsel yüklenir, RGB'ye çevrilir.
  2. ImageNet normalizasyonlu dönüşümle (Resize 224x224) ön işlenir.
  3. Dondurulmuş ``dinov2_vits14`` omurgasıyla [CLS] embedding'i (384-d)
     çıkarılır.
  4. ``CognitiveGatingModel`` ``--weights`` ile yüklenir, eval moduna alınır.
  5. Embedding beslenir -> ``fingerprint`` (4-d) ve ``D`` skoru alınır.
  6. Analitik Shannon Entropisi ve Turing Endeksi hesaplanır:
        H  = -[D*log2(D+1e-9) + (1-D)*log2(1-D+1e-9)]
        S_T = D * (1 + 0.5 * H)

Ayrıca bir görselin D/H/S_T değerlerini doğrulamak için ``--check`` yoktur;
rapor doğrudan terminale basılır.

Kullanım::

    python -m src.infer --image sample.jpg
    python -m src.infer --image sample.jpg --weights models/cognitive_gating_head.pt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from PIL import Image

from src.config import PROJECT_ROOT
from src.extract_embeddings import build_transform, extract_cls, load_model, resolve_device
from src.model import CognitiveGatingModel

DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "cognitive_gating_head.pt"
FINGERPRINT_ORDER = ("anatomy", "optics", "typography", "render")
FINGERPRINT_LABELS = {
    "anatomy": "Anatomi & Biyoloji",
    "optics": "Optik & Fizik",
    "typography": "Tipografi & Yazı",
    "render": "Render & Doku",
}
SEP = "=" * 50
HALF = "-" * 50
MAX_BAR_CHARS = 20


def shannon_entropy(d: float) -> float:
    return -(
        d * math.log2(d + 1e-9) + (1 - d) * math.log2(1 - d + 1e-9)
    )


def category_d(d: float) -> str:
    if d <= 0.33:
        return "Düşük İnandırıcılık / Deşifre"
    if d <= 0.66:
        return "Orta İnandırıcılık / Belirsiz"
    return "Yüksek İnandırıcılık / Gerçek İzlenimi"


def category_h(h: float) -> str:
    if h <= 0.33:
        return "Düşük Belirsizlik / Net Sinyal"
    if h <= 0.66:
        return "Orta Belirsizlik / Kararsız"
    return "Yüksek Kafa Karışıklığı / Araf"


def bar_for(pct: float) -> str:
    count = max(1, round(pct / (100.0 / MAX_BAR_CHARS)))
    count = min(count, MAX_BAR_CHARS)
    return "■" * count


def print_report(image_name: str, d: float, h: float, st: float, fp: torch.Tensor) -> None:
    print(SEP)
    print("COGNITIVE FRICTION INFERENCE REPORT".center(50))
    print(SEP)
    print(f"Görsel: {image_name}")
    print(f"{'Tahmini Aldatma Skoru (D)':<29}: {d:.3f} [{category_d(d)}]")
    print(f"{'Karar Entropisi (H)':<29}: {h:.3f} [{category_h(h)}]")
    print(f"{'Turing Endeksi (S_T)':<29}: {st:.3f}")
    print(HALF)
    print("Tahmini Kusur Parmak İzi (Anomaly Fingerprint):")
    for index, axis in enumerate(FINGERPRINT_ORDER):
        pct = float(fp[index]) * 100.0
        print(
            f"  - {FINGERPRINT_LABELS[axis]:<20}: %{pct:>5.1f} [{bar_for(pct)}]"
        )
    dominant = FINGERPRINT_ORDER[int(torch.argmax(fp))]
    print(f"{'Dominant Anomali':<29}: {FINGERPRINT_LABELS[dominant]}")
    print(SEP)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="tek görsel için bilişsel değerlendirme raporu"
    )
    parser.add_argument("--image", type=Path, required=True, help="analiz edilecek görsel dosyası")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="eğitilmiş CognitiveGatingModel ağırlıkları",
    )
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"[hata] görsel bulunamadı: {args.image}", file=sys.stderr)
        return 1
    if not args.weights.is_file():
        print(f"[hata] ağırlık dosyası bulunamadı: {args.weights}", file=sys.stderr)
        return 1

    device = resolve_device()
    transform = build_transform()

    with Image.open(args.image) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)

    backbone = load_model(device)
    with torch.inference_mode():
        embedding = extract_cls(backbone(tensor))

    model = CognitiveGatingModel(embedding_dim=int(embedding.shape[1])).to(device)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.inference_mode():
        output = model(embedding)

    fp = output["fingerprint"][0].cpu()
    d = float(output["D"][0].item())
    h = shannon_entropy(d)
    st = d * (1.0 + 0.5 * h)

    print_report(args.image.name, d, h, st, fp)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    raise SystemExit(main())