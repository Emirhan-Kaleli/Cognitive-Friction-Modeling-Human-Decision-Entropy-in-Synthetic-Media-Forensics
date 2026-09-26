"""Uçtan uca bilişsel analiz boru hattını tek komutla çalıştıran koordinatör.

Adımlar (her biri ayrı süreçte, ``python -m src.<modül>`` ile):
    1. Puanlama            : src/scorer.py         (D, H, S_T + parmak izi)
    2. Görsel indirme      : src/download_images.py (--skip-download ile atlanır)
    3. Görselleştirme      : src/visualize.py       (rapor grafikleri)
    4. DINOv2 embedding    : src/extract_embeddings.py (--skip-embed ile atlanır)
    5. Gating model eğitimi: src/train.py

Kullanım::

    python run_pipeline.py
    python run_pipeline.py --skip-download   # görseller diskteyse
    python run_pipeline.py --skip-embed      # embedding'ler hesaplanmışsa

Betik, çalıştırıldığı dizinden bağımsızdır (cwd = proje kökü); her adımın
çıktısı terminale akar; başarısız bir adım boru hattını durdurur.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def run_step(index: int, total: int, module: str, label: str) -> int:
    print(f"\n=== [{index}/{total}] {label} ({module}) ===")
    print("--- başlıyor ---")
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=PROJECT_ROOT,
        env=_child_env(),
    )
    if result.returncode != 0:
        print(f"[HATA] {module} çıkış kodu {result.returncode} ile başarısız oldu; boru hattı durduruldu.")
        return result.returncode
    print(f"--- {label} tamamlandı ---")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="uçtan uca bilişsel analiz boru hattı")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="görsel indirme adımını atla (görseller zaten data/raw/images altında ise)",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="DINOv2 embedding adımını atla (image_embeddings.pt zaten varsa)",
    )
    args = parser.parse_args(argv)

    steps: list[tuple[str, str]] = [
        ("src.scorer", "Puanlama ve metrik çıkarımı"),
        ("src.visualize", "Görselleştirme üretimi"),
        ("src.extract_embeddings", "DINOv2 embedding çıkarımı"),
        ("src.train", "Gating modeli eğitimi"),
    ]

    if not args.skip_download:
        steps.insert(1, ("src.download_images", "Görsel indirme"))

    scored_exists = (PROJECT_ROOT / "data" / "processed" / "scored_posts.jsonl").exists()
    if not args.skip_download and not scored_exists:
        print("[bilgi] scored_posts.jsonl bulunamadı; görsel indirme adımı atlanıyor "
              "(indirme hedefleri puanlama çıktısından türetilir).")
        steps = [s for s in steps if s[0] != "src.download_images"]

    total = len(steps)
    print(f"=== BORU HATTI BAŞLADI: {total} adım ===")
    for index, (module, label) in enumerate(steps, start=1):
        code = run_step(index, total, module, label)
        if code != 0:
            return code

    print("\n=== BORU HATTI TAMAMLANDI ===")
    print("Çıktılar:\n"
          "  data/processed/scored_posts.jsonl\n"
          "  reports/figures/*.png | *.svg\n"
          "  data/processed/image_embeddings.pt\n"
          "  models/cognitive_gating_head.pt\n"
          "  reports/training_metrics.json")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    raise SystemExit(main())