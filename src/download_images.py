"""Puanlanmış gönderilerin görsellerini diske indirir.

Hedef gönderiler scored_posts.jsonl'den seçilir (D null değil ve
parmak izi toplamı > 0). Görsel adresleri ham kazıma çıktısındaki
``media_urls`` alanının ilk elemanıdır (gallery gönderilerinde ilk kare).

Çıktı: data/raw/images/{post_id}.jpg
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.config import PROJECT_ROOT, build_client

DEFAULT_SCORED = PROJECT_ROOT / "data" / "processed" / "scored_posts.jsonl"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_IMAGES_DIR = DEFAULT_RAW_DIR / "images"
_RAW_POST_FILES = ("raw_posts.jsonl", "raw_posts_gallery.jsonl")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def target_ids(scored_path: Path) -> set[str]:
    ids: set[str] = set()
    for record in _read_jsonl(scored_path):
        if record.get("D") is None:
            continue
        fingerprint = record.get("fingerprint") or {}
        if sum(float(v) for v in fingerprint.values()) <= 0:
            continue
        ids.add(record["post_id"])
    return ids


def load_media_urls(raw_dir: Path) -> dict[str, str]:
    urls: dict[str, str] = {}
    for name in _RAW_POST_FILES:
        for post in _read_jsonl(raw_dir / name):
            media = post.get("media_urls") or []
            if media:
                urls[post["post_id"]] = html.unescape(media[0])
    return urls


def download_one(
    client, url: str, dest: Path, delay: float
) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "mevcut"
    time.sleep(delay)
    try:
        response = client.get(url)
        response.raise_for_status()
        if not response.content:
            return "bos"
        dest.write_bytes(response.content)
        return "ok"
    except Exception as exc:  # noqa: BLE001 - ağ hataları tekil atlanır
        return f"hata:{type(exc).__name__}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="post görsellerini indirir")
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--delay", type=float, default=0.15, help="istekler arası bekleme")
    args = parser.parse_args(argv)

    images_dir = args.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    ids = target_ids(args.scored)
    urls = load_media_urls(args.raw_dir)
    missing_url = sorted(ids - set(urls))
    print(f"[hedef] {len(ids):,} gönderi | URL bilgisi olmayan: {len(missing_url)}")
    for pid in missing_url[:10]:
        print(f"[eksik-url] {pid}")

    counters = {"ok": 0, "mevcut": 0, "bos": 0, "atlanan": 0}
    with build_client(timeout=30.0) as client:
        for pid in sorted(ids):
            url = urls.get(pid)
            if not url:
                counters["atlanan"] += 1
                continue
            dest = images_dir / f"{pid}.jpg"
            status = download_one(client, url, dest, args.delay)
            if status == "ok" or status == "mevcut":
                if status == "ok":
                    counters["ok"] += 1
                else:
                    counters["mevcut"] += 1
            elif status == "bos":
                counters["bos"] += 1
            else:
                print(f"[hata] {pid}: {status}")
                counters["atlanan"] += 1
            if (counters["ok"] + counters["atlanan"]) % 100 == 0 and (counters["ok"] + counters["atlanan"]):
                tqdm.write(f"... {counters['ok'] + counters['mevcut']} tamam")

    total = sum(counters.values())
    print(f"[özet] indirilen: {counters['ok']} | mevcut: {counters['mevcut']} | "
          f"boş: {counters['bos']} | atlanan: {counters['atlanan']} | toplam: {total}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    raise SystemExit(main())