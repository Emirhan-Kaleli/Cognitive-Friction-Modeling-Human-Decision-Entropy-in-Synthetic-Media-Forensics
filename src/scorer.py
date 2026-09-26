"""Hiyerarşik bilişsel metrikler: D, H, S_T ve anomali parmak izi.

Girdiler:
    - data/processed/unsupervised_clusters.json  (kümeler, post_cluster_map,
      comment_cluster_map)
    - data/raw/*.jsonl                            (ham yorum gövdeleri ve oyları)

Çıktı:
    - data/processed/scored_posts.jsonl (Pydantic ile doğrulanmış)

Kural notları:
    - Yön tayini kural bazlıdır: önce açık kalıplar (``not ai`` -> gerçek,
      ``not real`` -> yapay), sonra anahtar kelime sayımı; eşit/eksik durum
      nötr kabul edilir.
    - Anomali kümesine (4 ana eksen) düşen ve metinsel yönü olmayan yorumlar
      otomatik yapay-delil sayılır; tüm anomali yorumlarının ağırlığı gamma
      ile çarpılır.
    - Dışlanan kümeler (araç/dedektör, moderasyon, sohbet-finans) hiçbir
      metrik ve parmak izi hesabına katılmaz.
    - Eksen destek yorum sayısı ``MIN_SUPPORT`` altındaysa o eksenin
      D/H/S_T değerleri null bırakılır; yapay sayı üretilmez.

Formüller:
    D  = W_real / (W_real + W_ai)
    H  = -(D * log2(D) + (1 - D) * log2(1 - D))   (D in {0,1} -> 0)
    S_T = D * (1 + 0.5 * H)
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from src.schema import RedditPostRaw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_CLUSTERS = PROJECT_ROOT / "data" / "processed" / "unsupervised_clusters.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "scored_posts.jsonl"

GAMMA = 1.5
MIN_SUPPORT = 2

AXIS_DEFS = {
    "anatomy": {
        "label": "Anatomi ve Biyoloji",
        "sub_axes": {
            "hands": {"label": "El ve Parmaklar", "clusters": {76}},
            "limbs": {"label": "Kollar Bacaklar ve Uzuvlar", "clusters": {77, 78, 80}},
            "face": {"label": "Yüz ve Bakış", "clusters": {86, 87}},
            "mouth": {"label": "Ağız ve Diş", "clusters": {84}},
            "animal": {"label": "Canlı ve Hayvan Anatomisi", "clusters": {46, 75}},
        },
    },
    "optics": {
        "label": "Fizik ve Optik",
        "sub_axes": {
            "mirror": {"label": "Ayna ve Yansımalar", "clusters": {58}},
            "shadow": {"label": "Gölge ve Aydınlatma Tutarlılığı", "clusters": {61, 62}},
            "camera": {"label": "Kamera Fiziği ve Alan Derinliği", "clusters": {72}},
        },
    },
    "typography": {
        "label": "Tipografi ve Semantik",
        "sub_axes": {
            "font": {"label": "Font ve Karakter Bozulmaları", "clusters": {27}},
            "glyph": {"label": "Anlamsız Glifler ve Karakterler", "clusters": {25, 26}},
        },
    },
    "render": {
        "label": "Sentetik Doku ve Render",
        "sub_axes": {
            "cgi": {"label": "CGI ve Plastik Render Hissi", "clusters": {3, 112}},
            "color": {"label": "Renk Katmanı ve Filtre Sapması", "clusters": {32}},
            "pixel": {"label": "Çözünürlük ve Piksel Artefaktları", "clusters": {66}},
        },
    },
}

EXCLUDED_CLUSTERS = {0, 1, 2, 6, 7, 8, 9, 13, 22, 98, 103, 107, 110}
FINGERPRINT_ORDER = ("anatomy", "optics", "typography", "render")

MACRO_CLUSTERS: Dict[str, Set[int]] = {}
SUB_CLUSTERS: Dict[str, Set[int]] = {}
CLUSTER_AXIS: Dict[int, Tuple[str, str]] = {}
ANOMALY_CLUSTERS: Set[int] = set()
for _macro_id, _cfg in AXIS_DEFS.items():
    _merged: Set[int] = set()
    for _sub_id, _sub in _cfg["sub_axes"].items():
        SUB_CLUSTERS[_sub_id] = _sub["clusters"]
        _merged |= _sub["clusters"]
        for _c in _sub["clusters"]:
            CLUSTER_AXIS[_c] = (_macro_id, _sub_id)
    MACRO_CLUSTERS[_macro_id] = _merged
    ANOMALY_CLUSTERS |= _merged

REAL_PHRASE_PATTERNS = [
    re.compile(r"\bnot\s+(?:an?\s+)?ai\b"),
    re.compile(r"\bisn'?t\s+(?:an\s+)?ai\b"),
    re.compile(r"\bain'?t\s+(?:an\s+)?ai\b"),
    re.compile(r"\bno\s+ai\b"),
    re.compile(r"\breal\s+(?:human|person|photo|picture)\b"),
    re.compile(r"\b(?:authentic|genuine)\b"),
]
AI_PHRASE_PATTERNS = [
    re.compile(r"\bnot\s+(?:a\s+)?real\b"),
    re.compile(r"\bisn'?t\s+(?:a\s+)?real\b"),
    re.compile(r"\bnot\s+human\b"),
    re.compile(r"\bstable\s+diffusion\b"),
    re.compile(r"\bmidjourney\b"),
    re.compile(r"\bdall[-\s]?e\b"),
    re.compile(r"\bai[- ]generated\b"),
    re.compile(r"\bobviously\s+ai\b"),
    re.compile(r"\b(?:is|looks)\s+ai\b"),
]
REAL_WORDS = {"real", "authentic", "genuine", "human", "natural"}
AI_WORDS = {"ai", "fake", "bot", "generated", "midjourney"}
WORD_RE = re.compile(r"[a-z']+")


@dataclass
class CommentEntry:
    comment_id: str
    body: str
    score: int


@dataclass
class PostData:
    post_id: str
    title: str
    media_type: str
    comments: List[CommentEntry]


class AxisMetrics(BaseModel):
    support: int
    weight: float = 0.0
    D: Optional[float] = None
    H: Optional[float] = None
    S_T: Optional[float] = None


class SubAxisMetrics(AxisMetrics):
    id: str
    label: str
    clusters: List[int]


class ScoredPost(BaseModel):
    post_id: str
    title: str
    media_type: str
    comment_count: int
    excluded_comments: int
    anomaly_comments: int
    W_real: float
    W_ai: float
    D: Optional[float] = None
    H: Optional[float] = None
    S_T: Optional[float] = None
    fingerprint: Dict[str, float] = Field(default_factory=dict)
    axes: Dict[str, AxisMetrics] = Field(default_factory=dict)
    sub_axes: Dict[str, SubAxisMetrics] = Field(default_factory=dict)


def classify_direction(text: str) -> Literal["real", "ai", "neutral"]:
    normalized = " " + text.lower() + " "
    for pattern in REAL_PHRASE_PATTERNS:
        if pattern.search(normalized):
            return "real"
    for pattern in AI_PHRASE_PATTERNS:
        if pattern.search(normalized):
            return "ai"
    tokens = WORD_RE.findall(normalized)
    real_hits = sum(1 for t in tokens if t in REAL_WORDS)
    ai_hits = sum(1 for t in tokens if t in AI_WORDS)
    if ai_hits > real_hits:
        return "ai"
    if real_hits > ai_hits:
        return "real"
    return "neutral"


def _file_for(raw_dir: Path, media_type: str) -> Path:
    if media_type == "image":
        return raw_dir / "raw_posts.jsonl"
    return raw_dir / f"raw_posts_{media_type}.jsonl"


def load_raw_posts(raw_dir: Path, media_types: List[str]) -> Dict[str, PostData]:
    posts: Dict[str, PostData] = {}
    for media_type in media_types:
        path = _file_for(raw_dir, media_type)
        if not path.exists():
            print(f"[yükle] atla: {path.name} mevcut değil")
            continue
        scanned = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                post = RedditPostRaw.model_validate_json(line)
                comments = [
                    CommentEntry(comment.comment_id, comment.body, comment.score)
                    for comment in post.comments
                ]
                posts[post.post_id] = PostData(
                    post_id=post.post_id,
                    title=post.title,
                    media_type=post.media_type,
                    comments=comments,
                )
                scanned += 1
        print(f"[yükle] {path.name}: {scanned:,} gönderi okundu")
    return posts


def _entropy(d: float) -> float:
    if d <= 0.0 or d >= 1.0:
        return 0.0
    return -(d * math.log2(d) + (1 - d) * math.log2(1 - d))


def _summarize(
    w_real: float, w_ai: float, support: int, min_support: int
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if support < min_support:
        return None, None, None
    total = w_real + w_ai
    if total <= 0:
        return None, None, None
    d = w_real / total
    h = _entropy(d)
    st = d * (1.0 + 0.5 * h)
    return d, h, st


def score_post(
    post: PostData,
    comment_clusters: Dict[str, int],
    gamma: float,
    min_support: int,
) -> ScoredPost:
    w_real = 0.0
    w_ai = 0.0
    excluded_comments = 0
    anomaly_comments = 0

    macro_acc = {
        ax: {"support": 0, "w": 0.0, "w_real": 0.0, "w_ai": 0.0}
        for ax in FINGERPRINT_ORDER
    }
    sub_acc: Dict[str, Dict] = {
        sub_id: {"support": 0, "w": 0.0, "w_real": 0.0, "w_ai": 0.0}
        for sub_id in SUB_CLUSTERS
    }

    for comment in post.comments:
        cluster = comment_clusters.get(comment.comment_id)
        if cluster in EXCLUDED_CLUSTERS:
            excluded_comments += 1
            continue
        direction = classify_direction(comment.body)
        weight = float(max(1, comment.score))
        if cluster in ANOMALY_CLUSTERS:
            anomaly_comments += 1
            weight *= gamma
            if direction == "neutral":
                direction = "ai"
        if direction == "real":
            w_real += weight
        elif direction == "ai":
            w_ai += weight

        if cluster in CLUSTER_AXIS:
            macro_id, sub_id = CLUSTER_AXIS[cluster]
            macro = macro_acc[macro_id]
            macro["support"] += 1
            macro["w"] += weight
            if direction == "real":
                macro["w_real"] += weight
            elif direction == "ai":
                macro["w_ai"] += weight
            sub = sub_acc[sub_id]
            sub["support"] += 1
            sub["w"] += weight
            if direction == "real":
                sub["w_real"] += weight
            elif direction == "ai":
                sub["w_ai"] += weight

    d, h, st = _summarize(w_real, w_ai, len(post.comments), 1)

    axes: Dict[str, AxisMetrics] = {}
    for macro_id in FINGERPRINT_ORDER:
        acc = macro_acc[macro_id]
        m_d, m_h, m_st = _summarize(
            acc["w_real"], acc["w_ai"], acc["support"], min_support
        )
        axes[macro_id] = AxisMetrics(
            support=acc["support"], weight=round(acc["w"], 4), D=m_d, H=m_h, S_T=m_st
        )

    sub_axes: Dict[str, SubAxisMetrics] = {}
    for macro_id in FINGERPRINT_ORDER:
        for sub_id, sub_defs in AXIS_DEFS[macro_id]["sub_axes"].items():
            acc = sub_acc[sub_id]
            s_d, s_h, s_st = _summarize(
                acc["w_real"], acc["w_ai"], acc["support"], min_support
            )
            sub_axes[f"{macro_id}.{sub_id}"] = SubAxisMetrics(
                id=sub_id,
                label=sub_defs["label"],
                clusters=sorted(sub_defs["clusters"]),
                support=acc["support"],
                weight=round(acc["w"], 4),
                D=s_d,
                H=s_h,
                S_T=s_st,
            )

    total_w = sum(macro_acc[ax]["w"] for ax in FINGERPRINT_ORDER)
    fingerprint = {
        ax: (macro_acc[ax]["w"] / total_w if total_w > 0 else 0.0)
        for ax in FINGERPRINT_ORDER
    }

    return ScoredPost(
        post_id=post.post_id,
        title=post.title,
        media_type=post.media_type,
        comment_count=len(post.comments),
        excluded_comments=excluded_comments,
        anomaly_comments=anomaly_comments,
        W_real=round(w_real, 4),
        W_ai=round(w_ai, 4),
        D=None if d is None else round(d, 6),
        H=None if h is None else round(h, 6),
        S_T=None if st is None else round(st, 6),
        fingerprint=fingerprint,
        axes=axes,
        sub_axes=sub_axes,
    )


def run(
    raw_dir: Path,
    clusters_file: Path,
    output_path: Path,
    media_types: List[str],
    gamma: float,
    min_support: int,
) -> None:
    console = Console()

    data = json.loads(clusters_file.read_text(encoding="utf-8"))
    comment_clusters = {
        cid: entry["cluster"] for cid, entry in data["comment_cluster_map"].items()
    }
    console.print(
        f"[bold]Kümeleme:[reset] {clusters_file.name} "
        f"{len(comment_clusters):,} yorum -> küme eşlemesi yüklendi"
    )

    posts = load_raw_posts(raw_dir, media_types)
    console.print(f"[bold]Ham gönderi:[reset] {len(posts):,} gönderi okundu")

    scored: List[ScoredPost] = []
    axis_total_weight = {ax: 0.0 for ax in FINGERPRINT_ORDER}
    axis_affected_posts = {ax: 0 for ax in FINGERPRINT_ORDER}
    axis_comments = {ax: 0 for ax in FINGERPRINT_ORDER}

    for post in posts.values():
        result = score_post(post, comment_clusters, gamma, min_support)
        scored.append(result)
        for ax in FINGERPRINT_ORDER:
            metrics = result.axes[ax]
            axis_total_weight[ax] += metrics.weight
            if metrics.support > 0:
                axis_affected_posts[ax] += 1
                axis_comments[ax] += metrics.support

    non_null = [s for s in scored if s.D is not None]
    mean_d = sum(s.D for s in non_null) / len(non_null) if non_null else None
    mean_h = sum(s.H for s in non_null) / len(non_null) if non_null else None
    mean_st = sum(s.S_T for s in non_null) / len(non_null) if non_null else None

    summary = Table(title="Scorer özeti")
    summary.add_column("Metrik", justify="left")
    summary.add_column("Değer", justify="right")
    summary.add_row("Puanlanan gönderi", f"{len(scored):,}")
    summary.add_row(
        "Yönlü kanıt içeren gönderi",
        f"{len(non_null):,}",
    )
    summary.add_row(
        "Ortalama Aldatma Skoru (D)",
        "—" if mean_d is None else f"{mean_d:.4f}",
    )
    summary.add_row(
        "Ortalama Karar Entropisi (H)",
        "—" if mean_h is None else f"{mean_h:.4f}",
    )
    summary.add_row(
        "Ortalama Turing Dayanıklılık (S_T)",
        "—" if mean_st is None else f"{mean_st:.4f}",
    )
    console.print(summary)

    axis_table = Table(title="Anomali eksenleri (kümülatif delil)")
    axis_table.add_column("Eksen", justify="left")
    axis_table.add_column("Etkilenen gönderi", justify="right")
    axis_table.add_column("Delil yorumu", justify="right")
    axis_table.add_column("Ort. eksen ağırlığı/gönderi", justify="right")
    ranked = sorted(
        FINGERPRINT_ORDER,
        key=lambda ax: axis_total_weight[ax] / max(axis_affected_posts[ax], 1),
        reverse=True,
    )
    for ax in ranked:
        axis_table.add_row(
            AXIS_DEFS[ax]["label"],
            f"{axis_affected_posts[ax]:,}",
            f"{axis_comments[ax]:,}",
            f"{(axis_total_weight[ax] / max(axis_affected_posts[ax], 1)):.2f}",
        )
    console.print(axis_table)
    console.print("[bold]En çok hata veren ilk 3 anomali ekseni:[/bold] " + ", ".join(
        AXIS_DEFS[ax]["label"] for ax in ranked[:3]
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for post in scored:
            fh.write(post.model_dump_json() + "\n")
    console.print(f"[bold green]Kaydedildi:[reset] {output_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="r/isthisAI: hiyerarşik bilişsel metrikler (D, H, S_T) ve parmak izi"
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--media-types",
        nargs="+",
        default=["image", "gallery"],
        choices=["image", "gallery", "video", "other"],
    )
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    run(
        raw_dir=args.raw_dir,
        clusters_file=args.clusters,
        output_path=args.output,
        media_types=args.media_types,
        gamma=args.gamma,
        min_support=args.min_support,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())