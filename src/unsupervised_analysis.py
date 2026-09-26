"""Denetimsiz NLP analizi: r/isthisAI görsel (image + gallery) yorumları.

Boru hattı:
    yükle -> ön filtreleme -> stop-word temizliği -> embedding (all-MiniLM-L6-v2)
    -> UMAP (384 -> 5) -> HDBSCAN -> c-TF-IDF
    -> terminal özeti + data/processed/unsupervised_clusters.json

Çıktı JSON yapısı:
    clusters         : küme kimliği -> boyut, ortalama puan, top-15 anahtar kelime
    post_cluster_map : post_id -> yorumlarının küme dağılımı ve upvote ağırlıkları

Küme etiketi -1 = gürültü (anomali bölgesi); post bazında ``noise_ratio``
alanı, gürültü yoğunluğu yüksek gönderileri anomali olarak işaretler.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import torch
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from umap import UMAP

from src.schema import RedditPostRaw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "unsupervised_clusters.json"

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_WORDS = 3
BATCH_SIZE = 64

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DELETED_MARKERS = {"[deleted]", "[removed]"}
TOKEN_RE = re.compile(r"[a-z']+")

SUB_NOISE_WORDS = {
    "ai",
    "not",
    "real",
    "fake",
    "image",
    "picture",
    "photo",
    "looks",
    "look",
    "think",
    "definitely",
    "generated",
    "bot",
    "post",
    "repost",
    "people",
}
SUB_PHRASES = ("looks like", "look like")
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) | frozenset(SUB_NOISE_WORDS)


@dataclass
class Comment:
    comment_id: str
    post_id: str
    body: str
    cleaned: str
    score: int


class KeywordWeight(BaseModel):
    word: str
    weight: float


class ClusterInfo(BaseModel):
    size: int
    avg_score: float
    keywords: List[KeywordWeight] = Field(default_factory=list)


class PostClusterEntry(BaseModel):
    n_comments: int
    noise_count: int
    noise_ratio: float
    cluster_counts: Dict[str, int] = Field(default_factory=dict)
    cluster_upvotes: Dict[str, int] = Field(default_factory=dict)
    total_score: int


class CommentClusterEntry(BaseModel):
    post_id: str
    cluster: int


class AnalysisOutput(BaseModel):
    meta: Dict[str, object]
    clusters: Dict[str, ClusterInfo]
    post_cluster_map: Dict[str, PostClusterEntry]
    comment_cluster_map: Dict[str, CommentClusterEntry] = Field(default_factory=dict)


def _clean_body(text: str) -> str:
    lowered = " " + text.lower().replace("'", " ") + " "
    for phrase in SUB_PHRASES:
        lowered = lowered.replace(phrase, " ")
    tokens = TOKEN_RE.findall(lowered)
    kept = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(kept)


def _file_for(raw_dir: Path, media_type: str) -> Path:
    if media_type == "image":
        return raw_dir / "raw_posts.jsonl"
    return raw_dir / f"raw_posts_{media_type}.jsonl"


def load_media_comments(
    raw_dir: Path,
    media_types: List[str],
    min_words: int,
    max_comments: Optional[int] = None,
) -> Tuple[List[Comment], Counter]:
    comments: List[Comment] = []
    dropped: Counter = Counter()
    seen_ids: set = set()

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
                for comment in post.comments:
                    if max_comments is not None and len(comments) >= max_comments:
                        break
                    if comment.comment_id in seen_ids:
                        continue
                    seen_ids.add(comment.comment_id)
                    body = (comment.body or "").strip()
                    if not body or body.lower() in DELETED_MARKERS:
                        dropped["silinmiş"] += 1
                        continue
                    if URL_RE.search(body):
                        dropped["url"] += 1
                        continue
                    if len(body.split()) < min_words:
                        dropped["kısa"] += 1
                        continue
                    cleaned = _clean_body(body)
                    if not cleaned:
                        dropped["temizlik-sonrası-boş"] += 1
                        continue
                    comments.append(
                        Comment(comment.comment_id, post.post_id, body, cleaned, comment.score)
                    )
                scanned += 1
                if max_comments is not None and len(comments) >= max_comments:
                    break
        print(f"[yükle] {path.name}: {scanned:,} gönderi tarandı")
    return comments, dropped


def compute_ctfidf(
    cluster_texts: Dict[int, List[str]], top_n: int
) -> Dict[int, List[Tuple[str, float]]]:
    tf: Dict[int, Counter] = {}
    total_freq: Counter = Counter()

    for cluster_id, texts in cluster_texts.items():
        counter: Counter = Counter()
        for text in texts:
            counter.update(text.split())
        tf[cluster_id] = counter
        total_freq.update(counter)

    n_clusters = max(len(tf), 1)
    avg_words = sum(sum(c.values()) for c in tf.values()) / n_clusters

    result: Dict[int, List[Tuple[str, float]]] = {}
    for cluster_id, counter in tf.items():
        weights = {
            word: count * math.log(1.0 + avg_words / max(total_freq[word], 1))
            for word, count in counter.items()
        }
        result[cluster_id] = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return result


def run(
    raw_dir: Path,
    output_path: Path,
    media_types: List[str],
    min_words: int,
    max_comments: Optional[int],
    model_name: str,
    batch_size: int,
    n_neighbors: int,
    n_components: int,
    min_dist: float,
    umap_metric: str,
    random_state: int,
    min_cluster_size: int,
    min_samples: int,
    hdbscan_metric: str,
    top_n: int,
) -> None:
    console = Console()

    comments, dropped = load_media_comments(raw_dir, media_types, min_words, max_comments)
    if not comments:
        print("Analiz edilecek yorum bulunamadı.")
        return
    console.print(
        f"[bold green]Yükleme:[reset] {len(comments):,} yorum "
        f"(atlanan: {dict(dropped)})"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    console.print(f"[bold]Embedding:[reset] cihaz={device}, model={model_name}")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        [c.cleaned for c in comments],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    console.print(
        f"[bold]UMAP:[reset] boyut 384 -> {n_components} "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist})"
    )
    reducer = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=umap_metric,
        random_state=random_state,
    )
    umap_embeddings = reducer.fit_transform(embeddings)

    console.print(
        f"[bold]HDBSCAN:[reset] min_cluster_size={min_cluster_size}, "
        f"min_samples={min_samples}, metric={hdbscan_metric}"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=hdbscan_metric,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(umap_embeddings)
    labels_int = np.asarray(labels, dtype=int)

    n_total = int(len(labels_int))
    n_noise = int((labels_int == -1).sum())
    noise_ratio = n_noise / max(n_total, 1)
    console.print(
        f"[bold]Kümeleme:[reset] toplam yorum={n_total:,}, "
        f"gürültü (-1)={n_noise:,} (%{noise_ratio * 100:.1f}), "
        f"kümelenmiş={n_total - n_noise:,}"
    )

    cluster_texts: Dict[int, List[str]] = {}
    cluster_scores: Dict[int, List[int]] = {}
    for cluster_id, comment in zip(labels_int, comments):
        if cluster_id < 0:
            continue
        cid = int(cluster_id)
        cluster_texts.setdefault(cid, []).append(comment.cleaned)
        cluster_scores.setdefault(cid, []).append(comment.score)

    keywords = compute_ctfidf(cluster_texts, top_n=top_n)

    cluster_rows: List[Tuple[int, int, float, List[Tuple[str, float]]]] = []
    for cid in sorted(cluster_texts):
        scores = cluster_scores[cid]
        cluster_rows.append((cid, len(scores), float(np.mean(scores)), keywords[cid]))
    cluster_rows.sort(key=lambda row: row[1], reverse=True)

    table = Table(title="c-TF-IDF küme özeti")
    table.add_column("Küme", justify="right")
    table.add_column("Yorum", justify="right")
    table.add_column("Ort. Upvote", justify="right")
    table.add_column("En belirleyici 10 kelime")
    for cid, size, avg_score, kws in cluster_rows:
        top_words = ", ".join(word for word, _ in kws[:10])
        table.add_row(str(cid), f"{size:,}", f"{avg_score:.2f}", top_words)
    console.print(table)

    post_map: Dict[str, Dict] = {}
    for cluster_id, comment in zip(labels_int, comments):
        entry = post_map.setdefault(
            comment.post_id,
            {
                "n_comments": 0,
                "noise_count": 0,
                "cluster_counts": {},
                "cluster_upvotes": {},
                "total_score": 0,
            },
        )
        cid_str = str(int(cluster_id))
        entry["n_comments"] += 1
        entry["total_score"] += comment.score
        entry["cluster_counts"][cid_str] = entry["cluster_counts"].get(cid_str, 0) + 1
        entry["cluster_upvotes"][cid_str] = (
            entry["cluster_upvotes"].get(cid_str, 0) + comment.score
        )
        if cluster_id == -1:
            entry["noise_count"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clusters_out = {
        str(cid): ClusterInfo(
            size=size,
            avg_score=round(avg_score, 4),
            keywords=[
                KeywordWeight(word=word, weight=round(weight, 6)) for word, weight in kws
            ],
        )
        for cid, size, avg_score, kws in cluster_rows
    }
    post_map_out = {
        post_id: PostClusterEntry(
            n_comments=entry["n_comments"],
            noise_count=entry["noise_count"],
            noise_ratio=round(entry["noise_count"] / max(entry["n_comments"], 1), 4),
            cluster_counts=entry["cluster_counts"],
            cluster_upvotes=entry["cluster_upvotes"],
            total_score=entry["total_score"],
        )
        for post_id, entry in post_map.items()
    }

    meta: Dict[str, object] = {
        "model": model_name,
        "device": device,
        "embedding_dim": int(embeddings.shape[1]),
        "n_comments": n_total,
        "n_noise": n_noise,
        "noise_ratio": round(noise_ratio, 4),
        "n_posts": len(post_map),
        "media_types": media_types,
        "prefilter": {"min_words": min_words, "dropped": dict(dropped)},
        "umap": {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_dist": min_dist,
            "metric": umap_metric,
            "random_state": random_state,
        },
        "hdbscan": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": hdbscan_metric,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    comment_cluster_map = {
        comment.comment_id: CommentClusterEntry(post_id=comment.post_id, cluster=int(cluster_id))
        for cluster_id, comment in zip(labels_int, comments)
    }

    payload = AnalysisOutput(
        meta=meta,
        clusters=clusters_out,
        post_cluster_map=post_map_out,
        comment_cluster_map=comment_cluster_map,
    )
    output_path.write_text(
        json.dumps(payload.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"[bold green]Kaydedildi:[reset] {output_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="r/isthisAI görsel gönderi yorumları: denetimsiz NLP boru hattı"
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--media-types",
        nargs="+",
        default=["image", "gallery"],
        choices=["image", "gallery", "video", "other"],
    )
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    parser.add_argument("--max-comments", type=int, default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--min-dist", type=float, default=0.0)
    parser.add_argument("--umap-metric", default="cosine")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--hdbscan-metric", default="euclidean")
    parser.add_argument("--top-n", type=int, default=15)
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
        output_path=args.output,
        media_types=args.media_types,
        min_words=args.min_words,
        max_comments=args.max_comments,
        model_name=args.model_name,
        batch_size=args.batch_size,
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        min_dist=args.min_dist,
        umap_metric=args.umap_metric,
        random_state=args.random_state,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        hdbscan_metric=args.hdbscan_metric,
        top_n=args.top_n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())