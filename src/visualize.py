"""Bilişsel analiz çıktılarının görselleştirilmesi.

Girdi: data/processed/scored_posts.jsonl
Çıktı: reports/figures/ (her grafik hem 300 dpi PNG hem de SVG)

Üretilen grafikler:
    1. phase_space_scatter  : D-H faz uzayı + Shannon entropi sınırı
    2. axis_entropy_violin  : ana eksenlere göre Karar Entropisi H dağılımı
    3. axis_cognitive_radar       : 4 ana eksenin genel bilişsel profili radarı
    4. archetype_fingerprints     : 3 arketip gönderinin kusur parmak izi paneli
    5. support_matrix             : 12 alt eksenin geçerli skor doluluk matrisi
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCORED = PROJECT_ROOT / "data" / "processed" / "scored_posts.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "figures"

AXIS_ORDER = ("anatomy", "optics", "typography", "render")
AXIS_LABELS = {
    "anatomy": "Anatomi",
    "optics": "Optik",
    "typography": "Tipografi",
    "render": "Render",
}
AXIS_FULL_LABELS = {
    "anatomy": "Anatomi ve Biyoloji",
    "optics": "Fizik ve Optik",
    "typography": "Tipografi ve Semantik",
    "render": "Sentetik Doku ve Render",
}
AXIS_COLORS = {
    "anatomy": "#E8344E",
    "optics": "#2A6F97",
    "typography": "#2A9D8F",
    "render": "#F4A261",
}

SUB_AXIS_ORDER: Dict[str, List[str]] = {
    "anatomy": ["hands", "limbs", "face", "mouth", "animal"],
    "optics": ["mirror", "shadow", "camera"],
    "typography": ["font", "glyph"],
    "render": ["cgi", "color", "pixel"],
}

MEDIAN_CURVE_X = np.linspace(0.001, 0.999, 400)


def _shannon_curve(d: float) -> float:
    if d <= 0.0 or d >= 1.0:
        return 0.0
    return -(d * math.log2(d) + (1 - d) * math.log2(1 - d))


def _setup_theme() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 110,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "grid.color": "#D9DDE3",
            "grid.alpha": 0.7,
            "text.color": "#212529",
            "axes.edgecolor": "#495057",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        },
    )


def load_records(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _dominant_axis(record: dict) -> str:
    return max(AXIS_ORDER, key=lambda ax: record["fingerprint"].get(ax, 0.0))


def _save_figure(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def phase_space_scatter(records: List[dict], out_dir: Path) -> Path:
    points = [r for r in records if r["D"] is not None and r["H"] is not None]
    weights = np.array([max(float(r["W_real"]) + float(r["W_ai"]), 1.0) for r in points])
    max_weight = float(weights.max())
    sizes = 18.0 + 420.0 * (np.log1p(weights) / np.log1p(max_weight))
    colors = [AXIS_COLORS[_dominant_axis(r)] for r in points]
    ds = [float(r["D"]) for r in points]
    hs = [float(r["H"]) for r in points]

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    curve_hs = np.array([_shannon_curve(d) for d in MEDIAN_CURVE_X])
    ax.plot(
        MEDIAN_CURVE_X,
        curve_hs,
        linestyle="--",
        color="#868E96",
        linewidth=1.2,
        alpha=0.9,
        label="Shannon entropisi sınırı",
        zorder=2,
    )
    ax.scatter(
        ds,
        hs,
        s=sizes,
        c=colors,
        alpha=0.65,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    ax.annotate(
        "Hızlı Deşifre",
        xy=(0.14, 0.07),
        fontsize=10.5,
        color="#495057",
        zorder=4,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CED4DA", alpha=0.85),
    )
    ax.annotate(
        "Başarılı Aldatma",
        xy=(0.86, 0.07),
        fontsize=10.5,
        color="#495057",
        zorder=4,
        ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CED4DA", alpha=0.85),
    )
    ax.annotate(
        "Bilişsel Araf (Turing Sınırı)",
        xy=(0.5, 1.0),
        xytext=(0.5, 1.02),
        fontsize=10.5,
        color="#343A40",
        zorder=4,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3BF", ec="#F0D96A", alpha=0.9),
    )

    handles = [Patch(facecolor=AXIS_COLORS[ax], label=AXIS_FULL_LABELS[ax]) for ax in AXIS_ORDER]
    handles.append(Line2D([0], [0], color="#868E96", linestyle="--", label="Shannon sınırı"))
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("Aldatma Skoru D")
    ax.set_ylabel("Karar Entropisi H")
    ax.set_title("Bilişsel Faz Uzayı: Aldatma Skoru ve Karar Entropisi")
    ax.set_xticks(np.arange(0, 1.05, 0.1))
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    fig.tight_layout()
    return _save_figure(fig, "phase_space_scatter", out_dir)


def axis_entropy_violin(records: List[dict], out_dir: Path) -> Path:
    rows: List[Dict[str, object]] = []
    for r in records:
        for ax in AXIS_ORDER:
            metrics = r["axes"].get(ax, {})
            if metrics.get("D") is not None and metrics.get("H") is not None:
                rows.append({"eksen": AXIS_LABELS[ax], "H": float(metrics["H"])})

    if not rows:
        raise SystemExit("Eksen bazlı geçerli skor bulunamadı; violin çizilemedi.")

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    order = [AXIS_LABELS[ax] for ax in AXIS_ORDER]
    palette = [AXIS_COLORS[ax] for ax in AXIS_ORDER]
    sns.violinplot(
        data=df,
        x="eksen",
        y="H",
        hue="eksen",
        order=order,
        palette=palette,
        inner="quart",
        cut=0,
        density_norm="width",
        legend=False,
        ax=ax,
    )
    ax.set_ylabel("Karar Entropisi H")
    ax.set_xlabel("Ana Anomali Ekseni")
    ax.set_title("Eksen Bazlı Entropi Dağılımı (Hangi kusur türü daha çok belirsizlik yaratıyor?)")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return _save_figure(fig, "axis_entropy_violin", out_dir)


def _short_title(title: str, limit: int = 42) -> str:
    title = " ".join(title.split())
    return title if len(title) <= limit else title[: limit - 1] + "…"


ARCHETYPE_PREFERENCES = {
    "anatomy": ["1v0wdzg", "1sw7h6z"],
    "optics": ["1u35u3f", "1rhb54i"],
    "typography": ["1sfdarj", "1vw3m6d"],
}


def _pick_archetype(records: List[dict], dominant: str) -> Optional[dict]:
    by_id = {r["post_id"]: r for r in records}
    for preferred_id in ARCHETYPE_PREFERENCES.get(dominant, []):
        if preferred_id in by_id:
            return by_id[preferred_id]

    scored = [
        r
        for r in records
        if r["axes"].get(dominant, {}).get("support", 0) >= 2
        and r["axes"].get(dominant, {}).get("D") is not None
    ]
    if not scored:
        return None
    if dominant == "anatomy":
        key = lambda r: (r["axes"][dominant]["support"], -r["axes"][dominant]["D"])
    elif dominant == "optics":
        key = lambda r: (r["axes"][dominant].get("H") or -1.0, r["axes"][dominant]["support"])
    else:
        key = lambda r: (
            r["axes"][dominant]["support"],
            r["fingerprint"].get(dominant, 0.0),
        )
    return max(scored, key=key)


RADAR_AXIS_COLORS = {
    "anatomy": "#E63946",
    "optics": "#1D3557",
    "typography": "#2A9D8F",
    "render": "#F4A261",
}

ARCHETYPE_PANEL = [
    ("1v0wdzg", "anatomy", "El / Anatomi Deşifresi"),
    ("1u35u3f", "optics", "Yansıma / Optik Tartışması"),
    ("1sfdarj", "typography", "Sahte Yazı / Tipografi Deşifresi"),
]

RADAR_DIMENSIONS = ("mean_d", "mean_h", "mean_s_t", "freq")
RADAR_DIMENSION_LABELS = ("Ort. D", "Ort. H", "Ort. S_T", "Frekans")


def _axis_profile(records: List[dict]) -> Dict[str, dict]:
    total = len(records)
    profiles: Dict[str, dict] = {}
    for ax in AXIS_ORDER:
        valid = [
            r
            for r in records
            if r["axes"].get(ax, {}).get("D") is not None
            and r["axes"].get(ax, {}).get("H") is not None
        ]
        if valid:
            mean_d = float(np.mean([r["axes"][ax]["D"] for r in valid]))
            mean_h = float(np.mean([r["axes"][ax]["H"] for r in valid]))
            st_vals = [
                r["axes"][ax]["S_T"]
                for r in valid
                if r["axes"][ax].get("S_T") is not None
            ]
            mean_st = float(np.mean(st_vals)) if st_vals else 0.0
            freq = len(valid) / max(total, 1)
        else:
            mean_d = mean_h = mean_st = freq = 0.0
        profiles[ax] = {
            "mean_d": mean_d,
            "mean_h": mean_h,
            "mean_s_t": mean_st,
            "freq": freq,
            "n_valid": len(valid),
        }
    return profiles


def axis_cognitive_radar(records: List[dict], out_dir: Path) -> Path:
    profiles = _axis_profile(records)

    normalized: Dict[str, Dict[str, float]] = {ax: {} for ax in AXIS_ORDER}
    for dim in RADAR_DIMENSIONS:
        values = [profiles[ax][dim] for ax in AXIS_ORDER]
        lo, hi = min(values), max(values)
        for ax in AXIS_ORDER:
            if hi > lo:
                normalized[ax][dim] = (profiles[ax][dim] - lo) / (hi - lo)
            else:
                normalized[ax][dim] = 0.5

    angles = np.linspace(0, 2 * np.pi, len(RADAR_DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7.6, 7.2))
    for ax_key in AXIS_ORDER:
        values = [normalized[ax_key][dim] for dim in RADAR_DIMENSIONS]
        values += values[:1]
        ax.plot(
            angles,
            values,
            color=RADAR_AXIS_COLORS[ax_key],
            linewidth=2,
            label=AXIS_FULL_LABELS[ax_key],
            zorder=3,
        )
        ax.fill(angles, values, color=RADAR_AXIS_COLORS[ax_key], alpha=0.15, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_DIMENSION_LABELS, fontsize=10)
    ax.tick_params(pad=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=8)
    ax.grid(color="#D9DDE3", alpha=0.8)
    ax.set_title("4 Ana Eksenin Genel Bilişsel Profili (normalize)", pad=24, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12), framealpha=0.9)
    fig.tight_layout()
    return _save_figure(fig, "axis_cognitive_radar", out_dir)


def _resolve_panel_record(records: List[dict], post_id: str, fallback_axis: str) -> Optional[dict]:
    for record in records:
        if record["post_id"] == post_id:
            return record
    return _pick_archetype(records, fallback_axis)


def archetype_fingerprints(records: List[dict], out_dir: Path) -> Path:
    panels: List[Tuple[dict, str]] = []
    for post_id, fallback_axis, caption in ARCHETYPE_PANEL:
        record = _resolve_panel_record(records, post_id, fallback_axis)
        if record is None:
            raise SystemExit(
                f"Arketip {post_id} bulunamadı; fingerprint paneli çizilemedi."
            )
        panels.append((record, caption))

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for i, (record, caption) in enumerate(panels):
        ax = axes[i]
        cats = [AXIS_LABELS[a] for a in AXIS_ORDER]
        vals = [float(record["fingerprint"].get(a, 0.0)) for a in AXIS_ORDER]
        colors = [AXIS_COLORS[a] for a in AXIS_ORDER]
        y_pos = np.arange(len(cats))
        ax.barh(y_pos, vals, color=colors, alpha=0.88, edgecolor="white")
        for y, v in zip(y_pos, vals):
            ax.text(
                v + 0.008,
                y,
                f"%{v * 100:.1f}",
                va="center",
                ha="left",
                fontsize=8,
                color="#212529",
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cats, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.1)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7)
        d = record.get("D")
        h = record.get("H")
        st = record.get("S_T")
        d_s = f"{d:.2f}" if d is not None else "—"
        h_s = f"{h:.2f}" if h is not None else "—"
        st_s = f"{st:.2f}" if st is not None else "—"
        ax.set_title(
            f"{record['post_id']} – {_short_title(record['title'], 34)}\n{caption}",
            fontsize=9.5,
            pad=8,
        )
        ax.text(
            0.5,
            -0.16,
            f"D: {d_s} | H: {h_s} | S_T: {st_s}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.8,
            color="#343A40",
        )
    fig.suptitle(
        "Vaka İncelemesi: Arketip Gönderilerin Kusur Parmak İzleri",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    return _save_figure(fig, "archetype_fingerprints", out_dir)


def support_matrix(records: List[dict], out_dir: Path) -> Path:
    total = len(records)
    sub_rows: List[Dict[str, object]] = []
    for macro in AXIS_ORDER:
        for sub_id in SUB_AXIS_ORDER[macro]:
            key = f"{macro}.{sub_id}"
            metrics = {r["post_id"]: r["sub_axes"].get(key, {}) for r in records}
            valid = sum(1 for m in metrics.values() if m.get("support", 0) >= 2)
            label = next(r["sub_axes"][key]["label"] for r in records if key in r["sub_axes"])
            sub_rows.append(
                {
                    "key": key,
                    "label": label,
                    "macro": macro,
                    "valid": valid,
                    "pct": valid / max(total, 1) * 100.0,
                }
            )

    order = [row["key"] for row in sub_rows]
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    bar_colors = [AXIS_COLORS[row["macro"]] for row in sub_rows]
    vals = [row["valid"] for row in sub_rows]
    y_positions = np.arange(len(order))
    ax.barh(y_positions, vals, color=bar_colors, alpha=0.88, edgecolor="white")

    for row, y, value in zip(sub_rows, y_positions, vals):
        ax.text(
            value + max(vals) * 0.01,
            y,
            f"{value:,}  ({row['pct']:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#212529",
        )

    labels = [row["label"] for row in sub_rows]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Geçerli skor üreten gönderi sayısı")
    ax.set_title(f"Alt Eksen Veri Doluluk Matrisi (toplam {total:,} gönderi)")
    ax.set_xlim(0, max(vals) * 1.32)
    handles = [Patch(facecolor=AXIS_COLORS[m], label=AXIS_FULL_LABELS[m]) for m in AXIS_ORDER]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    return _save_figure(fig, "support_matrix", out_dir)


def run(scored_file: Path, out_dir: Path) -> None:
    records = load_records(scored_file)
    valid = [r for r in records if r["D"] is not None]
    median_d = float(np.median([float(r["D"]) for r in valid])) if valid else float("nan")
    median_h = float(np.median([float(r["H"]) for r in valid])) if valid else float("nan")

    _setup_theme()

    produced = [
        phase_space_scatter(records, out_dir),
        axis_entropy_violin(records, out_dir),
        axis_cognitive_radar(records, out_dir),
        archetype_fingerprints(records, out_dir),
        support_matrix(records, out_dir),
    ]

    print(f"[veri] {len(records):,} gönderi | geçerli skor: {len(valid):,}")
    print(f"[özet] medyan Aldatma Skoru (D): {median_d:.4f}")
    print(f"[özet] medyan Karar Entropisi (H): {median_h:.4f}")
    for path in produced:
        print(f"[kayıt] {path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="r/isthisAI bilişsel analiz görselleştirmeleri"
    )
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    run(args.scored, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())