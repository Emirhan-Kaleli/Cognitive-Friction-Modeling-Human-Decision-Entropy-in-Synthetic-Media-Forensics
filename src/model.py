"""DINOv2 embedding'lerinden çok görevli bilişsel tahmin modeli.

``CognitiveGatingModel`` 384 boyutlu [CLS] embedding alır ve iki kafa çıkarır:
  - Parmak İzi (Head A): 4 eksenli softmax dağılımı
                        (anatomy, optics, typography, render)
  - Aldatma Skoru (Head B): tek boyutlu D tahmini (sigmoid, [0, 1])

Mimari::
    x (384) -> Linear(384,128) -> LayerNorm -> GELU -> Dropout(0.25)
            -> Head A: Linear(128,4) -> Softmax
            -> Head B: Linear(128,32) -> GELU -> Linear(32,1) -> Sigmoid
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CognitiveGatingModel(nn.Module):
    """Paylaşılan boyun katmanı üzerinde iki görevli (multi-task) kafa."""

    def __init__(self, embedding_dim: int = 384) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.neck = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
        )

        self.fingerprint_head = nn.Sequential(
            nn.Linear(128, 4),
            nn.Softmax(dim=-1),
        )

        self.deception_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.neck(x)
        pred_fp = self.fingerprint_head(hidden)
        pred_d = self.deception_head(hidden).squeeze(-1)
        return {"fingerprint": pred_fp, "D": pred_d}