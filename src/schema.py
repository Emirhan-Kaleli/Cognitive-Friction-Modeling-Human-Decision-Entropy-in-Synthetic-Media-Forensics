"""Pydantic şemaları: r/isthisAI ham veri modeli.

Bu modül, kazıma (scraping) işleminin ürettiği tip güvenli (type-safe)
veri yapılarını tanımlar. JSON-L çıktısı tamamen bu modeller üzerinden
doğrulanarak serileştirilir.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

MediaType = Literal["image", "video", "gallery", "other"]


class RedditComment(BaseModel):
    """Bir Reddit yorumunun immutably saklanan temel alanları."""

    model_config = ConfigDict(extra="allow", frozen=True)

    comment_id: str = Field(description="Yorumun benzersiz Reddit kimliği")
    author: str = Field(description="Yorum yazarının kullanıcı adı")
    body: str = Field(description="Yorum metni")
    score: int = Field(description="Yorumun toplam puanı (negatif olabilir)")
    created_utc: float = Field(description="Unix zaman damgası (saniye)")


class RedditPostRaw(BaseModel):
    """Reddit gönderisinin medya tipine göre ham kaydı.

    ``is_image`` yalnızca doğrudan görsel gönderiler için ``True`` olur;
    ``media_type``, medyanın analitik olarak ayrıştırılabilmesi için
    gönderinin fiili türünü (image / video / gallery / other) taşır.
    """

    model_config = ConfigDict(extra="allow")

    post_id: str = Field(description="Gönderinin benzersiz Reddit kimliği")
    title: str = Field(description="Gönderinin başlığı")
    url: str = Field(description="Gönderinin kaynak URL'si")
    media_type: MediaType = Field(
        description="Medya türü: image | video | gallery | other",
        default="image",
    )
    media_urls: List[str] = Field(
        default_factory=list,
        description="Doğrudan medya varlıklarının URL'leri (galeri görselleri, video mp4 vb.)",
    )
    flair: Optional[str] = Field(
        default=None, description="Gönderinin flair etiketi, yoksa None"
    )
    score: int = Field(description="Gönderinin toplam puanı (negatif olabilir)")
    upvote_ratio: float = Field(ge=0.0, le=1.0, description="Upvote oranı (0-1)")
    created_utc: float = Field(description="Unix zaman damgası (saniye)")
    num_comments: int = Field(ge=0, description="Toplam yorum sayısı")
    is_image: bool = Field(description="Gönderinin doğrudan görsel içerip içermediği")
    comments: List[RedditComment] = Field(
        default_factory=list,
        description="Stickied/AutoModerator yorumları hariç ilk N yorum",
    )