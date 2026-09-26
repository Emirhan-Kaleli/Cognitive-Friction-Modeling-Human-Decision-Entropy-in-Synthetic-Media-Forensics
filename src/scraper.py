"""r/isthisAI kazıma motoru ve CLI giriş noktası (httpx + Cookie Injection).

Reddit'in herkese açık JSON uçlarını, tarayıcı oturumu çerezleriyle taklit
ederek kullanır:
    - Liste:    GET /r/{subreddit}/{sort}.json?limit=N[&t=day]
    - Yorumlar: GET /r/{subreddit}/comments/{post_id}.json

Kullanım: ``python -m src.scraper --limit 10 --sort hot``
          ``python -m src.scraper --limit 1500 --combine``

Gönderiler medya türüne göre ayrı JSON-L dosyalarına yazılır:
    - image   -> {output}                 (varsayılan: raw_posts.jsonl)
    - video   -> {output}_videos.jsonl
    - gallery -> {output}_galleries.jsonl
    - other   -> {output}_other.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import httpx
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

from src.config import build_client
from src.schema import MediaType, RedditComment, RedditPostRaw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "raw_posts.jsonl"

SUBREDDIT_NAME = "isthisAI"
MAX_COMMENTS = 30
IMAGE_HOSTS = ("i.redd.it", "i.imgur.com")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_HOSTS = (
    "v.redd.it",
    "youtube.com",
    "youtu.be",
    "twitch.tv",
    "vimeo.com",
    "instagram.com/reel/",
    "tiktok.com/",
)
SORT_OPTIONS = ("hot", "new", "top")
DEFAULT_RETRIES = 3
REQUEST_DELAY = 1.5

# --combine modunda sırayla taranan kaynaklar (sıralama, time_filter).
# Tek bir sıralamada Reddit sayfalamayı ~1000 gönderiyle sabitler; birden çok
# pencere birleştirilip post_id üzerinden tekilleştirilerek daha fazla
# benzersiz gönderi elde edilir.
COMBINE_SOURCES: tuple[tuple[str, Optional[str]], ...] = (
    ("new", None),
    ("top", "year"),
    ("top", "all"),
    ("top", "month"),
    ("hot", None),
)

FORBIDDEN_WARNING = (
    "[!] HTTP 403 Hatası: .env dosyasındaki REDDIT_COOKIE değerinin "
    "güncel olduğundan emin olun."
)

_console = Console()


@dataclass(frozen=True)
class ScrapeConfig:
    """Kazıma işleminin çalışma parametreleri."""

    limit: int = 50
    sort: str = "hot"
    output: Path = DEFAULT_OUTPUT
    time_filter: Optional[str] = None
    delay: float = REQUEST_DELAY


class ScraperError(RuntimeError):
    """Kazıma sırasında kurtarılamayan hatalar."""


def _is_direct_image(url: str) -> bool:
    lowered = url.lower()
    if any(host in lowered for host in IMAGE_HOSTS):
        return True
    return lowered.split("?", 1)[0].endswith(IMAGE_EXTENSIONS)


def _classify_media(url: str, is_video: bool = False) -> MediaType:
    lowered = url.lower()
    if "/gallery/" in lowered:
        return "gallery"
    if is_video or any(host in lowered for host in VIDEO_HOSTS):
        return "video"
    if _is_direct_image(lowered):
        return "image"
    return "other"


def _gallery_media_urls(data: dict) -> List[str]:
    """Galeri gönderisinin bireysel görsellerini (preview URL'leri) toplar."""
    metadata = data.get("media_metadata") or {}
    items = (data.get("gallery_data") or {}).get("items") or []
    urls: List[str] = []
    for item in items:
        meta = metadata.get(str(item.get("media_id", "")))
        if not meta:
            continue
        source = meta.get("s") or {}
        url = source.get("u") or source.get("gif")
        if url:
            urls.append(url)
    return urls


def _video_media_urls(data: dict, url: str) -> List[str]:
    video = (data.get("media") or {}).get("reddit_video") or {}
    urls: List[str] = []
    fallback = video.get("fallback_url")
    if fallback:
        urls.append(fallback)
    thumbnail = data.get("thumbnail")
    if isinstance(thumbnail, str) and thumbnail.startswith("http"):
        urls.append(thumbnail)
    if not urls:
        urls.append(url)
    return list(dict.fromkeys(urls))


def _media_urls(media_type: MediaType, data: dict, url: str) -> List[str]:
    if media_type == "gallery":
        urls = _gallery_media_urls(data)
        return urls or [url]
    if media_type == "video":
        return _video_media_urls(data, url)
    return [url]


def _output_for(output: Path, media_type: MediaType) -> Path:
    if media_type == "image":
        return output
    return output.with_name(f"{output.stem}_{media_type}{output.suffix}")


def _flatten_comments(
    children: List[dict], out: List[RedditComment]
) -> List[RedditComment]:
    """Yorum Listing ağacını düzleştirir; sticky ve AutoModerator'ı atlar."""
    for child in children:
        if len(out) >= MAX_COMMENTS:
            break
        if child.get("kind", "") != "t1":
            continue
        data = child.get("data", {})
        if data.get("stickied"):
            continue
        author = str(data.get("author") or "[deleted]")
        if author.lower() == "automoderator":
            continue
        body = str(data.get("body") or "").strip()
        if body and body not in ("[deleted]", "[removed]"):
            out.append(
                RedditComment(
                    comment_id=str(data.get("id", "")),
                    author=author,
                    body=body,
                    score=int(data.get("score", 0)),
                    created_utc=float(data.get("created_utc", 0.0)),
                )
            )
        replies = data.get("replies")
        if isinstance(replies, dict) and replies.get("kind") == "Listing":
            _flatten_comments(replies.get("data", {}).get("children", []), out)
    return out


class RedditScraper:
    """JSON uçları üzerinden gönderi ve yorum toplayıcı (httpx)."""

    def __init__(
        self,
        client: httpx.Client,
        limit: int = 50,
        sort: str = "hot",
        time_filter: Optional[str] = None,
        output: Path = DEFAULT_OUTPUT,
        delay: float = REQUEST_DELAY,
    ) -> None:
        if limit < 1:
            raise ValueError("--limit en az 1 olmalı")
        if sort not in SORT_OPTIONS:
            raise ValueError(f"--sort şunlardan biri olmalı: {', '.join(SORT_OPTIONS)}")
        self.client = client
        self.limit = limit
        self.sort = sort
        self.time_filter = time_filter
        self.output = output
        self.delay = delay
        self._wrote_rows: int = 0
        self._counts: dict[str, int] = {}
        self._seen_ids: set[str] = set()
        self._already_complete: bool = False

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, DEFAULT_RETRIES + 1):
            try:
                response = self.client.get(path, params=params)
                if response.status_code == 403:
                    _console.print(f"[bold yellow]{FORBIDDEN_WARNING}[/bold yellow]")
                    raise ScraperError(FORBIDDEN_WARNING)
                if response.status_code == 429:
                    raise ScraperError(f"{path}: 429 Too Many Requests")
                response.raise_for_status()
                return response.json()
            except ScraperError as exc:
                last_exc = exc
            except (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
            time.sleep(REQUEST_DELAY * attempt)
        raise ScraperError(f"{path} başarısız (3 deneme): {last_exc}")

    def _list_posts(self, limit: Optional[int] = None) -> List[dict]:
        """Listing uçlarını ``after`` imleciyle sayfalayarak gönderi toplar.

        Reddit JSON listing'i sayfa başına en fazla 100 gönderi döndürür;
        ``limit`` 100'ün üzerindeyse sayfalar arası ``data.after`` takibiyle
        daha fazlası çekilir. Her sayfa arasında ``delay`` kadar güvenli boşluk
        bırakılır.
        """
        target = self.limit if limit is None else limit
        if target is None or target <= 0:
            return []
        posts: List[dict] = []
        base_params: dict[str, Any] = {"limit": min(target, 100)}
        if self.sort == "top":
            base_params["t"] = self.time_filter or "all"
        after: Optional[str] = None
        while len(posts) < target:
            params = dict(base_params)
            if after:
                params["after"] = after
            payload = self._get(f"/r/{SUBREDDIT_NAME}/{self.sort}.json", params=params)
            data = payload.get("data", {})
            children = data.get("children", [])
            if not children:
                break
            posts.extend(
                child.get("data", {})
                for child in children
                if child.get("kind") == "t3"
            )
            new_after = data.get("after")
            if not new_after or new_after == after:
                break
            after = new_after
            if len(posts) < target:
                time.sleep(self.delay)
        return posts[:target]

    def _fetch_comments(self, post_id: str) -> List[RedditComment]:
        payload = self._get(f"/r/{SUBREDDIT_NAME}/comments/{post_id}.json")
        if not isinstance(payload, list) or len(payload) < 2:
            return []
        children = payload[1].get("data", {}).get("children", [])
        return _flatten_comments(children, [])

    def _extract_post(self, data: dict) -> Optional[RedditPostRaw]:
        post_id = str(data.get("id", ""))
        if not post_id:
            return None
        raw_url = str(data.get("url", ""))
        media_type = _classify_media(raw_url, bool(data.get("is_video")))
        return RedditPostRaw(
            post_id=post_id,
            title=str(data.get("title", "")),
            url=raw_url,
            media_type=media_type,
            media_urls=_media_urls(media_type, data, raw_url),
            flair=data.get("link_flair_text"),
            score=int(data.get("score", 0)),
            upvote_ratio=float(data.get("upvote_ratio", 0.0)),
            created_utc=float(data.get("created_utc", 0.0)),
            num_comments=int(data.get("num_comments", 0)),
            is_image=media_type == "image",
            comments=self._fetch_comments(post_id),
        )

    @staticmethod
    def _jsonl_row(post: RedditPostRaw) -> str:
        return json.dumps(post.model_dump(mode="json"), ensure_ascii=False) + "\n"

    def _write_row(self, post: RedditPostRaw) -> None:
        target = _output_for(self.output, post.media_type)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(self._jsonl_row(post))
        self._wrote_rows += 1
        self._counts[post.media_type] = self._counts.get(post.media_type, 0) + 1

    def run(self) -> dict[str, int]:
        """Gönderileri listeler, medya türüne göre sınıflandırır ve ayrı
        JSON-L dosyalarına satır satır yazar."""
        self._wrote_rows = 0
        self._counts = {}
        posts = self._list_posts()

        with tqdm(total=len(posts), desc="Gönderiler", unit="post") as bar:
            for raw in posts:
                post_id = str(raw.get("id", ""))
                bar.set_description(f"işleniyor: {post_id}")
                try:
                    post = self._extract_post(raw)
                    if post is None:
                        bar.update(1)
                        continue
                    bar.set_postfix_str(f"{post.media_type}: {post_id}")
                    self._write_row(post)
                except (ValidationError, ScraperError):
                    bar.set_postfix_str(f"hata: {post_id}")
                bar.update(1)
                time.sleep(self.delay)
        return self._counts

    def _seed_seen_ids(self) -> None:
        """Mevcut çıktı dosyalarındaki post_id'leri okuyarak tekrar kaydı engeller."""
        for media_type in ("image", "video", "gallery", "other"):
            path = _output_for(self.output, media_type)
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        pid = row.get("post_id")
                        if pid:
                            self._seen_ids.add(pid)
            except OSError:
                continue

    def _run_posts(self, sort: str, time_filter: Optional[str], budget: int, bar) -> int:
        """Tek kaynağın tam listing'ini (1000 satıra kadar) tarar ve
        ``budget`` kadarında bütçe dolduğunda durur. Tekilleştirme sayesinde
        kaynakların derin satırları da taranır."""
        self.sort = sort
        self.time_filter = time_filter
        collected = 0
        for raw in self._list_posts(limit=1000):
            if collected >= budget:
                break
            post_id = str(raw.get("id", ""))
            if not post_id or post_id in self._seen_ids:
                continue
            self._seen_ids.add(post_id)
            try:
                post = self._extract_post(raw)
                if post is None:
                    continue
                bar.set_postfix_str(f"{sort}: {post.media_type}: {post_id}")
                self._write_row(post)
                collected += 1
                bar.update(1)
            except (ValidationError, ScraperError):
                bar.set_postfix_str(f"hata: {post_id}")
        return collected

    def run_combined(self) -> dict[str, int]:
        """``COMBINE_SOURCES`` listesini tarar; post_id ile tekilleştirip
        çıktı dosyalarındaki benzersiz toplamı ``self.limit`` düzeyine tamamlar.

        ``self.limit`` tek çekimde toplanacak miktarı değil, dosyalardaki
        nihai benzersiz gönderi hedefini (mevcutlar dahil) ifade eder.
        """
        self._wrote_rows = 0
        self._counts = {}
        self._seen_ids = set()
        self._seed_seen_ids()
        initial = len(self._seen_ids)
        remaining = self.limit - initial
        if remaining <= 0:
            self._already_complete = True
            return self._counts
        self._already_complete = False
        added = 0
        with tqdm(total=remaining, desc=f"Eklenecek (mevcut {initial})", unit="post") as bar:
            for sort, time_filter in COMBINE_SOURCES:
                if added >= remaining:
                    break
                bar.set_description(f"kaynak: {sort}/{time_filter or 'all'}")
                gained = self._run_posts(sort, time_filter, remaining - added, bar)
                added += gained
                if gained == 0 and self._wrote_rows == 0:
                    time.sleep(self.delay)
        return self._counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.scraper",
        description=f"r/{SUBREDDIT_NAME} gönderilerini ve yorumlarını medya türüne göre ayrılmış JSON-L dosyalarına toplar (httpx + Cookie Injection).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Çekilecek gönderi sayısı (varsayılan: 50)",
    )
    parser.add_argument(
        "--sort",
        choices=SORT_OPTIONS,
        default="hot",
        help="Sıralama türü (varsayılan: hot)",
    )
    parser.add_argument(
        "--time-filter",
        choices=("hour", "day", "week", "month", "year", "all"),
        default=None,
        help="Yalnızca --sort top ile anlamlıdır (varsayılan: all)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            f"JSON-L çıktı yolu (varsayılan: {DEFAULT_OUTPUT}). "
            "Tür başına ayrı dosya: video/gallery/other için türev yollar "
            "(raw_posts_videos.jsonl vb.) otomatik oluşturulur."
        ),
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help=(
            "Tekil birleştirmeli mod: new + top/year + top/all + top/month + hot "
            "sırayla taranır; post_id ile tekilleştirilerek çıktı dosyalarındaki "
            "benzersiz gönderi toplamı --limit hedefine tamamlanır. Mevcut "
            "kayıtlar tekrar yazılmaz; aynı komut güvenle tekrarlanabilir."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"İstekler arası bekleme, saniye (varsayılan: {REQUEST_DELAY})",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    client = build_client()
    scraper = RedditScraper(
        client=client,
        limit=args.limit,
        sort=args.sort,
        time_filter=args.time_filter,
        output=Path(args.output),
        delay=args.delay,
    )

    try:
        counts = scraper.run_combined() if args.combine else scraper.run()
        written = sum(counts.values())
        if args.combine and scraper._already_complete:
            _console.print(
                Panel(
                    f"Hedef olan {args.limit} benzersiz kayıt zaten mevcut, yeni kayıt eklenmedi.",
                    title="Tamamlandı",
                    style="bold green",
                )
            )
            return 0
        _console.print(
            Panel(
                f"Yazılan kayıt: {written} satır\n"
                + "\n".join(
                    f"  {media_type:<8}: {count} satır -> {_output_for(scraper.output, media_type)}"
                    for media_type, count in sorted(counts.items())
                ),
                title="Tamamlandı",
                style="bold green",
            )
        )
        return 0 if written > 0 else 1
    except ScraperError as exc:
        _console.print(Panel(f"Hata: {exc}", title="Başarısız", style="bold red"))
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())