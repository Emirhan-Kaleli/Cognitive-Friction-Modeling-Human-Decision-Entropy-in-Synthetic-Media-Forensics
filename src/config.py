"""Yapılandırma: .env yükleme ve httpx istemcisi kurulumu (Cookie Injection)."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_BASE_URL = "https://www.reddit.com"
TIMEOUT_SECONDS = 15.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_COOKIE_PLACEHOLDER = "buraya_tarayicidan_alinan_cookie_gelecek"


def _load_env() -> None:
    load_dotenv(dotenv_path=_ENV_PATH, override=False)


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def build_client(
    base_url: str = DEFAULT_BASE_URL,
    user_agent: str | None = None,
    cookie: str | None = None,
    timeout: float = TIMEOUT_SECONDS,
) -> httpx.Client:
    """Tarayıcı oturumunu taklit eden httpx istemcisini kurar.

    Başlıklar standart bir tarayıcıyı (Chrome) andırır; ``REDDIT_COOKIE``
    ortam değişkeni tanımlıysa tam değeri ``Cookie`` başlığı olarak eklenir.
    Bu, Reddit'in bot koruması (403/Cloudflare WAF) tarafından engellenmeden
    herkese açık JSON uçlarına erişimi sağlar.
    """
    _load_env()

    resolved_agent = user_agent or _env("USER_AGENT") or DEFAULT_USER_AGENT
    resolved_cookie = cookie or _env("REDDIT_COOKIE")
    if resolved_cookie == _COOKIE_PLACEHOLDER:
        resolved_cookie = ""

    headers: dict[str, str] = {
        "User-Agent": resolved_agent,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_url + "/",
    }
    if resolved_cookie:
        headers["Cookie"] = resolved_cookie

    return httpx.Client(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
    )