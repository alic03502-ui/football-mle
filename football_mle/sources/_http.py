"""Small HTTP helpers for fetching remote CSV/JSON reliably."""

from __future__ import annotations

import io
import time

import pandas as pd
import requests


__all__ = ["read_csv_url", "read_json_url"]


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def _get(
    url: str,
    *,
    timeout: int = 30,
    retries: int = 4,
) -> requests.Response:
    """GET a URL with retries for temporary server/network failures."""

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                headers=_HEADERS,
                timeout=timeout,
            )

            if response.status_code not in _RETRY_STATUS_CODES:
                response.raise_for_status()
                return response

            last_error = requests.HTTPError(
                f"HTTP {response.status_code} for {url}"
            )

        except requests.RequestException as exc:
            last_error = exc

        if attempt < retries:
            # 1s, 2s, 4s, 8s
            time.sleep(2**attempt)

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"Unable to download {url}")


def read_csv_url(
    url: str,
    *,
    encoding: str = "utf-8",
    timeout: int = 30,
    **kwargs: object,
) -> pd.DataFrame:
    """Fetch a CSV URL with retry/backoff and parse it into a DataFrame."""

    response = _get(
        url,
        timeout=timeout,
    )

    text = response.content.decode(
        encoding,
        errors="ignore",
    )

    return pd.read_csv(
        io.StringIO(text),
        **kwargs,
    )


def read_json_url(
    url: str,
    *,
    timeout: int = 30,
) -> dict:
    """Fetch a JSON URL with retry/backoff."""

    response = _get(
        url,
        timeout=timeout,
    )

    return response.json()
