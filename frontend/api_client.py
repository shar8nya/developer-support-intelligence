"""Thin HTTP client for the FastAPI backend (used by the Streamlit UI)."""
from __future__ import annotations

from typing import Any, Optional

import httpx


class ApiError(Exception):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.message, self.status = message, status


def _friendly(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}: {resp.text[:200]}"
    if isinstance(body, dict):
        if "error" in body and isinstance(body["error"], dict):
            return body["error"].get("message", str(body["error"]))
        detail = body.get("detail")
        if isinstance(detail, list):  # FastAPI/Pydantic validation errors
            return "; ".join(f"{'.'.join(str(p) for p in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail)
        if detail:
            return str(detail)
    return f"HTTP {resp.status_code}"


class ApiClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 90.0,
                 client: Optional[httpx.Client] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self._client = client or httpx.Client(timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        try:
            resp = self._client.request(method, f"{self.base_url}{path}", headers=headers, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url}. Is the backend running? "
                           f"(uvicorn app.main:app)") from exc
        except httpx.TimeoutException as exc:
            raise ApiError("The API took too long to respond. Try again or reduce top_k.") from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Network error: {exc}") from exc
        if resp.status_code >= 400:
            raise ApiError(_friendly(resp), resp.status_code)
        return resp.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def chat(self, payload: dict) -> dict:
        return self._request("POST", "/chat", json=payload)

    def search(self, payload: dict) -> dict:
        return self._request("POST", "/search", json=payload)

    def ingest(self, payload: dict) -> dict:
        return self._request("POST", "/ingest", json=payload)

    def ingest_status(self, job_id: str) -> dict:
        return self._request("GET", f"/ingest/{job_id}")

    def feedback(self, interaction_id: str, rating: int, comment: Optional[str] = None) -> dict:
        return self._request("POST", "/feedback", json={"interaction_id": interaction_id, "rating": rating,
                                                        "comment": comment})
