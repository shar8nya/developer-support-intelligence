from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request

from app.container import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


def require_api_key(
    container: Container = Depends(get_container),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    expected = container.settings.api_key
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
