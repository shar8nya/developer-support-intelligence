from __future__ import annotations

from app.config import Settings
from app.repositories.base import Repository
from app.repositories.memory import InMemoryRepository


def build_repository(s: Settings) -> Repository:
    if s.is_demo or not s.database_url:
        return InMemoryRepository()
    from app.repositories.supabase import SupabaseRepository  # lazy: needs a reachable DB
    return SupabaseRepository(s.database_url)
