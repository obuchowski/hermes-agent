"""Resolve sessions across profile-local and canonical gateway storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ResolvedSessionRef:
    """A resolved session together with its authoritative database."""

    session_id: str
    profile_name: str
    storage_scope: str
    db: Any
    owns_db: bool = False


def _same_db(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        return Path(left.db_path).resolve() == Path(right.db_path).resolve()
    except Exception:
        return False


def _resolve_in_db(db: Any, target: str) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    session = db.get_session(target)
    session_id = str(session["id"]) if session else db.resolve_session_by_title(target)
    if not session_id:
        return None, None
    try:
        session_id = db.resolve_resume_session_id(session_id) or session_id
    except Exception:
        pass
    return session_id, db.get_session(session_id)


def _profile_owns_root_row(row: dict[str, Any], active_profile: str) -> bool:
    row_profile = str(row.get("profile_name") or "").strip()
    if active_profile == "default":
        return row_profile in {"", "default"}
    return row_profile == active_profile


def _profile_owns_local_row(row: dict[str, Any], active_profile: str) -> bool:
    row_profile = str(row.get("profile_name") or "").strip()
    return row_profile in {"", active_profile}


def resolve_profile_session(
    target: str,
    *,
    active_profile: str,
    local_db: Any,
    root_db: Any,
) -> Optional[ResolvedSessionRef]:
    """Resolve ``target`` without crossing a profile ownership boundary."""

    active_profile = str(active_profile or "default").strip() or "default"
    local_id, local_row = _resolve_in_db(local_db, target)
    root_id, root_row = _resolve_in_db(root_db, target)
    root_owned = bool(root_row and _profile_owns_root_row(root_row, active_profile))

    local_owned = bool(
        local_row
        and (
            _profile_owns_local_row(local_row, active_profile)
            if not _same_db(local_db, root_db)
            else _profile_owns_root_row(local_row, active_profile)
        )
    )
    if local_owned:
        assert local_id is not None
        local_handoff_completed = local_row.get("handoff_state") in {
            "transferred",
            "completed",
            "transferred_with_warning",
        }
        if local_handoff_completed and root_owned and root_id == local_id:
            return ResolvedSessionRef(root_id, active_profile, "gateway_root", root_db)
        return ResolvedSessionRef(local_id, active_profile, "profile", local_db)

    if root_owned:
        assert root_id is not None
        return ResolvedSessionRef(root_id, active_profile, "gateway_root", root_db)
    return None


def open_profile_session_dbs(
    *, active_profile: str, current_db: Any = None
) -> tuple[Any, Any, list[Any]]:
    """Open the active local and canonical root stores, reusing ``current_db``."""
    from hermes_cli.profiles import _get_default_hermes_home, get_profile_dir
    from hermes_state import SessionDB

    active_profile = str(active_profile or "default").strip() or "default"
    local_path = Path(get_profile_dir(active_profile)) / "state.db"
    root_path = Path(_get_default_hermes_home()) / "state.db"
    opened: list[Any] = []

    def _db_for(path: Path):
        if current_db is not None:
            try:
                if Path(current_db.db_path).resolve() == path.resolve():
                    return current_db
            except Exception:
                pass
        db = SessionDB(db_path=path)
        opened.append(db)
        return db

    try:
        local_db = _db_for(local_path)
        root_db = (
            local_db
            if local_path.resolve() == root_path.resolve()
            else _db_for(root_path)
        )
        return local_db, root_db, opened
    except Exception:
        for db in opened:
            close = getattr(db, "close", None)
            if callable(close):
                close()
        raise


def resolve_active_profile_session(
    target: str,
    *,
    active_profile: Optional[str] = None,
    current_db: Any = None,
) -> Optional[ResolvedSessionRef]:
    """Resolve against the current profile and retain only the owning handle."""
    if active_profile is None:
        from hermes_cli.profiles import get_active_profile_name

        active_profile = get_active_profile_name() or "default"
    if current_db is not None and not isinstance(
        getattr(current_db, "db_path", None), (str, Path)
    ):
        session_id, row = _resolve_in_db(current_db, target)
        if row and session_id:
            return ResolvedSessionRef(
                session_id=session_id,
                profile_name=active_profile,
                storage_scope="profile",
                db=current_db,
            )
        return None
    local_db, root_db, opened = open_profile_session_dbs(
        active_profile=active_profile, current_db=current_db
    )
    selected = None
    try:
        resolved = resolve_profile_session(
            target,
            active_profile=active_profile,
            local_db=local_db,
            root_db=root_db,
        )
        selected = resolved.db if resolved is not None else None
        if resolved is None:
            return None
        return ResolvedSessionRef(
            session_id=resolved.session_id,
            profile_name=resolved.profile_name,
            storage_scope=resolved.storage_scope,
            db=resolved.db,
            owns_db=any(db is resolved.db for db in opened),
        )
    finally:
        for db in opened:
            if db is not selected:
                close = getattr(db, "close", None)
                if callable(close):
                    close()


def canonical_root_owns_session(session_id: str, *, active_profile: str) -> bool:
    """Return whether the canonical root contains this profile-owned session."""
    from hermes_cli.profiles import _get_default_hermes_home
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(_get_default_hermes_home()) / "state.db")
    try:
        row = db.get_session(session_id)
        return bool(row and _profile_owns_root_row(row, active_profile))
    finally:
        db.close()


def merge_profile_session_rows(
    *,
    active_profile: str,
    local_db: Any,
    root_db: Any,
    load_rows: Callable[[Any], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge listings without exposing another profile's canonical rows."""
    local_rows = list(load_rows(local_db) or [])
    if _same_db(local_db, root_db):
        return [
            row for row in local_rows
            if _profile_owns_root_row(row, active_profile)
        ]
    local_rows = [
        row for row in local_rows
        if _profile_owns_local_row(row, active_profile)
    ]
    root_rows = [
        row for row in (load_rows(root_db) or [])
        if _profile_owns_root_row(row, active_profile)
    ]
    merged: dict[str, dict[str, Any]] = {
        str(row.get("id")): dict(row) for row in local_rows if row.get("id")
    }
    for row in root_rows:
        session_id = str(row.get("id") or "")
        if not session_id:
            continue
        local = merged.get(session_id)
        local_state = None
        if local is not None:
            try:
                local_state = (local_db.get_session(session_id) or {}).get("handoff_state")
            except Exception:
                pass
        if local is None or local_state in {
            "transferred",
            "completed",
            "transferred_with_warning",
        }:
            merged[session_id] = dict(row)
    return sorted(
        merged.values(),
        key=lambda row: float(
            row.get("last_active") or row.get("started_at") or 0
        ),
        reverse=True,
    )
