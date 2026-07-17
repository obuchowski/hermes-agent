"""Running-process checkpoint persistence and PID-safe recovery."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger("tools.process_registry")


class ProcessCheckpointMixin:
    def _detached_host_fate(self, pid: Optional[int], expected_start: Optional[int]) -> str:
        """Subclass supplies the live PID decision. See ProcessRegistry."""
        raise NotImplementedError

    # ----- Checkpoint (crash recovery) -----

    def _write_checkpoint(self, extra_entries: Optional[List[Dict[str, Any]]] = None):
        """Atomically persist live jobs and durable completion tombstones."""
        from tools.process_registry import _checkpoint_path, _CHECKPOINT_FIELDS

        with self._checkpoint_lock:
            try:
                with self._lock:
                    sessions = [s for s in self._running.values() if not s.exited]
                    sessions += [
                        s for s in self._finished.values()
                        if s.notify_on_complete
                        and s.notification_state in {"pending", "delivering", "delivered"}
                    ]
                    entries = []
                    for s in sessions:
                        if s.host_start_time is None and s.pid_scope == "host" and s.pid:
                            s.host_start_time = self._safe_host_start_time(s.pid)
                        entry = {"session_id": s.id, **{f: getattr(s, f) for f in _CHECKPOINT_FIELDS}}
                        entry["command"] = redact_sensitive_text(s.command, code_file=True)
                        entry["owner_task_id"] = s.owner_task_id or s.task_id
                        if s.exited and s.notify_on_complete:
                            entry["output_buffer"] = s.output_buffer[-2000:]
                        entries.append(entry)
                    if extra_entries:
                        tracked_ids = {item.get("session_id") for item in entries}
                        entries.extend(item for item in extra_entries if item.get("session_id") not in tracked_ids)
                from utils import atomic_json_write
                atomic_json_write(_checkpoint_path(), entries)
                return True
            except Exception as e:
                logger.debug("Failed to write checkpoint file: %s", e, exc_info=True)
                return False

    def recover_from_checkpoint(self) -> int:
        """Recover live processes and durable completion records after restart."""
        from tools.process_registry import (
            ProcessSession, _CHECKPOINT_FIELDS, _checkpoint_path,
            _CHECKPOINT_DEFAULTS, _stop_systemd_unit,
        )

        checkpoint_path = _checkpoint_path()
        if not checkpoint_path.exists():
            return 0
        try:
            entries = json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return 0
        recovered = 0
        unresolved_scope_entries: List[Dict[str, Any]] = []
        for entry in entries:
            with self._lock:
                if entry.get("session_id") in self._running or entry.get("session_id") in self._finished:
                    continue
            launch_state = str(entry.get("launch_state") or "running")
            if launch_state != "running":
                logger.info(
                    "Discarding unconfirmed process launch: %s (state=%s)",
                    entry.get("command", "unknown")[:60], launch_state,
                )

                continue
            pid, pid_scope = entry.get("pid"), entry.get("pid_scope", "host")
            notify_requested = bool(entry.get("notify_on_complete", False))
            notification_state = str(entry.get("notification_state") or "none")
            fate = (self._detached_host_fate(pid, entry.get("host_start_time"))
                    if pid_scope == "host" and pid else "gone")
            # Never signal a recycled PID or invent completion from an unreadable start time.
            if fate == "reused" and not entry.get("exited", False):
                continue
            pid_is_ours = fate == "running"
            if pid_scope == "host" and fate == "gone":
                systemd_unit = entry.get("systemd_unit", "")
                if systemd_unit and not _stop_systemd_unit(systemd_unit):
                    unresolved_scope_entries.append(entry)
                    continue
            recover_as_finished = bool(entry.get("exited", False)) or (
                pid_scope == "host"
                and notify_requested
                and notification_state in {"pending", "delivering"}
                and not pid_is_ours
            )
            fields = {f: entry.get(f, _CHECKPOINT_DEFAULTS[f]) for f in _CHECKPOINT_FIELDS}
            fields.update(
                command=entry.get("command", "unknown"),
                owner_task_id=entry.get("owner_task_id", "") or entry.get("task_id", ""),
                started_at=entry.get("started_at", time.time()),
                launch_state=launch_state,
                notification_state=notification_state,
                notify_on_complete=notify_requested,
            )
            session = ProcessSession(
                id=entry["session_id"], detached=not recover_as_finished,
                output_buffer=entry.get("output_buffer", ""), **fields,
            )
            if recover_as_finished:
                session.exited = True
                session._completion_event.set()
                if session.notification_state == "delivering":
                    session.notification_state = "pending"
                with self._lock:
                    self._finished[session.id] = session
                recovered += 1
                if session.notification_state == "pending":
                    self.pending_watchers.append(self._watcher_for_session(session))
                continue
            if pid_scope != "host":
                logger.info(
                    "Skipping live recovery for non-host process: %s (pid=%s, scope=%s)",
                    session.command[:60], pid, pid_scope,
                )
                continue
            if not pid:
                continue
            if not pid_is_ours:
                continue
            with self._lock:
                self._running[session.id] = session
            recovered += 1
            logger.info("Recovered detached process: %s (pid=%d)", session.command[:60], pid)
            if session.watcher_interval > 0:
                self.pending_watchers.append(self._watcher_for_session(session))
        self._write_checkpoint(extra_entries=unresolved_scope_entries)
        return recovered
