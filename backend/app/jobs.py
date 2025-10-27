import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional


class QueryJobManager:
    """In-memory job tracker so the frontend can poll progress."""

    def __init__(self, ttl_seconds: int = 900) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds

    def _prune_locked(self) -> None:
        now = time.time()
        expired: List[str] = []
        for job_id, job in self._jobs.items():
            if now - job.get("updated_at", now) > self._ttl:
                expired.append(job_id)
        for job_id in expired:
            self._jobs.pop(job_id, None)

    def create_job(self, initial: Optional[Dict[str, Any]] = None) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        payload = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "created_at": now,
            "updated_at": now,
            "attempts": [],
            "fact_ai_status": None,
            "fact_claims_status": None,
        }
        if initial:
            payload.update(initial)
        with self._lock:
            self._prune_locked()
            self._jobs[job_id] = payload
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            # return a shallow copy so callers can't mutate internal state
            return dict(job)

    def update_job(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(fields)
            job["updated_at"] = time.time()

    def append_attempt(self, job_id: str, attempt: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            attempts = list(job.get("attempts") or [])
            attempts.append(attempt)
            job["attempts"] = attempts
            job["updated_at"] = time.time()


jobs = QueryJobManager()
