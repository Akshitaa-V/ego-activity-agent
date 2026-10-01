"""A searchable index of predicted moments: what happened, when, and how sure the model is."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sensors import WindowSet
from .synthetic import ACTIVITIES


@dataclass(frozen=True)
class Moment:
    session_id: int
    start: float
    end: float
    activity: str
    confidence: float

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "start_s": round(self.start, 1),
            "end_s": round(self.end, 1),
            "activity": self.activity,
            "confidence": round(self.confidence, 3),
        }


class MomentIndex:
    """Holds one row per window (prediction, confidence, embedding) and answers queries."""

    def __init__(
        self,
        session_ids: np.ndarray,
        t_start: np.ndarray,
        t_end: np.ndarray,
        probs: np.ndarray,
        embeddings: np.ndarray,
        hop_s: float | None = None,
    ) -> None:
        self.session_ids = session_ids
        self.t_start = t_start
        self.t_end = t_end
        self.probs = probs
        self.pred = probs.argmax(axis=1)
        self.conf = probs.max(axis=1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.unit = embeddings / np.clip(norms, 1e-9, None)
        if hop_s is None:  # infer the window hop from one session's start times
            first = np.sort(t_start[session_ids == session_ids[0]])
            hop_s = float(np.diff(first).min()) if first.size > 1 else float(t_end[0] - t_start[0])
        self.hop_s = hop_s

    @classmethod
    def from_windows(cls, ws: WindowSet, probs: np.ndarray, embeddings: np.ndarray) -> MomentIndex:
        return cls(ws.session_ids, ws.t_start, ws.t_end, probs, embeddings)

    def sessions(self) -> list[int]:
        return sorted(int(s) for s in np.unique(self.session_ids))

    def _rows(self, session_id: int) -> np.ndarray:
        rows = np.flatnonzero(self.session_ids == session_id)
        if rows.size == 0:
            raise KeyError(f"unknown session {session_id}; known sessions: {self.sessions()}")
        return rows[np.argsort(self.t_start[rows])]

    def timeline(self, session_id: int) -> list[Moment]:
        """Merge consecutive windows with the same predicted activity into moments."""
        moments: list[Moment] = []
        run: list[int] = []
        for r in self._rows(session_id):
            if run and self.pred[r] != self.pred[run[-1]]:
                moments.append(self._merge(run))
                run = []
            run.append(int(r))
        if run:
            moments.append(self._merge(run))
        return moments

    def _merge(self, rows: list[int]) -> Moment:
        # windows overlap, so a run is reported from its first start to the start of its
        # last window plus one hop; consecutive moments then tile without overlapping
        last = rows[-1]
        return Moment(
            session_id=int(self.session_ids[rows[0]]),
            start=float(self.t_start[rows[0]]),
            end=float(self.t_start[last] + self.hop_s),
            activity=ACTIVITIES[int(self.pred[rows[0]])],
            confidence=float(self.conf[rows].mean()),
        )

    def find(self, activity: str, session_id: int | None = None, min_confidence: float = 0.0) -> list[Moment]:
        if activity not in ACTIVITIES:
            raise ValueError(f"unknown activity '{activity}'; choose from {list(ACTIVITIES)}")
        sessions = [session_id] if session_id is not None else self.sessions()
        return [
            m for s in sessions for m in self.timeline(s) if m.activity == activity and m.confidence >= min_confidence
        ]

    def stats(self, session_id: int) -> dict[str, float]:
        """Seconds spent per predicted activity in one session."""
        totals = dict.fromkeys(ACTIVITIES, 0.0)
        for m in self.timeline(session_id):
            totals[m.activity] += m.end - m.start
        return {k: round(v, 1) for k, v in totals.items() if v > 0}

    def similar(self, session_id: int, time_s: float, k: int = 5) -> list[dict]:
        """Windows in other sessions whose embedding is closest to the one at ``time_s``."""
        rows = self._rows(session_id)
        inside = rows[(self.t_start[rows] <= time_s) & (time_s < self.t_end[rows])]
        if inside.size == 0:
            raise ValueError(f"no window covers {time_s}s in session {session_id}")
        query = self.unit[inside[0]]
        others = np.flatnonzero(self.session_ids != session_id)
        scores = self.unit[others] @ query
        best = others[np.argsort(-scores)[:k]]
        return [
            {
                "session_id": int(self.session_ids[r]),
                "start_s": round(float(self.t_start[r]), 1),
                "activity": ACTIVITIES[int(self.pred[r])],
                "similarity": round(float(self.unit[r] @ query), 3),
            }
            for r in best
        ]
