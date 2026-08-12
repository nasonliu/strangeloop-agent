"""A rebuildable, provenance-only graph projection of one event session.

The SQLite event ledger remains the sole authority.  This module never appends
events, approves memory, or interprets text.  It projects only opaque event
identifiers, fixed relation labels, event kinds, and digests already present
in the ledger, so the graph can be deleted with the session container and
deterministically rebuilt from its hash-chained records.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, List, Sequence, Tuple

from .contracts import CognitiveEvent, EventKind
from .store import SQLiteEventStore, canonical_json


GRAPH_VERSION = "memory_graph_v1"
_MAX_NEIGHBORS = 64


@dataclass(frozen=True)
class MemoryGraphSummary:
    """Public aggregate status for a non-authoritative graph projection."""

    session_id: str
    graph_version: str
    node_count: int
    edge_count: int
    ledger_head_event_id: str
    ledger_head_sequence: int
    rebuilt_from_event_count: int

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "graph_version": self.graph_version,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "ledger_head_event_id": self.ledger_head_event_id,
            "ledger_head_sequence": self.ledger_head_sequence,
            "rebuilt_from_event_count": self.rebuilt_from_event_count,
            "authority": "derived_from_event_ledger",
            "limitations": [
                "Contains opaque identifiers and fixed relation labels only.",
                "Cannot approve memory, alter rewards, or authorize tools or lifecycle actions.",
                "Is removed with the SQLite session container and can be rebuilt from the ledger.",
            ],
        }


class MemoryGraph:
    """Session-local graph index reconstructed entirely from ledger events."""

    def __init__(self, event_store: SQLiteEventStore) -> None:
        self.event_store = event_store
        self.connection = event_store.connection
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS memory_graph_nodes (
                session_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                node_type TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                snapshot_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, node_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_graph_nodes_event
                ON memory_graph_nodes(session_id, source_event_id);
            CREATE TABLE IF NOT EXISTS memory_graph_edges (
                session_id TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                evidence_event_id TEXT NOT NULL,
                PRIMARY KEY (session_id, source_node_id, relation, target_node_id, evidence_event_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_graph_edges_source
                ON memory_graph_edges(session_id, source_node_id);
            CREATE INDEX IF NOT EXISTS idx_memory_graph_edges_target
                ON memory_graph_edges(session_id, target_node_id);
            CREATE TABLE IF NOT EXISTS memory_graph_meta (
                session_id TEXT PRIMARY KEY,
                graph_version TEXT NOT NULL,
                ledger_head_event_id TEXT NOT NULL,
                ledger_head_sequence INTEGER NOT NULL,
                rebuilt_from_event_count INTEGER NOT NULL
            );
        """)

    @staticmethod
    def _event_node_id(event_id: str) -> str:
        return "event:" + event_id

    @staticmethod
    def _seed_revision_node_id(seed_id: str, authority_event_id: str) -> str:
        return "seed:%s:%s" % (seed_id, authority_event_id)

    @staticmethod
    def _digest_event(event: CognitiveEvent) -> str:
        # Payload content stays out of the graph. The digest is sufficient to
        # detect whether a projection was built from the exact event record.
        material = {
            "event_id": event.event_id,
            "kind": event.kind.value,
            "parents": list(event.parent_event_ids),
            "payload": event.payload,
            "created_at": event.created_at,
        }
        return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()

    def rebuild(self, session_id: str) -> MemoryGraphSummary:
        """Replace only this session's derived rows with a ledger projection."""
        events = self.event_store.list(session_id)
        event_ids = {event.event_id for event in events}
        with self.event_store.transaction():
            self.connection.execute("DELETE FROM memory_graph_edges WHERE session_id = ?", (session_id,))
            self.connection.execute("DELETE FROM memory_graph_nodes WHERE session_id = ?", (session_id,))
            for event in events:
                self._insert_event_node(session_id, event)
            for event in events:
                self._insert_event_edges(session_id, event, event_ids)
            head = events[-1] if events else None
            self.connection.execute(
                """INSERT OR REPLACE INTO memory_graph_meta
                   (session_id, graph_version, ledger_head_event_id, ledger_head_sequence,
                    rebuilt_from_event_count) VALUES (?, ?, ?, ?, ?)""",
                (session_id, GRAPH_VERSION, "" if head is None else head.event_id,
                 0 if head is None else head.sequence, len(events)),
            )
        return self.summary(session_id)

    def ensure_current(self, session_id: str) -> MemoryGraphSummary:
        """Rebuild only when the immutable ledger head has advanced."""
        head = self._ledger_head(session_id)
        meta = self.connection.execute(
            """SELECT graph_version, ledger_head_event_id, ledger_head_sequence
                 FROM memory_graph_meta WHERE session_id = ?""", (session_id,)
        ).fetchone()
        if (meta is None or meta["graph_version"] != GRAPH_VERSION
                or meta["ledger_head_event_id"] != ("" if head is None else head[0])
                or int(meta["ledger_head_sequence"]) != (0 if head is None else int(head[1]))):
            return self.rebuild(session_id)
        return self.summary(session_id)

    def _ledger_head(self, session_id: str):
        return self.connection.execute(
            """SELECT event_id, sequence FROM cognitive_events
                 WHERE session_id = ? ORDER BY sequence DESC LIMIT 1""", (session_id,)
        ).fetchone()

    def _insert_event_node(self, session_id: str, event: CognitiveEvent) -> None:
        event_node = self._event_node_id(event.event_id)
        self.connection.execute(
            """INSERT INTO memory_graph_nodes
               (session_id, node_id, node_type, source_event_id, snapshot_digest, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, event_node, "event:" + event.kind.value, event.event_id,
             self._digest_event(event), event.created_at),
        )
        seed = self._seed_snapshot(event)
        if seed is not None:
            seed_id, authority_event_id, digest = seed
            self.connection.execute(
                """INSERT INTO memory_graph_nodes
                   (session_id, node_id, node_type, source_event_id, snapshot_digest, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, self._seed_revision_node_id(seed_id, authority_event_id),
                 "seed_revision", event.event_id, digest, event.created_at),
            )

    @staticmethod
    def _seed_snapshot(event: CognitiveEvent):
        payload = event.payload
        if event.kind == EventKind.SEED_PROPOSED:
            seed = payload.get("seed")
            if isinstance(seed, dict) and isinstance(seed.get("seed_id"), str):
                return seed["seed_id"], event.event_id, hashlib.sha256(
                    canonical_json(seed).encode("utf-8")).hexdigest()
        if event.kind == EventKind.SEED_AUTO_APPLIED:
            seed_id = payload.get("seed_id")
            digest = payload.get("new_seed_digest")
            if isinstance(seed_id, str) and isinstance(digest, str) and len(digest) == 64:
                return seed_id, event.event_id, digest
        if event.kind == EventKind.SEED_APPROVED:
            seed_id = payload.get("seed_id")
            if isinstance(seed_id, str):
                return seed_id, event.event_id, hashlib.sha256(
                    ("legacy-approved:" + seed_id).encode("utf-8")).hexdigest()
        return None

    def _edge(self, session_id: str, source: str, relation: str, target: str,
              evidence_event_id: str) -> None:
        self.connection.execute(
            """INSERT OR IGNORE INTO memory_graph_edges
               (session_id, source_node_id, relation, target_node_id, evidence_event_id)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, source, relation, target, evidence_event_id),
        )

    def _insert_event_edges(self, session_id: str, event: CognitiveEvent,
                            event_ids: Sequence[str]) -> None:
        source = self._event_node_id(event.event_id)
        known = set(event_ids)
        for parent_id in event.parent_event_ids:
            if parent_id in known:
                self._edge(session_id, source, "derived_from", self._event_node_id(parent_id), event.event_id)

        payload = event.payload
        if event.kind == EventKind.CORRECTION:
            target = payload.get("target_event_id")
            evidence = payload.get("counterevidence_event_id")
            if target in known:
                self._edge(session_id, source, "contradicts", self._event_node_id(target), event.event_id)
            if evidence in known:
                self._edge(session_id, self._event_node_id(evidence), "supports", source, event.event_id)

        seed = self._seed_snapshot(event)
        if seed is not None:
            seed_id, authority_event_id, _ = seed
            revision = self._seed_revision_node_id(seed_id, authority_event_id)
            self._edge(session_id, revision, "represented_by", source, event.event_id)
            if event.kind == EventKind.SEED_AUTO_APPLIED:
                base = payload.get("base_event_id")
                if base in known:
                    self._edge(session_id, revision, "supersedes", self._event_node_id(base), event.event_id)
            elif event.kind == EventKind.SEED_APPROVED:
                proposal = payload.get("proposal_event_id")
                if proposal in known:
                    self._edge(session_id, revision, "activates", self._event_node_id(proposal), event.event_id)

        if event.kind == EventKind.EXPERIMENT_EXECUTION_STARTED:
            plan = event.parent_event_ids[0] if event.parent_event_ids else None
            if plan in known:
                self._edge(session_id, source, "executes", self._event_node_id(plan), event.event_id)
        elif event.kind == EventKind.EXPERIMENT_RESULT:
            start = event.parent_event_ids[0] if event.parent_event_ids else None
            if start in known:
                self._edge(session_id, source, "result_of", self._event_node_id(start), event.event_id)
        elif event.kind == EventKind.FRONTIER_EVIDENCE_OBSERVATION:
            for parent_id in event.parent_event_ids:
                if parent_id in known:
                    self._edge(session_id, source, "verified_by", self._event_node_id(parent_id), event.event_id)

    def summary(self, session_id: str) -> MemoryGraphSummary:
        meta = self.connection.execute(
            """SELECT graph_version, ledger_head_event_id, ledger_head_sequence,
                      rebuilt_from_event_count FROM memory_graph_meta WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        node_count = self.connection.execute(
            "SELECT COUNT(*) FROM memory_graph_nodes WHERE session_id = ?", (session_id,)).fetchone()[0]
        edge_count = self.connection.execute(
            "SELECT COUNT(*) FROM memory_graph_edges WHERE session_id = ?", (session_id,)).fetchone()[0]
        return MemoryGraphSummary(
            session_id=session_id, graph_version=GRAPH_VERSION, node_count=int(node_count),
            edge_count=int(edge_count), ledger_head_event_id="" if meta is None else meta["ledger_head_event_id"],
            ledger_head_sequence=0 if meta is None else int(meta["ledger_head_sequence"]),
            rebuilt_from_event_count=0 if meta is None else int(meta["rebuilt_from_event_count"]),
        )

    def neighbors(self, session_id: str, event_id: str) -> List[dict]:
        """Return a bounded, payload-free one-hop provenance explanation."""
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be a non-empty identifier")
        node = self._event_node_id(event_id)
        rows = self.connection.execute(
            """SELECT relation, target_node_id, evidence_event_id, 'outgoing' AS direction
                 FROM memory_graph_edges WHERE session_id = ? AND source_node_id = ?
               UNION ALL
               SELECT relation, source_node_id, evidence_event_id, 'incoming' AS direction
                 FROM memory_graph_edges WHERE session_id = ? AND target_node_id = ?
               ORDER BY direction, relation, target_node_id LIMIT ?""",
            (session_id, node, session_id, node, _MAX_NEIGHBORS),
        ).fetchall()
        return [{"direction": row["direction"], "relation": row["relation"],
                 "node_id": row["target_node_id"], "evidence_event_id": row["evidence_event_id"]}
                for row in rows]
