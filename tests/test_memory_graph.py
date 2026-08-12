from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from strangeloop.contracts import CognitiveEvent, EventKind, SourceKind
from strangeloop.engine import StrangeloopAgent
from strangeloop.memory import SessionMemoryManager
from strangeloop.memory_graph import MemoryGraph
from strangeloop.store import SQLiteEventStore


class MemoryGraphTests(unittest.TestCase):
    def test_graph_rebuilds_from_events_without_payload_projection(self):
        agent = StrangeloopAgent(session_id="graph-turn")
        try:
            turn = agent.run_turn("Please cite a source before making this claim.")
            graph = agent.memory_graph_status()
            self.assertEqual("derived_from_event_ledger", graph["authority"])
            self.assertGreaterEqual(graph["node_count"], 4)
            self.assertGreater(graph["edge_count"], 0)
            self.assertEqual(agent.event_store.list(agent.session_id)[-1].event_id,
                             graph["ledger_head_event_id"])
            explanation = agent.explain_memory_event(turn.event_ids[1])
            self.assertTrue(any(item["relation"] == "derived_from"
                                for item in explanation["neighbors"]))
            self.assertNotIn("cite a source", repr(explanation))
            self.assertEqual(graph, agent.export_session()["memory_graph"])
        finally:
            agent.event_store.close()

    def test_correction_has_explicit_conflict_and_evidence_edges(self):
        agent = StrangeloopAgent(session_id="graph-correction")
        try:
            turn = agent.run_turn("A preliminary statement.")
            correction = agent.record_correction(turn.event_ids[0], turn.event_ids[-1])
            explanation = agent.explain_memory_event(correction.event_id)
            self.assertTrue(any(item["relation"] == "contradicts"
                                and item["direction"] == "outgoing"
                                for item in explanation["neighbors"]))
            evidence = agent.explain_memory_event(turn.event_ids[-1])
            self.assertTrue(any(item["relation"] == "supports"
                                and item["direction"] == "outgoing"
                                for item in evidence["neighbors"]))
        finally:
            agent.event_store.close()

    def test_sessions_are_isolated_and_unknown_event_kinds_stay_provenance_only(self):
        store = SQLiteEventStore()
        try:
            first = store.append(CognitiveEvent(
                session_id="graph-a", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user",
                payload={"content": "a", "channel": "text"},
            ))
            second = store.append(CognitiveEvent(
                session_id="graph-b", kind=EventKind.OBSERVATION,
                source_kind=SourceKind.USER, source_ref="user",
                payload={"content": "b", "channel": "text"},
            ))
            graph = MemoryGraph(store)
            graph.rebuild("graph-a")
            graph.rebuild("graph-b")
            self.assertEqual([], graph.neighbors("graph-a", second.event_id))
            self.assertEqual([], graph.neighbors("graph-b", first.event_id))
            self.assertEqual(1, graph.summary("graph-a").node_count)
            self.assertEqual(1, graph.summary("graph-b").node_count)
        finally:
            store.close()

    def test_graph_is_removed_with_the_session_container(self):
        with tempfile.TemporaryDirectory() as root:
            manager = SessionMemoryManager(root)
            try:
                store = manager.open_session("graph-purge")
                agent = StrangeloopAgent(session_id="graph-purge", event_store=store)
                agent.run_turn("bounded graph retention")
                path = manager.database_path("graph-purge")
                self.assertTrue(path.exists())
                self.assertGreater(agent.memory_graph_status()["node_count"], 0)
                report = manager.purge_session("graph-purge", confirmed=True)
                self.assertTrue(report.container_deleted)
                self.assertFalse(path.exists())
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
