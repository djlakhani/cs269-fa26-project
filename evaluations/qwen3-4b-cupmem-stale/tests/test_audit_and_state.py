from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from cupmem_stale_eval.audit import AuditCollector
from cupmem_stale_eval.cupmem_bridge import stale_repository_root
from cupmem_stale_eval.io_utils import (
    atomic_write_json_zst,
    atomic_write_jsonl_zst,
    load_json_zst,
    load_jsonl_zst,
)

STALE_ROOT = stale_repository_root()
if str(STALE_ROOT) not in sys.path:
    sys.path.insert(0, str(STALE_ROOT))

from cup_mem.llm_layer.client import LLMClient  # noqa: E402
from cup_mem.memory.models import (  # noqa: E402
    BucketTrackRef,
    ProfileItem,
    SessionChunk,
    SessionDelta,
    StaleSupportLink,
    UnknownCurrent,
)
from cup_mem.pipeline import CupMemEngine  # noqa: E402
from cup_mem.store_layer import ProfileStore  # noqa: E402


class FakeCompletions:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer": "ok"}'))],
            usage=usage,
        )


class FakeOpenAI:
    last_instance = None

    def __init__(self, **_kwargs) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())
        FakeOpenAI.last_instance = self


class AuditAndStateTests(unittest.TestCase):
    def test_compressed_json_round_trips(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_write_json_zst(root / "one.json.zst", {"snowman": "☃"})
            atomic_write_jsonl_zst(root / "many.jsonl.zst", [{"i": 1}, {"i": 2}])
            self.assertEqual(load_json_zst(root / "one.json.zst"), {"snowman": "☃"})
            self.assertEqual(load_jsonl_zst(root / "many.jsonl.zst"), [{"i": 1}, {"i": 2}])

    def test_llm_audit_captures_exact_request_without_credentials(self) -> None:
        events: list[dict] = []
        client = LLMClient(
            model="model",
            api_key="secret-not-for-audit",
            base_url="http://localhost/v1",
            audit_sink=events.append,
            default_request_kwargs={"temperature": 0, "seed": 42},
            request_overrides={"temperature": 0},
            openai_cls=FakeOpenAI,
        )
        parsed = client.call_json(
            "system",
            "user",
            extra_meta={"phase": "answer_composer", "query_label": "dim1_query"},
            extra_request_kwargs={"temperature": 0.7},
        )
        self.assertEqual(parsed, {"answer": "ok"})
        request = FakeOpenAI.last_instance.chat.completions.requests[0]
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["seed"], 42)
        attempt = next(event for event in events if event["event"] == "llm_attempt")
        self.assertEqual(attempt["messages"], request["messages"])
        self.assertEqual(attempt["raw_response"], '{"answer": "ok"}')
        self.assertNotIn("secret-not-for-audit", repr(events))
        self.assertTrue(any(event.get("parsed_json") == parsed for event in events))

        collector = AuditCollector()
        for event in events:
            collector(event)
        enriched_attempt = next(
            event for event in collector.snapshot() if event["event"] == "llm_attempt"
        )
        self.assertEqual(enriched_attempt["json_parse_status"], "parsed")
        self.assertEqual(enriched_attempt["parsed_json"], parsed)

    def test_collector_extracts_and_compresses_answer_composer_request(self) -> None:
        collector = AuditCollector()
        collector(
            {
                "event": "llm_attempt",
                "phase": "answer_composer",
                "query_label": "dim1_query",
                "messages": [{"role": "user", "content": "question"}],
                "request_kwargs": {"temperature": 0},
                "raw_response": "answer",
            }
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl.zst"
            collector.flush(path)
            restored = AuditCollector()
            restored.load(path)
            request = restored.answer_composer_request("dim1_query")
            self.assertEqual(request["messages"][0]["content"], "question")
            self.assertEqual(request["audit_sequence"], 1)

    def test_engine_state_round_trip_includes_all_mutable_stores_and_counters(self) -> None:
        store = ProfileStore()
        active = ProfileItem(
            item_id="pi_00001",
            bucket="personal_details",
            local_track="location",
            value="Lives in Los Angeles",
            status="ACTIVE",
            confidence=0.9,
            first_seen="2025-01-01",
            last_updated="2025-01-02",
            revision_history=[{"old": "Boston"}],
        )
        stale = ProfileItem(
            item_id="pi_00002",
            bucket="personal_details",
            local_track="location",
            value="Lives in Boston",
            status="STALE",
            confidence=0.8,
            first_seen="2024-01-01",
            last_updated="2025-01-02",
        )
        store.active_items[(active.bucket, active.local_track)].append(active)
        store.stale_archive.append(stale)
        store.unknown_current[("work", "role")] = UnknownCurrent(
            bucket="work",
            local_track="role",
            status="UNKNOWN_CURRENT",
            reason="role changed",
            invalidated_by_delta_id="d_00001",
            last_updated="2025-01-02",
        )
        store.stale_support_links.append(
            StaleSupportLink(
                stale_item_id=stale.item_id,
                linking_delta_id="d_00001",
                link_type="SUPPORTS_STALE",
                reason="moved",
                link_session_idx=41,
            )
        )
        store._counter = 9

        engine = CupMemEngine.__new__(CupMemEngine)
        engine.store = store
        engine.chunk_bank = [
            SessionChunk(
                chunk_id="c_00001",
                session_id="s_00007",
                source_turn_ids=["t_0"],
                text="I moved",
                chunk_type="explicit_state",
                candidate_bucket_tracks=[BucketTrackRef("personal_details", "location")],
                temporal_scope="current",
                state_salience=1.0,
                evidence_strength="strong",
            )
        ]
        engine.delta_store = [
            SessionDelta(
                delta_id="d_00001",
                session_id="s_00007",
                session_index=7,
                session_time="2025-01-02",
                bucket="personal_details",
                local_track="location",
                proposed_value="Lives in Los Angeles",
                source_type="explicit",
                evidence_chunk_ids=["c_00001"],
                confidence=0.9,
            )
        ]
        engine._chunk_counter = 4
        engine._delta_counter = 5
        engine._proposal_counter = 6

        checkpoint = engine.export_state()
        restored = CupMemEngine.__new__(CupMemEngine)
        restored.store = ProfileStore()
        restored.restore_state(checkpoint)
        self.assertEqual(restored.export_state(), checkpoint)
        self.assertEqual(restored._next_chunk_id(), "c_00005")
        self.assertEqual(restored._next_delta_id(), "d_00006")
        self.assertEqual(restored._next_proposal_id(), "lip_00007")


if __name__ == "__main__":
    unittest.main()
