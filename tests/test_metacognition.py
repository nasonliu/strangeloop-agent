import unittest

from strangeloop.contracts import SourceKind
from strangeloop.metacognition import (
    CheckCode, EvidencePolarity, MetaStatus, MirrorAuditor, MirrorDisposition,
    MirrorRecord, PublicEvidence, SelfStatus,
)


def evidence(event_id, polarity=EvidencePolarity.SUPPORTS, confidence=.9, **changes):
    values = dict(event_id=event_id, event_kind="external_verification",
                  source_kind=SourceKind.EXTERNAL_VERIFIER, polarity=polarity,
                  confidence=confidence)
    values.update(changes)
    return PublicEvidence(**values)


class MirrorAuditorTests(unittest.TestCase):
    def setUp(self):
        self.auditor = MirrorAuditor()

    def test_flat_schema_and_one_bounded_independent_mirror(self):
        record = self.auditor.assess("episode-1", "target-1", "judgment-1", (evidence("e1"),))
        self.assertIsInstance(record, MirrorRecord)
        self.assertEqual(MetaStatus.CONFIRMED, record.meta_status)
        self.assertEqual(MirrorDisposition.PROVISIONAL, record.disposition)
        self.assertLessEqual(record.meta_confidence_cap, record.self_confidence)
        self.assertEqual(set(("mirror_id", "episode_id", "target_event_id", "judgment_event_id",
            "evidence_event_ids", "self_status", "self_confidence", "self_uncertainty",
            "meta_status", "meta_confidence_cap", "check_codes", "disposition",
            "method_version", "public_summary")), set(record.to_payload()))
        self.assertEqual(1, self.auditor.maximum_depth)

    def test_only_public_typed_evidence_is_permitted(self):
        with self.assertRaises(ValueError):
            evidence("e1", source_kind=SourceKind.MODEL)
        with self.assertRaises(ValueError):
            evidence("e1", event_kind="chain_of_thought")
        with self.assertRaises(ValueError):
            self.auditor.assess("episode-1", "target-1", "judgment-1", ())

    def test_conflict_requires_review_and_never_confirms(self):
        record = self.auditor.assess("episode-1", "target-1", "judgment-1", (
            evidence("e1"), evidence("e2", EvidencePolarity.CONTRADICTS, .8)))
        self.assertEqual(SelfStatus.CONFLICTED, record.self_status)
        self.assertEqual(MetaStatus.CONFLICTED, record.meta_status)
        self.assertEqual(MirrorDisposition.REVIEW_REQUIRED, record.disposition)
        self.assertIn(CheckCode.SOURCE_CONFLICT, record.check_codes)

    def test_missing_evidence_abstains_and_meta_never_exceeds_self(self):
        record = self.auditor.assess("episode-1", "target-1", "judgment-1", (
            evidence("e1", EvidencePolarity.NEUTRAL),))
        self.assertEqual(SelfStatus.INSUFFICIENT, record.self_status)
        self.assertEqual(MetaStatus.LIMITED, record.meta_status)
        self.assertEqual(MirrorDisposition.ABSTAIN, record.disposition)
        self.assertLessEqual(record.meta_confidence_cap, record.self_confidence)

    def test_record_rejects_confirmation_for_insufficient_or_unbounded_schema(self):
        with self.assertRaises(ValueError):
            MirrorRecord("episode-1", "target-1", "judgment-1", ("e1",),
                SelfStatus.INSUFFICIENT, .2, __import__("strangeloop.metacognition", fromlist=["SelfUncertainty"]).SelfUncertainty.HIGH,
                MetaStatus.CONFIRMED, .2, (CheckCode.EVIDENCE_MISSING,), MirrorDisposition.ABSTAIN)
        record = self.auditor.assess("episode-1", "target-1", "judgment-1", (evidence("e1"),))
        for forbidden in ("reward", "approval", "capability", "action", "secret", "reasoning"):
            self.assertNotIn(forbidden, record.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
