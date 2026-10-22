import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from aegis import (
    parse_time,
    retrieve,
    verify,
    poison_document,
    train_policy,
    evaluate,
)


class TestAEGIS(unittest.TestCase):

    def setUp(self):
        self.now = parse_time(
            "2023-01-10T16:00:00Z"
        )
        self.docs = [
            {
                "ticker": "NVDA",
                "id": "SEC-001",
                "time": parse_time(
                    "2022-12-01T16:00:00Z"
                ),
                "text": "supplier supply risk",
                "tokens": {
                    "supplier", "supply", "risk"
                },
            },
            {
                "ticker": "NVDA",
                "id": "SEC-002",
                "time": parse_time(
                    "2023-02-01T16:00:00Z"
                ),
                "text": "supplier supply growth",
                "tokens": {
                    "supplier", "supply", "growth"
                },
            },
        ]
        self.edges = []

    def test_no_future_retrieval(self):
        evidence = retrieve(
            self.docs, "NVDA",
            "supplier", self.now
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(
            evidence[0]["id"], "SEC-001"
        )

    def test_unknown_identity_is_detected(self):
        attacked = poison_document(
            self.docs[0], "identity"
        )
        score, issues = verify(
            attacked, self.now, {"SEC-001"}
        )
        self.assertLess(score, 1)
        self.assertIn(
            "unknown_document_id", issues
        )

    def test_future_document_rejected(self):
        score, issues = verify(
            self.docs[1], self.now,
            {"SEC-001", "SEC-002"}
        )
        self.assertEqual(score, 0)
        self.assertIn(
            "future_document", issues
        )

    def test_content_attack_limitation(self):
        attacked = poison_document(
            self.docs[0], "content"
        )
        score, issues = verify(
            attacked, self.now, {"SEC-001"}
        )
        # Current heuristic cannot detect a
        # tampered text with a preserved ID.
        self.assertEqual(score, 1)
        self.assertEqual(issues, [])

    def test_policy_output(self):
        W = train_policy(
            self.docs, self.edges,
            [self.now],
            epochs=5, seed=42
        )
        self.assertEqual(W.shape, (2, 4))

        rows = evaluate(
            self.docs, self.edges,
            [self.now], W,
            policy="never"
        )
        self.assertTrue(
            all(r["retrieved"] == 0 for r in rows)
        )


if __name__ == "__main__":
    unittest.main()
