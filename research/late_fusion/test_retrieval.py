import json
import tempfile
import unittest
from pathlib import Path

from research.late_fusion.retrieval import (
    fuse_evidence, normalize_lightrag_query_data, retrieve_findkg_csv, retrieve_text_jsonl
)


class LateFusionTests(unittest.TestCase):
    def test_point_in_time_and_two_hops(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "t.jsonl"
            g = Path(d) / "g.csv"
            t.write_text(
                '{"id":"past","ticker":"AAPL","published_at":"2021-05-20","content":"AAPL supply risk"}\n'
                '{"id":"future","ticker":"AAPL","published_at":"2021-06-02","content":"AAPL future"}\n'
                '{"id":"undated","ticker":"AAPL","content":"AAPL undated"}\n'
            )
            g.write_text(
                "id,subject,relation,object,observed_at,available_at\n"
                "a,AAPL,partner,X,2021-05-01,2021-05-02\n"
                "b,X,partner,Y,2021-05-02,2021-05-03\n"
                "c,AAPL,partner,Z,2021-05-01,2021-06-04\n"
            )
            texts = retrieve_text_jsonl(t, "AAPL", "2021-06-01")
            edges = retrieve_findkg_csv(g, "AAPL", "2021-06-01", hops=2)
            self.assertEqual([x.evidence_id for x in texts], ["past"])
            self.assertEqual([x.evidence_id for x in edges], ["a", "b"])
            self.assertEqual(edges[1].hop, 2)
            bundle = fuse_evidence("AAPL", "2021-06-01", texts, edges)
            self.assertEqual(len(bundle["graph_evidence"]), 2)

    def test_lightrag_metadata_required(self):
        data = {"status": "success", "data": {"chunks": [
            {"chunk_id": "a", "content": "past"},
            {"chunk_id": "b", "content": "future"},
            {"chunk_id": "c", "content": "unknown"}]}}
        result = normalize_lightrag_query_data(data, "AAPL", "2021-06-01",
                   {"a": {"published_at": "2021-05-01"},
                    "b": {"published_at": "2021-06-02"}})
        self.assertEqual([x.evidence_id for x in result], ["a"])


if __name__ == "__main__":
    unittest.main()
