import csv, tempfile, unittest
from pathlib import Path
from dynamic_kg import dbconnect, import_edges, snapshot, changes, ingest_sec
import json
class DynamicTest(unittest.TestCase):
 def test_temporal_and_zero_edge(self):
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp); db=dbconnect(t/'test.sqlite')
   sec=t/'sec.jsonl'
   with sec.open('w') as f:
    for i,d in enumerate(['2020-04-30','2021-04-29','2021-05-20','2022-04-29']):
     f.write(json.dumps({'ticker':'AAPL','published_at':d,'doc_id':f'D{i}','chunk_id':f'C{i}','text':'...'} )+'\n')
   ingest_sec(db,sec)
   self.assertEqual(db.execute('select count(*) from filings').fetchone()[0],4)
   f=t/'edges.csv'; f.write_text('subject,relation,object,published_at,doc_id,chunk_id,action\nAAPL,DEPENDS_ON,X,2020-04-30,D0,C0,ASSERT\nAAPL,DEPENDS_ON,Y,2021-04-29,D1,C1,ASSERT\nAAPL,DEPENDS_ON,X,2022-04-29,D3,C3,REMOVE\n')
   import_edges(db,f)
   self.assertEqual(len(snapshot(db,'2021-06-01','AAPL')),2)
   self.assertEqual(len(snapshot(db,'2022-05-01','AAPL')),1)
   self.assertEqual(len(changes(db,'2021-04-29',90,'AAPL')),0)
   self.assertEqual(len(changes(db,'2021-06-01',90,'AAPL')),1)
   self.assertEqual(db.execute("select extraction_status from filings where doc_id='D2'").fetchone()[0],'unprocessed')
if __name__=='__main__':unittest.main()
