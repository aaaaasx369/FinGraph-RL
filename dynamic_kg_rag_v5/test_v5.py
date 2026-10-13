import csv,json,tempfile,unittest
from argparse import Namespace
from pathlib import Path
from late_fusion_v5 import execute

class TestV5(unittest.TestCase):
 def test_point_in_time_and_no_invented_findkg_links(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);sec=d/'sec.jsonl';fin=d/'fin.csv';ref=d/'ref.csv';out=d/'out.json'
   sec.write_text(''.join(json.dumps({'ticker':'AAPL','published_at':date,'doc_id':date,'chunk_id':date,'text':'Apple supply chain risk'})+'\n' for date in ('2021-03-01','2021-06-01','2021-07-01')))
   with fin.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['subject_id','relation_id','object_id','date']);w.writeheader();w.writerows([{'subject_id':'123','relation_id':'7','object_id':'999','date':'2021-03-02'}])
   with ref.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['filing_ticker','source','target','relation','filing_date','doc_id','chunk_id','validation_status']);w.writeheader();w.writerows([
     {'filing_ticker':'AAPL','source':'Apple','target':'Supplier','relation':'DEPENDS_ON','filing_date':'2021-03-05','doc_id':'d1','chunk_id':'c1','validation_status':'ACCEPT'},
     {'filing_ticker':'AAPL','source':'Apple','target':'Future','relation':'DEPENDS_ON','filing_date':'2021-06-10','doc_id':'d2','chunk_id':'c2','validation_status':'ACCEPT'}])
   a=Namespace(command='query',ticker='AAPL',as_of='2021-06-01',query='supply chain risk',sec=str(sec),graph=str(fin),reflect=str(ref),mapping_dir=None,days=90,top_k=8,graph_k=14,v4_script='none',v4_output=None,run_v4=False,invoke_qwen=False,ollama_url='',llm='',max_prompt_chars=16000,save_prompt=None,out=str(out))
   result=execute(a)
   self.assertEqual(result['sec']['visible_chunks'],1)
   self.assertEqual(len(result['reflectkg']['snapshot']),1)
   self.assertEqual(result['findkg']['focal_matched'],0)
   self.assertEqual(result['meta']['as_of_exclusive'],'2021-06-01')
   self.assertTrue(out.exists())

if __name__=='__main__':unittest.main()
