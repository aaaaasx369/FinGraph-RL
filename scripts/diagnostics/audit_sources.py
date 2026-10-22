#!/usr/bin/env python3
"""Read-only schema/time audit for SEC/FinDKG/ReflectKG on lab Linux."""
import argparse, csv, json, re
from pathlib import Path
from datetime import datetime

FILES = {
    'reflectkg_edges': 'kg/data/edges_v4_full.csv',
    'reflectkg_previous': 'kg/data/edges_v3_full.csv',
    'reflectkg_filing_map': 'kg/output/finreflectkg_dynamic_filing_map.csv',
    'reflectkg_reflections': 'kg/output/finreflect_reimplementation/reflection.jsonl',
    'findkg_edges': 'kg/data/findkg/findkg_full_edges_with_dates.csv',
    'sec_chunks': 'rag/data/processed/sec/sec_chunks.jsonl',
    'v4_output': 'fusion_v4.json',
}
DATE_HINTS = ('date','published','filed','accepted','effective','valid','timestamp','time','as_of','available')
ENTITY_HINTS = ('subject','source','head','ticker','company','object','target','tail','relation','predicate','doc_id','chunk_id','accession')

def inspect_file(path, max_rows=2000):
    p=Path(path)
    if not p.is_file(): return {'exists': False, 'path':str(p)}
    out={'exists':True,'path':str(p),'bytes':p.stat().st_size,'format':p.suffix.lower()}
    try:
        with p.open('r',encoding='utf-8-sig',newline='') as f:
            if p.suffix.lower()=='.csv':
                reader=csv.DictReader(f)
                fields=reader.fieldnames or []
                rows=[]
                for i,row in enumerate(reader):
                    if i>=max_rows: break
                    rows.append(row)
            elif p.suffix.lower()=='.jsonl':
                rows=[]
                for i,line in enumerate(f):
                    if i>=max_rows: break
                    try:
                        r=json.loads(line)
                        if isinstance(r,dict):rows.append(r)
                    except json.JSONDecodeError:pass
                fields=list(dict.fromkeys(k for r in rows for k in r.keys()))
            elif p.suffix.lower()=='.json':
                obj=json.load(f)
                fields=list(obj.keys()) if isinstance(obj,dict) else ['<json_list>']
                rows=[obj] if isinstance(obj,dict) else obj[:max_rows]
            else:
                return {**out,'error':'Unsupported format'}
        out['columns']=fields
        out['sampled_rows']=len(rows)
        out['date_columns']={}
        for k in fields:
            if any(h in k.lower() for h in DATE_HINTS):
                samples=[]
                for row in rows:
                    v=row.get(k) if isinstance(row,dict) else None
                    if v not in ('',None):
                        samples.append(str(v)[:40])
                    if len(samples)>=3: break
                out['date_columns'][k]=samples
        out['entity_columns']=[k for k in fields if any(h in k.lower() for h in ENTITY_HINTS)]
        out['missing_date_candidates']=not bool(out['date_columns'])
        out['date_like_value_counts']={}
        for k in out['date_columns']:
            valid=0; nonempty=0
            for row in rows:
                v=str(row.get(k,'')).strip() if isinstance(row,dict) else ''
                if v:
                    nonempty+=1
                    if re.match(r'^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2})?',v):valid+=1
            out['date_like_value_counts'][k]={'nonempty':nonempty,'ISO_prefix':valid}
    except Exception as exc:
        out['error']=f'{type(exc).__name__}: {exc}'
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='.',help='FinRL-Trading repository root')
    ap.add_argument('--out',default='temporal_source_audit.json')
    ap.add_argument('--sample',type=int,default=2000)
    args=ap.parse_args()
    root=Path(args.root).expanduser().resolve()
    report={'project_root':str(root),'generated_at':datetime.now().astimezone().isoformat(),
            'files':{key:inspect_file(root/path,args.sample) for key,path in FILES.items()},
            'warnings':['No evidence of true streaming updates is implied by date columns alone.',
                        'For point-in-time trading, use earliest independently verified public availability time.',
                        'Do not infer a REMOVE from a missing edge in a new filing.',
                        'FinDKG date may be relation date, not public availability date.',
                        'SEC text and SEC-derived graph edges are correlated sources; deduplicate shared provenance.']}
    out=Path(args.out)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Audit report:',out.resolve())
    for name,r in report['files'].items():
        print(f'[{name}] exists={r["exists"]} rows_sampled={r.get("sampled_rows",0)}')
        print('  columns:',', '.join(r.get('columns',[]))[:600])
        print('  dates:',json.dumps(r.get('date_columns',{}),ensure_ascii=False)[:300])
        if r.get('error'): print('  error:',r['error'])
    print('Read-only audit complete; no project data modified.')

if __name__=='__main__':main()
