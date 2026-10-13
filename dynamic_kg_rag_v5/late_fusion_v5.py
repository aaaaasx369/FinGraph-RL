#!/usr/bin/env python3
"""Time-aware three-source evidence merger; read-only inputs, stdlib only.
SEC lexical fallback is NOT LightRAG: use --run-v4 for actual LightRAG + FinDKG V4 output.
FinDKG edge dates are observations, not proof of edge creation/removal.
"""
import argparse,csv,datetime as dt,json,os,re,subprocess,sys,urllib.request
from collections import defaultdict
from pathlib import Path

DEFAULT_Q='What changed in the company’s financial and operational risks over the last 90 days?'
ALIASES={'AAPL':['aapl','apple'],'MSFT':['msft','microsoft'],'NVDA':['nvda','nvidia'],'AMZN':['amzn','amazon'],'GOOGL':['googl','alphabet','google']}

def date_only(s):
    s=str(s or '').strip()
    if not s:return None
    try:return dt.date.fromisoformat(s[:10])
    except ValueError:return None

def valid(row,t):
    d=date_only(t)
    return bool(d and d<row)

def norm(s):return re.sub(r'\W+',' ',str(s or '').casefold()).strip()

def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)

def toks(s):return set(re.findall(r'[a-zA-Z]{3,}',str(s).lower())) - {'the','and','for','are','with','how','what','over','has','from','this','that'}

def score(text,q,aliases):
    t=toks(text);qq=toks(q)
    return len(t&qq)*3+min(len(t&toks(' '.join(aliases))),2)

def sec_evidence(path,ticker,cut,low,question,max_chunks=8):
    # As-of filtered fallback; NOT LightRAG. Useful for triage and changes of disclosures.
    picks=[];count=0
    with open(path,encoding='utf8') as f:
        for line in f:
            if not line.strip():continue
            x=json.loads(line)
            if str(x.get('ticker','')).upper()!=ticker or not valid(cut,x.get('published_at')):continue
            count+=1
            text=str(x.get('text') or '')
            rank=score(text,question,ALIASES[ticker])
            evidence={'source':'sec','kind':'text_chunk','ticker':ticker,'published_at':str(x.get('published_at')),
                      'doc_id':str(x.get('doc_id') or ''),'chunk_id':str(x.get('chunk_id') or ''),
                      'text':text[:1200],'is_recent':date_only(x.get('published_at'))>=low,'score':rank}
            picks.append(evidence)
    picks.sort(key=lambda x:(x['score'],x['published_at']),reverse=True)
    main=picks[:max_chunks]
    recent=sorted((x for x in picks if x['is_recent']),key=lambda x:(x['score'],x['published_at']),reverse=True)[:max_chunks]
    return {'mode':'temporal_lexical_fallback_not_lightrag','visible_chunks':count,'snapshot':main,'changes':recent}

def resolve_fin_id(raw, mappings):
    return mappings.get(str(raw),str(raw))

def load_mapping(mapping_dir):
    # No guessing mapping file schemas: maps only explicit id,name pairs found in CSV files.
    if not mapping_dir:return {}
    lookup={}
    p=Path(mapping_dir)
    if not p.exists():return {}
    for file in p.glob('*.csv'):
        try:
            rows=read_csv(file)
            for r in rows:
                k=next((r[x] for x in ('entity_id','id','node_id') if r.get(x)),None)
                v=next((r[x] for x in ('entity','name','label','node_name') if r.get(x)),None)
                if k and v:lookup[str(k)]=str(v)
        except (OSError,UnicodeError,csv.Error):continue
    return lookup

def findkg_evidence(path,ticker,cut,low,question,k=14,mapping_dir=None):
    # Dates may mean observation dates, not availability; do not infer edge expiry.
    mapping=load_mapping(mapping_dir)
    names=set(map(norm,ALIASES[ticker])); edges=[]; dates=set(); eligible=0
    for r in read_csv(path):
        day=date_only(r.get('date'))
        if not day or day>=cut:continue
        eligible+=1;dates.add(day.isoformat())
        a=resolve_fin_id(r.get('subject_id'),mapping);b=resolve_fin_id(r.get('object_id'),mapping)
        relation=resolve_fin_id(r.get('relation_id'),mapping)
        edge={'source':'findkg','kind':'dated_edge_observation','subject':a,'relation':relation,'object':b,
              'subject_id':r.get('subject_id'),'relation_id':r.get('relation_id'),'object_id':r.get('object_id'),
              'published_at':day.isoformat(),'date_semantics':'unverified_observation_date',
              'mapping_resolved':(a!=str(r.get('subject_id')) or b!=str(r.get('object_id'))),
              'is_recent':day>=low,'score':score(a+' '+b+' '+relation,question,ALIASES[ticker])}
        edges.append(edge)
    # Restrict focal graph to ticker names if known. If IDs unresolved, refuse to invent focal edges.
    focal=[x for x in edges if norm(x['subject']) in names or norm(x['object']) in names]
    known={norm(y) for x in focal for y in (x['subject'],x['object'])}
    two=[x for x in edges if norm(x['subject']) in known or norm(x['object']) in known]
    seen=set(); relevant=[]
    for x in sorted(focal+two,key=lambda z:(z['score'],z['published_at']),reverse=True):
        key=(x['subject_id'],x['relation_id'],x['object_id'])
        if key not in seen:seen.add(key);relevant.append(x)
    recent=[x for x in relevant if x['is_recent']]
    return {'mode':'dated_graph_observations','eligible_rows':eligible,'eligible_dates':len(dates),
            'focal_matched':len(focal),'mapping_entries':len(mapping),
            'warning':('No ticker node resolved: supply --mapping-dir with entity IDs to names' if not focal else
                       'Dates treated as observations, not confirmed creation/removal events'),
            'snapshot':relevant[:k],'changes':recent[:k]}

def reflect_evidence(path,ticker,cut,low,question,k=14,filing_map=None):
    # Prefer publication date from filing map/doc only where join is verifiable.
    candidates=[];unsupported=0;discarded=0
    for r in read_csv(path):
        if str(r.get('filing_ticker','')).upper()!=ticker:continue
        if str(r.get('validation_status','')).upper() not in ('','ACCEPT','ACCEPTED','VERIFIED'):
            discarded+=1;continue
        raw=r.get('filing_date'); date=date_only(raw)
        if not date or date>=cut:unsupported+=1;continue
        a=str(r.get('source_name') or r.get('source') or '')
        b=str(r.get('target_name') or r.get('target') or '')
        evidence={'source':'reflectkg','kind':'sec_extracted_relation','subject':a,'relation':str(r.get('relation') or ''),
                  'object':b,'published_at':date.isoformat(),'date_semantics':'filing_date_unverified_publication',
                  'doc_id':str(r.get('doc_id') or ''),'chunk_id':str(r.get('chunk_id') or ''),
                  'text':str(r.get('evidence') or '')[:700], 'confidence':r.get('confidence'),
                  'is_recent':date>=low,'score':score(a+' '+b+' '+str(r.get('evidence') or ''),question,ALIASES[ticker])}
        candidates.append(evidence)
    candidates.sort(key=lambda x:(x['score'],x['published_at']),reverse=True)
    # No absence-based REMOVE. Recent = newly observed disclosures, not necessarily new real-world relations.
    recent=[x for x in candidates if x['is_recent']]
    return {'mode':'dated_sec_relation_evidence','eligible_edges':len(candidates),'excluded_nonaccepted':discarded,
            'unsupported_dates':unsupported,'snapshot':candidates[:k],'changes':recent[:k],
            'warning':'filing_date used conservatively; verify SEC published_at join before formal backtest'}

def v4_execute(args):
    out=Path(args.v4_output or 'outputs/v4_for_v5.json');out.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,args.v4_script,'query','--ticker',args.ticker,'--as-of',args.as_of,'--llm',args.llm,'--out',str(out)]
    if args.query:cmd+=['--query',args.query]
    subprocess.run(cmd,check=True)
    if not out.exists():raise RuntimeError('V4 command returned but output missing: '+str(out))
    return out

def compact(items,limit=8):
    return [{k:v for k,v in item.items() if k not in ('score',)} for item in items[:limit]]

def qwen(prompt,url,model):
    body=json.dumps({'model':model,'prompt':prompt,'stream':False,'format':'json'}).encode()
    req=urllib.request.Request(url.rstrip('/')+'/api/generate',data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as resp:return json.loads(resp.read())['response']

def execute(args):
    cut=date_only(args.as_of)
    if not cut:raise ValueError('--as-of must be YYYY-MM-DD')
    if args.days<=0:raise ValueError('--days must be positive')
    low=cut-dt.timedelta(days=args.days)
    for file in (args.sec,args.graph,args.reflect):
        if not Path(file).exists():raise FileNotFoundError(file)
    sec=sec_evidence(args.sec,args.ticker,cut,low,args.query or DEFAULT_Q,args.top_k)
    fin=findkg_evidence(args.graph,args.ticker,cut,low,args.query or DEFAULT_Q,args.graph_k,args.mapping_dir)
    ref=reflect_evidence(args.reflect,args.ticker,cut,low,args.query or DEFAULT_Q,args.graph_k)
    v4_path=v4_execute(args) if args.run_v4 else (Path(args.v4_output) if args.v4_output else None)
    v4=None
    if v4_path:
        if not v4_path.exists():raise FileNotFoundError(v4_path)
        v4=json.loads(v4_path.read_text(encoding='utf-8'))
    output={'meta':{'ticker':args.ticker,'as_of_exclusive':args.as_of,'lookback_days':args.days,
                    'question':args.query or DEFAULT_Q,'time_policy':'strict date < as_of; day-resolution',
                    'v4_used':v4 is not None},'sec':sec,'findkg':fin,'reflectkg':ref,
            'v4_output':v4,'qwen_output':None}
    # No fake LightRAG claims. When V4 exists it is the authoritative LightRAG+FinDKG pipeline;
    # temporal SEC fallback and dated CSV graph remain explicitly labeled as auxiliary evidence.
    concise={k:{'mode':output[k]['mode'],'snapshot':compact(output[k]['snapshot']),
                'changes':compact(output[k]['changes'])} for k in ('sec','findkg','reflectkg')}
    prompt=('You are an evidence-grounded financial analyst. Analyze only information available strictly before '+args.as_of+
        '. Dates in FinDKG are observations and may NOT mean public availability. ReflectKG filing_date may NOT equal public availability. '
        'Do NOT infer an edge was removed merely because it is absent. Do not double-count SEC and ReflectKG citations with matching doc_id/chunk_id. '
        'Return ONLY JSON with keys impact, risk, confidence, graph_change, event_intensity, reasoning, citations, limitations. '
        'impact in [-1,1]; the four other numeric values in [0,1]. Use null for unsupported numeric features rather than inventing numbers. '
        'You MUST mention missing FinDKG mappings if present.\nQuestion: '+(args.query or DEFAULT_Q)+'\nEvidence JSON:\n'+
        json.dumps({'v4':v4,'temporal_evidence':concise},ensure_ascii=False,default=str)[:args.max_prompt_chars])
    if args.invoke_qwen:
        raw=qwen(prompt,args.ollama_url,args.llm)
        try:output['qwen_output']=json.loads(raw)
        except json.JSONDecodeError:output['qwen_output']={'raw_unparsed':raw}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.save_prompt:Path(args.save_prompt).write_text(prompt,encoding='utf-8')
    print(json.dumps({'out':str(out),'v4_used':v4 is not None,'sec_mode':sec['mode'],
       'sec_snapshot':len(sec['snapshot']),'fin_snapshot':len(fin['snapshot']),
       'fin_focal_matched':fin['focal_matched'],'reflect_snapshot':len(ref['snapshot']),
       'reflect_changes':len(ref['changes']),'qwen_called':args.invoke_qwen,
       'warnings':[fin['warning'],ref['warning']]},ensure_ascii=False,indent=2))
    return output

def parser():
    p=argparse.ArgumentParser(description='V5 time-aware three-way evidence prototype. Does NOT silently impersonate LightRAG.')
    p.add_argument('command',choices=['query'])
    p.add_argument('--ticker',choices=sorted(ALIASES),required=True)
    p.add_argument('--as-of',required=True);p.add_argument('--query')
    p.add_argument('--sec',default='rag/data/processed/sec/sec_chunks.jsonl')
    p.add_argument('--graph',default='kg/data/findkg/findkg_full_edges_with_dates.csv')
    p.add_argument('--reflect',default='kg/data/edges_v4_full.csv')
    p.add_argument('--mapping-dir',help='directory of CSV entity_id,name maps; required for numeric-ID FinDKG relevance')
    p.add_argument('--days',type=int,default=90);p.add_argument('--top-k',type=int,default=8)
    p.add_argument('--graph-k',type=int,default=14)
    p.add_argument('--v4-script',default='late_fusion_v4.py')
    p.add_argument('--v4-output',help='existing V4 JSON; can also be path for --run-v4 output')
    p.add_argument('--run-v4',action='store_true',help='actually invoke existing LightRAG+FinDKG V4; requires V4 index ready')
    p.add_argument('--invoke-qwen',action='store_true',help='send evidence to Ollama')
    p.add_argument('--ollama-url',default='http://140.113.87.164:11434')
    p.add_argument('--llm',default='qwen2.5:3b')
    p.add_argument('--max-prompt-chars',type=int,default=16000)
    p.add_argument('--save-prompt');p.add_argument('--out',default='outputs/v5_result.json')
    return p

if __name__=='__main__':
    try:execute(parser().parse_args())
    except Exception as e:print('ERROR: '+str(e),file=sys.stderr);sys.exit(2)
