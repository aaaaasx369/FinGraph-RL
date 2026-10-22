#!/usr/bin/env python3
"""Point-in-time event KG. Python standard library only."""
import argparse,csv,datetime as dt,hashlib,json,re,sqlite3,sys
from pathlib import Path

KNOWN={'AAPL':'Apple','MSFT':'Microsoft','NVDA':'NVIDIA','AMZN':'Amazon','GOOGL':'Alphabet'}
RELATIONS=('SUPPLIES_TO','CUSTOMER_OF','COMPETES_WITH','SAME_SECTOR','DEPENDS_ON','SUBJECT_TO','IMPACTED_BY')

def dateval(s):
    s=str(s or '').strip()
    if not s: return ''
    try:
        v=dt.datetime.fromisoformat(s.replace('Z','+00:00'))
        return v.date().isoformat()
    except ValueError:
        raise ValueError('Invalid ISO date: '+s)

def dbconnect(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(path)
    con.row_factory=sqlite3.Row
    con.executescript('''
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS filings (
      doc_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, published_at TEXT NOT NULL,
      chunk_count INTEGER NOT NULL DEFAULT 0, extracted_count INTEGER NOT NULL DEFAULT 0,
      extraction_status TEXT NOT NULL DEFAULT 'unprocessed');
    CREATE TABLE IF NOT EXISTS evidence (
      evidence_id TEXT PRIMARY KEY, subject TEXT NOT NULL, relation TEXT NOT NULL,
      object TEXT NOT NULL, doc_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
      published_at TEXT NOT NULL, confidence REAL NOT NULL, text_snippet TEXT,
      extractor TEXT NOT NULL, action TEXT NOT NULL DEFAULT 'ASSERT',
      FOREIGN KEY(doc_id) REFERENCES filings(doc_id));
    CREATE INDEX IF NOT EXISTS ev_date ON evidence(published_at);
    CREATE INDEX IF NOT EXISTS ev_subj ON evidence(subject);
    CREATE TABLE IF NOT EXISTS events (
      event_id TEXT PRIMARY KEY, subject TEXT NOT NULL, relation TEXT NOT NULL,
      object TEXT NOT NULL, event_type TEXT NOT NULL CHECK(event_type IN ('ADD','REASSERT','REMOVE')),
      published_at TEXT NOT NULL, evidence_id TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS events_time ON events(published_at);
    ''')
    return con

def stable(*parts): return hashlib.sha256('\x1f'.join(str(x) for x in parts).encode()).hexdigest()[:24]

def ingest_sec(con,src,as_of=None):
    docs={}; counts={'chunks':0,'filings':0}
    with open(src,encoding='utf8') as f:
      for line in f:
        if not line.strip(): continue
        v=json.loads(line); ts=dateval(v.get('published_at')); ticker=str(v.get('ticker','')).upper().strip()
        if not ts or not ticker: continue
        if as_of and ts>as_of: continue
        doc=str(v.get('doc_id') or v.get('source') or v.get('chunk_id') or '').strip()
        if not doc: continue
        docs[(doc,ticker,ts)]=docs.get((doc,ticker,ts),0)+1;counts['chunks']+=1
    with con:
      for (doc,ticker,ts),n in docs.items():
        # Never erase extraction status or imported evidence when registering a filing.
        con.execute('''INSERT INTO filings(doc_id,ticker,published_at,chunk_count) VALUES (?,?,?,?)
        ON CONFLICT(doc_id) DO UPDATE SET chunk_count=excluded.chunk_count''',(doc,ticker,ts,n))
    counts['filings']=len(docs)
    return counts

def rows_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
      for row in csv.DictReader(f): yield row

def import_edges(con,path,strict=True):
    """Import VERIFIED extracted edges, never infer relation validity from absence."""
    num=0; skipped=0
    with con:
      for r in rows_csv(path):
        subject=(r.get('subject') or r.get('subject_name') or '').strip()
        relation=(r.get('relation') or r.get('predicate') or '').strip().upper()
        obj=(r.get('object') or r.get('object_name') or '').strip()
        ts=dateval(r.get('published_at') or r.get('available_at'))
        doc=(r.get('doc_id') or r.get('source_doc_id') or '').strip()
        chunk=(r.get('chunk_id') or r.get('source_chunk_id') or doc).strip()
        action=(r.get('action') or 'ASSERT').upper().strip()
        conf=float(r.get('confidence') or 1.0)
        if not all((subject,relation,obj,ts,doc)) or action not in ('ASSERT','REMOVE') or not 0<=conf<=1:
          skipped+=1
          if strict: raise ValueError('Invalid relation row: '+str(r))
          continue
        # Filing can come from edge import without prior sec import.
        con.execute('INSERT OR IGNORE INTO filings(doc_id,ticker,published_at,chunk_count,extraction_status) VALUES (?,?,?,?,?)',
                    (doc,subject,ts,0,'edge_import'))
        eid=stable(subject,relation,obj,doc,chunk,action)
        con.execute('''INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                    (eid,subject,relation,obj,doc,chunk,ts,conf,
                     (r.get('text_snippet') or r.get('evidence') or '')[:800],
                     (r.get('extractor') or 'imported'),action))
        num+=1
      con.execute('''UPDATE filings SET extracted_count=(SELECT COUNT(*) FROM evidence WHERE evidence.doc_id=filings.doc_id),
                   extraction_status=CASE WHEN EXISTS(SELECT 1 FROM evidence WHERE evidence.doc_id=filings.doc_id)
                   THEN 'extracted' ELSE extraction_status END''')
    rebuild_events(con)
    return {'imported_rows':num,'skipped':skipped}

def rebuild_events(con):
    # Recalculate from full evidence history, deterministic even for out-of-order imports.
    with con:
      con.execute('DELETE FROM events')
      latest={}
      records=con.execute('SELECT * FROM evidence ORDER BY published_at,doc_id,evidence_id').fetchall()
      for e in records:
        key=(e['subject'],e['relation'],e['object'])
        live=latest.get(key,False)
        typ=('REMOVE' if live else None) if e['action']=='REMOVE' else ('REASSERT' if live else 'ADD')
        if typ is None: continue
        latest[key]=typ!='REMOVE'
        con.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',
         (stable(e['evidence_id'],typ),*key,typ,e['published_at'],e['evidence_id']))

def snapshot(con,as_of,subject=None,hops=2):
    # As-of exclusive: for an open-of-day trading decision, this prevents same-day leakage.
    events=con.execute('SELECT * FROM events WHERE published_at < ? ORDER BY published_at,event_id',(as_of,)).fetchall()
    active={};last={}
    for x in events:
      key=(x['subject'],x['relation'],x['object'])
      if x['event_type']=='REMOVE': active.pop(key,None)
      else: active[key]=dict(x)
      last[key]=dict(x)
    if not subject: return list(active.values())
    seen={subject.lower()}; chosen=[]
    for _ in range(hops):
      frontier=[]
      for k,v in active.items():
        a,b=k[0].lower(),k[2].lower()
        if a in seen or b in seen:
          if k not in {tuple(z['subject_relation_object']) for z in chosen}:
            chosen.append({**v,'subject_relation_object':k})
          frontier.extend([a,b])
      new=set(frontier)-seen
      if not new: break
      seen.update(new)
    return [{k:v for k,v in x.items() if k!='subject_relation_object'} for x in chosen]

def changes(con,as_of,days=90,subject=None):
    low=(dt.date.fromisoformat(as_of)-dt.timedelta(days=days)).isoformat()
    rows=con.execute('SELECT * FROM events WHERE published_at >= ? AND published_at < ? ORDER BY published_at,event_id',(low,as_of)).fetchall()
    if subject: rows=[r for r in rows if subject.lower() in (r['subject'].lower(),r['object'].lower())]
    return [dict(r) for r in rows]

def evidence_for(con,records):
    ret=[]
    for r in records:
      ev=con.execute('SELECT * FROM evidence WHERE evidence_id=?',(r['evidence_id'],)).fetchone()
      ret.append({**r,'source':dict(ev) if ev else None})
    return ret

def cli():
    p=argparse.ArgumentParser(description='Dynamic SEC KG time-aware event store (stdlib only).')
    p.add_argument('--db',default='kg/data/dynamic/reflectkg.sqlite')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('init-sec');a.add_argument('--sec',default='rag/data/processed/sec/sec_chunks.jsonl');a.add_argument('--as-of')
    a=sub.add_parser('import-edges');a.add_argument('--edges',required=True);a.add_argument('--skip-invalid',action='store_true')
    a=sub.add_parser('snapshot');a.add_argument('--as-of',required=True);a.add_argument('--ticker');a.add_argument('--hops',type=int,default=2)
    a=sub.add_parser('changes');a.add_argument('--as-of',required=True);a.add_argument('--ticker');a.add_argument('--days',type=int,default=90)
    a=sub.add_parser('stats')
    args=p.parse_args();con=dbconnect(args.db)
    if args.command=='init-sec': result=ingest_sec(con,args.sec,dateval(args.as_of) if args.as_of else None)
    elif args.command=='import-edges': result=import_edges(con,args.edges,not args.skip_invalid)
    elif args.command=='snapshot':result={'as_of_exclusive':args.as_of,'edges':evidence_for(con,snapshot(con,dateval(args.as_of),args.ticker,args.hops))}
    elif args.command=='changes':result={'as_of_exclusive':args.as_of,'events':evidence_for(con,changes(con,dateval(args.as_of),args.days,args.ticker))}
    else: result={t:con.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in ('filings','evidence','events')}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': cli()
