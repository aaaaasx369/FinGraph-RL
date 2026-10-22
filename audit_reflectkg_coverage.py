#!/usr/bin/env python3
"""Read-only audit of SEC-derived relations and missing relation coverage."""
import argparse, collections, csv, json, re
from pathlib import Path

RELATIONS = ('SUPPLIES_TO', 'CUSTOMER_OF', 'COMPETES_WITH', 'SAME_SECTOR')
DEFAULT_FILES = [
    'kg/data/edges_kg_v1.csv', 'kg/data/edges_v2_full.csv',
    'kg/data/edges_v3_full.csv', 'kg/data/edges_v4_full.csv',
    'kg/data/same_sector_edges.csv',
    'kg/output/finreflect_reimplementation/reflection.jsonl',
]
PATTERNS = {
 'supply_candidate': re.compile(r'\b(suppl(?:y|ies|ier|iers|ied)|components?|manufactur(?:e|er|ers|ing)|sourc(?:e|ing)|vendors?|procure(?:ment)?)\b',re.I),
 'customer_candidate': re.compile(r'\b(customers?|clients?|purchase(?:s|d)?|sales to|sell(?:s|ing)? to|major customer|concentration of revenue)\b',re.I),
 'competition_candidate': re.compile(r'\b(compet(?:e|es|ing|ition|itor|itors)|rivals?)\b',re.I),
}

def read_rows(path):
    with path.open(encoding='utf-8-sig', errors='replace') as f:
        if path.suffix == '.jsonl':
            for ln, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                    if isinstance(row, dict): yield row
                except json.JSONDecodeError: continue
        else:
            for row in csv.DictReader(f): yield row

def get_value(r, *names):
    for n in names:
        v = r.get(n)
        if v is not None and str(v).strip(): return str(v).strip()
    return ''

def summarize(path):
    counts, accepted, tickers, years, statuses = (collections.Counter() for _ in range(5))
    examples = {}
    total = 0
    unknown_date = 0
    for row in read_rows(path):
        total += 1
        relation = get_value(row,'relation','predicate').upper()
        status = get_value(row, 'validation_status') or 'UNSPECIFIED'
        ticker = get_value(row, 'filing_ticker', 'ticker', 'source_ticker') or 'UNKNOWN'
        date = get_value(row, 'published_at','filing_date','available_at')
        year = date[:4] if re.match(r'^\d{4}-\d{2}-\d{2}',date) else 'UNKNOWN'
        if year == 'UNKNOWN': unknown_date += 1
        counts[relation or 'MISSING'] += 1
        statuses[status] += 1
        if status.upper() in ('ACCEPT','ACCEPTED','VALID','VERIFIED') or status == 'UNSPECIFIED':
            accepted[relation or 'MISSING'] += 1
        tickers[(ticker,relation or 'MISSING')] += 1
        years[(year,relation or 'MISSING')] += 1
        if relation not in examples:
            examples[relation] = {k:row.get(k) for k in ('source','target','subject','object','relation','filing_ticker','ticker','filing_date','published_at','doc_id','chunk_id','evidence','validation_status') if k in row}
    return {'file':str(path),'rows':total,'relation_counts':dict(counts),
            'accepted_or_unlabeled_counts':dict(accepted),'validation_status_counts':dict(statuses),
            'by_filing_ticker':[{ 'ticker': k[0], 'relation':k[1], 'count':v} for k,v in sorted(tickers.items())],
            'by_year':[{ 'year': k[0], 'relation':k[1], 'count':v} for k,v in sorted(years.items())],
            'unknown_date_rows':unknown_date,'examples':examples}

def scan_chunks(path, limit):
    if not path.exists(): return {'exists':False}
    counts = collections.Counter(); samples = {x:[] for x in PATTERNS}
    scanned = 0
    for r in read_rows(path):
        scanned += 1
        txt = get_value(r,'text')
        for name, pat in PATTERNS.items():
            if pat.search(txt):
                counts[name] += 1
                if len(samples[name]) < 3:
                    samples[name].append({'ticker':r.get('ticker'),'published_at':r.get('published_at'),
                        'doc_id':r.get('doc_id'),'chunk_id':r.get('chunk_id'),
                        'snippet':txt[:350]})
        if limit and scanned >= limit: break
    return {'exists':True,'scanned_chunks':scanned,'keyword_hit_counts':dict(counts),'examples':samples,
            'disclaimer':'Keyword hits are retrieval candidates, NOT verified relations or companies.'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='outputs/reflectkg_coverage_audit.json')
    ap.add_argument('--sec', default='rag/data/processed/sec/sec_chunks.jsonl')
    ap.add_argument('--max-sec-chunks', type=int,default=20000)
    ap.add_argument('--files',nargs='*',default=DEFAULT_FILES)
    a = ap.parse_args(); root = Path(a.root)
    results=[]
    for rel in a.files:
        path = root / rel
        if path.exists(): results.append(summarize(path))
        else: results.append({'file':str(path),'exists':False})
    output={'relation_types':list(RELATIONS),'files':results,
            'sec_keyword_candidates':scan_chunks(root/a.sec,a.max_sec_chunks),
            'notes':['Never treat SAME_SECTOR SIC similarity as event-based SEC relation.',
                     'Never merge duplicate experimental extraction runs into a production graph without version/provenance policy.',
                     'No relation in a filing does not imply a relation ended.',
                     'Verify observed filing_date against SEC published_at before backtests.']}
    dest = root/a.out; dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',dest)
    for f in results:
        if not f.get('exists',True): print('MISSING',f['file']); continue
        print('\n',f['file'],'rows=',f['rows'],'unknown_dates=',f['unknown_date_rows'])
        for rel in RELATIONS:
            print('  ',rel,'total=',f['relation_counts'].get(rel,0),'accepted/unlabeled=',f['accepted_or_unlabeled_counts'].get(rel,0))
        print('   other relations:', {k:v for k,v in f['relation_counts'].items() if k not in RELATIONS})
    print('\nSEC candidate keyword hits:',output['sec_keyword_candidates'].get('keyword_hit_counts'))
    print('Keyword hits are NOT extracted company-to-company edges.')
if __name__ == '__main__': main()
