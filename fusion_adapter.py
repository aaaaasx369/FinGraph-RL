#!/usr/bin/env python3
"""Adapter to append Dynamic ReflectKG evidence to existing V4 SEC/FinDKG output."""
import argparse,json,urllib.request
from dynamic_kg import dbconnect,snapshot,changes,evidence_for

def dynamic_context(db,ticker,as_of,days=90,hops=2):
    con=dbconnect(db)
    active=evidence_for(con,snapshot(con,as_of,ticker,hops))
    delta=evidence_for(con,changes(con,as_of,days,ticker))
    def line(x):
        s=x.get('source') or {}
        return f"- {x['subject']} --{x['relation']}--> {x['object']} | event={x['event_type']} | available={x['published_at']} | doc={s.get('doc_id','')} | chunk={s.get('chunk_id','')} | confidence={s.get('confidence','')} | evidence={s.get('text_snippet','')[:200]}"
    return '[REFLECTKG_EVIDENCE]\n'+'\n'.join(map(line,active))+ '\n[KG_CHANGES_LAST_'+str(days)+'_DAYS]\n'+'\n'.join(map(line,delta))

def qwen_generate(prompt,base_url,model):
    # Ollama generate endpoint, compatible with remote HTTP Ollama.
    payload=json.dumps({'model':model,'prompt':prompt,'stream':False,'format':'json'}).encode()
    req=urllib.request.Request(base_url.rstrip('/')+'/api/generate',data=payload,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as res:
        return json.loads(res.read())['response']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--db',default='kg/data/dynamic/reflectkg.sqlite')
    p.add_argument('--ticker',required=True);p.add_argument('--as-of',required=True)
    p.add_argument('--question',required=True)
    p.add_argument('--sec-evidence-file',help='Saved SEC evidence text produced by V4')
    p.add_argument('--findkg-evidence-file',help='Saved FinDKG evidence text produced by V4')
    p.add_argument('--days',type=int,default=90)
    p.add_argument('--ollama-url',default='http://140.113.87.164:11434')
    p.add_argument('--model',default='qwen2.5:3b')
    p.add_argument('--invoke-qwen',action='store_true')
    a=p.parse_args()
    def source(path): return open(path,encoding='utf8').read() if path else '[not supplied]'
    ctxt=dynamic_context(a.db,a.ticker,a.as_of,a.days)
    prompt=f'''You are an evidence-grounded financial analyst. Use only statements publicly available before {a.as_of}. Do not invent facts. Distinguish missing evidence from low risk. Return valid JSON with keys impact,risk,confidence,graph_change,event_intensity,reasoning,citations. The numeric values must be between -1 and 1 for impact and between 0 and 1 for others. Cite source doc/chunk IDs when possible.\nTicker: {a.ticker}\nQuestion: {a.question}\n[SEC_TEXT_EVIDENCE]\n{source(a.sec_evidence_file)}\n[FINDKG_EVIDENCE]\n{source(a.findkg_evidence_file)}\n{ctxt}\n'''
    if a.invoke_qwen: print(qwen_generate(prompt,a.ollama_url,a.model))
    else: print(prompt)
if __name__=='__main__':main()
