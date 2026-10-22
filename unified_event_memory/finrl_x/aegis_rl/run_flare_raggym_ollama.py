#!/usr/bin/env python3
"""FinRL-X live FLARE-inspired LLM question generation + multi-step KG/SEC-RAG.
Run from repository root. Strict data checks; NOT a reproduction of FLARE or RAG-Gym.
"""
import argparse, bisect, hashlib, json, math, os, random, re, sqlite3, sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from ollama import Client

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from unified_event_memory.finrl_x.aegis_rl import run_self_questioning_v1 as legacy
from unified_event_memory.finrl_x.aegis_rl import run_portfolio_aware_v2 as v2
TICKERS=['AAPL','AMZN','GOOGL','MSFT','NVDA']
REL=['SUPPLIES_TO','CUSTOMER_OF','COMPETES_WITH','SAME_SECTOR']
SOURCES=['STOP','KG','SEC','BOTH']
COST_BPS={'STOP':0.,'KG':.3,'SEC':.8,'BOTH':1.1}
ALIAS={'AAPL':['aapl','apple'],'AMZN':['amzn','amazon'],'GOOGL':['googl','alphabet','google'],'MSFT':['msft','microsoft'],'NVDA':['nvda','nvidia']}

def stamp(x):
    t=pd.to_datetime(x,utc=True,errors='coerce')
    return pd.Timestamp(t).tz_convert(None) if not pd.isna(t) else pd.NaT

def mandatory(condition,message):
    if not condition: raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

class LLMEngine:
    """Remote Ollama generation + token logprobs + remote semantic embeddings."""
    def __init__(self, name, embed_name, cache_path, device, max_tokens, host):
        self.name = name
        self.embed_name = embed_name
        self.max_tokens = max_tokens
        self.host = host.rstrip('/')
        self.client = Client(host=self.host, timeout=180)
                # Ensure SQLite cache directory exists.
        # Do not alter the selected cache database.
        cache_file = Path(cache_path).expanduser()
        if str(cache_file) != ":memory:":
            cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_file), timeout=120)
        self.db.execute('CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, response TEXT, confidence REAL)')
        self.db.commit()
        try:
            sample = self.client.embed(model=self.embed_name, input='finance retrieval semantic encoder')
            vec = sample.embeddings[0]
            self.dim = len(vec)
            mandatory(self.dim >= 64, f'Unexpected embedding dimensions: {self.dim}')
        except Exception as e:
            raise RuntimeError(f'Ollama embedding model {self.embed_name!r} failed at {self.host}: {e}') from e

    def generate(self, prompt):
        key = hashlib.sha256((self.host+'|'+self.name+'|'+str(self.max_tokens)+'|'+prompt).encode()).hexdigest()
        row = self.db.execute('SELECT response,confidence FROM llm WHERE key=?', (key,)).fetchone()
        if row:
            return row[0], float(row[1])
        try:
            response = self.client.generate(
                model=self.name, prompt=prompt, stream=False,
                logprobs=True, top_logprobs=3,
                options={'num_predict': self.max_tokens, 'temperature': 0},
            )
        except Exception as e:
            raise RuntimeError(f'Ollama generation/logprobs failed for model {self.name}: {e}') from e
        token_data = response.logprobs or []
        mandatory(bool(token_data), 'Ollama did not return token logprobs; FLARE confidence cannot be measured')
        probs = [math.exp(max(-50.0, float(tok.logprob))) for tok in token_data]
        confidence = float(np.mean(probs))
        sentence = (response.response or '').strip()
        mandatory(bool(sentence), 'Ollama generated empty text')
        self.db.execute('INSERT OR REPLACE INTO llm VALUES(?,?,?)', (key, sentence, confidence))
        self.db.commit()
        return sentence, confidence

    @lru_cache(maxsize=8192)
    def embed(self, text):
        response = self.client.embed(model=self.embed_name, input=str(text)[:3000])
        v = np.asarray(response.embeddings[0], dtype=np.float32)
        mandatory(len(v) == self.dim and np.isfinite(v).all(), 'Ollama returned invalid embedding')
        return v / (np.linalg.norm(v) + 1e-9)

class TemporalRetrieval:
    def __init__(self,kg_rows,sec_rows,llm,topk):
        self.llm=llm;self.topk=topk
        self.db={}
        self.times={}
        for source,rows in [('KG',kg_rows),('SEC',sec_rows)]:
            clean=[]
            for r in rows:
                dt=stamp(r.get('available_at'))
                if pd.isna(dt):continue
                z=dict(r);z['available_at']=dt
                clean.append(z)
            clean.sort(key=lambda z:z['available_at'])
            self.db[source]=clean
            self.times[source]=[r['available_at'] for r in clean]
            mandatory(bool(clean),f'Empty dated {source} retrieval index')

    @lru_cache(maxsize=60000)
    def query(self,source,ticker,relation,day_iso,question):
        day=pd.Timestamp(day_iso)
        rows=self.db[source]
        stop=bisect.bisect_left(self.times[source],day)
        qwords=set(re.findall(r'[a-z0-9]+',question.lower()))
        relwords=set(re.findall(r'[a-z]+',relation.lower()))
        candidates=[]
        for r in rows[:stop]:
            text=str(r.get('text',''))
            low=text.lower()
            sec_tic=str(r.get('ticker','')).upper()
            if source=='SEC' and sec_tic and sec_tic!=ticker:continue
            if not (sec_tic==ticker or any(x in low for x in ALIAS[ticker])):continue
            w=set(re.findall(r'[a-z0-9]+',low))
            lex=len(qwords&w)/max(len(qwords),1)
            if source=='KG' and str(r.get('relation','')).upper()==relation:lex+=.5
            age=max(0,(day-r['available_at']).total_seconds()/86400)
            preliminary=lex+.08*math.exp(-age/365)
            candidates.append((preliminary,r))
        candidates.sort(key=lambda p:p[0],reverse=True)
        short=candidates[:max(12,self.topk*8)]
        if not short:return []
        qvec=self.llm.embed(question)
        scored=[]
        for lex,r in short:
            sim=float(np.dot(qvec,self.llm.embed(str(r['text']))))
            scored.append((.4*lex+.6*sim,r))
        scored.sort(key=lambda p:p[0],reverse=True)
        return scored[:self.topk]

    def retrieve(self,source,ticker,relation,day,question):
        vecs=[];audit=[]
        sources=['KG','SEC'] if source=='BOTH' else ([source] if source!='STOP' else [])
        for src in sources:
            hits=self.query(src,ticker,relation,str(day.date()),question)
            if hits:
                scores=np.array([max(0,s) for s,_ in hits]);scores=scores/(scores.sum()+1e-8)
                matrix=np.stack([self.llm.embed(str(r['text'])) for _,r in hits]);vec=(matrix*scores[:,None]).sum(0)
            else:vec=np.zeros(self.llm.dim,dtype=np.float32)
            vecs.append((src,vec))
            for score,r in hits:
                mandatory(r['available_at']<day,'PIT violation: retrieved future evidence')
                audit.append({'source':src,'id':str(r['evidence_id']),'available_at':str(r['available_at']),'score':round(float(score),5),'text_excerpt':str(r['text'])[:220]})
        result=np.zeros((2,self.llm.dim),dtype=np.float32)
        for src,vec in vecs:result[0 if src=='KG' else 1]=vec
        return result,audit

class MultiStepEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,market,base_cols,kgt,retriever,llm,mode,steps,trade_cost,cost_scale,split):
        super().__init__();self.mode=mode;self.steps=steps;self.ret=retriever;self.llm=llm;self.trade_cost=trade_cost;self.cost_scale=cost_scale;self.split=split
        sub=market[market.ticker.isin(TICKERS)].copy()
        good=sub.groupby('date').ticker.nunique();sub=sub[sub.date.isin(good[good==5].index)]
        self.dates=[pd.Timestamp(t) for t in sorted(sub.date.unique())]
        mandatory(len(self.dates)>3,'Too few complete trading days')
        def pivot(c):
            x=sub.pivot(index='date',columns='ticker',values=c).reindex(index=self.dates,columns=TICKERS)
            mandatory(not x.isna().any().any(),f'Missing market feature {c}')
            return x.to_numpy(float)
        self.price=pivot('market_close')
        raw=np.stack([pivot(c) for c in base_cols],axis=-1)
        emb=np.stack([np.stack([kgt[(d,t)] for t in TICKERS]) for d in self.dates])
        self.features=np.concatenate([raw,emb],axis=2).astype(np.float32)
        self.edim=llm.dim
        self.action_space=spaces.Box(low=-1,high=1,shape=(18,),dtype=np.float32)
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.features.shape[1]*self.features.shape[2]+2*self.edim+5+5,),dtype=np.float32)
        self.reset()

    def _question_seed(self):
        d=self.dates[self.day]
        # No historic outcome/future price is supplied in the prompt.
        prompt=(f'Date: {d.date()}. Portfolio stocks: {", ".join(TICKERS)}. '
                'Write one short question about a currently unknown company relationship or SEC risk '
                'that would matter for portfolio allocation. Do not answer it. Question:')
        self.seed_text,self.confidence=self.llm.generate(prompt)

    def _obs(self):
        extras=np.array([float(self.phase),self.used/max(1,self.steps),self.confidence,float(self.last_hits>0),self.remaining_cost],dtype=np.float32)
        return np.concatenate([self.features[self.day].reshape(-1),self.evidence.reshape(-1),self.weights.astype(np.float32),extras]).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.day=0;self.phase=0;self.used=0;self.weights=np.ones(5)/5;self.equity=1e6;self.values=[self.equity];self.returns=[];self.history=[];self.day_queries=[];self.evidence=np.zeros((2,self.edim),dtype=np.float32);self.remaining_cost=0.;self.last_hits=0;self._question_seed()
        return self._obs(),{}

    @staticmethod
    def softmax(x):
        z=np.clip(np.asarray(x,float),-20,20);v=np.exp(z-z.max());return v/v.sum()

    def step(self,action):
        a=np.asarray(action,dtype=np.float32).reshape(-1)
        mandatory(a.shape==(18,),'Action must have 18 entries')
        date=self.dates[self.day]
        if self.phase==0:
            source=SOURCES[int(np.argmax(a[:4]))]
            if self.mode=='NEVER':source='STOP'
            if self.mode=='ALWAYS_BOTH':source='BOTH'
            ticker=TICKERS[int(np.argmax(a[4:9]))]
            relation=REL[int(np.argmax(a[9:13]))]
            if source=='STOP':
                self.phase=1
                return self._obs(),0.,False,False,{'phase':'allocation','stopped':True}
            # Actual LLM query generation inside step, conditioned on RL action and previous evidence.
            summaries=' '.join(q['summary'] for q in self.day_queries[-2:])[:500]
            prompt=(f'Historical decision date {date.date()}. Company {ticker}; relation {relation}. '
                    f'Earlier retrieved evidence: {summaries or "none"}. '
                    'Write a single specific English search query for historically available company relations or SEC filings. '
                    'Do not state facts you cannot verify. Query:')
            question,confidence=self.llm.generate(prompt)
            if not question:question=f'{ticker} {relation} supplier customer competition risk'
            vec,audit=self.ret.retrieve(source,ticker,relation,date,question)
            # Running mean of actual semantic evidence per source, not counts-only 8D.
            for j in range(2):
                if np.linalg.norm(vec[j])>0:self.evidence[j]=.5*self.evidence[j]+.5*vec[j]
            self.last_hits=len(audit);self.confidence=confidence;self.used+=1
            cost=COST_BPS[source]/10000.*self.cost_scale;self.remaining_cost+=cost
            summary=' '.join(x['text_excerpt'] for x in audit[:2])[:400]
            # Post-retrieval reasoning and follow-up question generated by LLM.
            follow,follow_conf=self.llm.generate(f'Date {date.date()}; request: {question}. Verified historical retrieved excerpts: {summary or "NO EVIDENCE"}. Identify one unresolved financial question, without inventing facts. Question:')
            self.day_queries.append({'date':str(date.date()),'step':self.used,'source':source,'ticker':ticker,'relation':relation,'query':question,'generation_token_confidence':confidence,'followup':follow,'followup_token_confidence':follow_conf,'evidence':audit,'summary':summary,'retrieval_cost':cost})
            if self.used>=self.steps or self.mode=='ALWAYS_BOTH':self.phase=1
            return self._obs(),0.,False,False,{'phase':'retrieval','n_evidence':len(audit)}
        w=self.softmax(a[13:18]);returns=self.price[self.day+1]/self.price[self.day]-1
        trans=self.trade_cost*np.abs(w-self.weights).sum();gross=float(w@returns)
        net=max(gross-trans-self.remaining_cost,-.999999);self.equity*=1+net
        after=w*(1+returns);self.weights=after/after.sum() if after.sum()>0 else np.ones(5)/5
        self.returns.append(net);self.values.append(self.equity)
        self.history.append({'date':str(date.date()),'queries':json.dumps(self.day_queries,ensure_ascii=False),'n_queries':len(self.day_queries),'KG_hits':sum(sum(z['source']=='KG' for z in q['evidence']) for q in self.day_queries),'SEC_hits':sum(sum(z['source']=='SEC' for z in q['evidence']) for q in self.day_queries),'total_retrieval_cost':self.remaining_cost,'net_return':net,'equity':self.equity,**{f'weight_{t}':float(w[j]) for j,t in enumerate(TICKERS)}})
        self.day+=1;done=self.day>=len(self.dates)-1
        self.phase=0;self.used=0;self.evidence[:]=0;self.remaining_cost=0.;self.day_queries=[];self.last_hits=0
        if not done:self._question_seed()
        return self._obs(),float(np.log1p(net)),done,False,{'phase':'allocation'}

def build_model(agent,env,seed,learning_rate,device):
    common=dict(seed=seed,learning_rate=learning_rate,device=device,verbose=0)
    if agent=='A2C':return A2C('MlpPolicy',env,n_steps=10,**common)
    if agent=='PPO':return PPO('MlpPolicy',env,n_steps=256,batch_size=64,**common)
    if agent=='DDPG':
        noise=NormalActionNoise(mean=np.zeros(18),sigma=.1*np.ones(18))
        return DDPG('MlpPolicy',env,action_noise=noise,learning_starts=1000,batch_size=128,buffer_size=100000,**common)
    raise ValueError(agent)

def evaluate(model,env):
    obs,_=env.reset();done=False
    while not done:
        a,_=model.predict(obs,deterministic=True)
        obs,_,done,truncated,_=env.step(a)
        if truncated:break
    r=np.array(env.returns);v=np.array(env.values);peak=np.maximum.accumulate(v)
    metrics={'total_return':float(v[-1]/v[0]-1),'sharpe':float(r.mean()/max(r.std(),1e-12)*np.sqrt(252)),'max_drawdown':float((v/peak-1).min()),'query_rate':float(np.mean([z['n_queries']>0 for z in env.history])),'KG_hits':int(sum(z['KG_hits'] for z in env.history)),'SEC_hits':int(sum(z['SEC_hits'] for z in env.history))}
    return metrics,pd.DataFrame(env.history)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kgt',required=True);p.add_argument('--kg',default='outputs/unified_temporal_kg_v3/unified_temporal_edges.csv');p.add_argument('--sec',default='rag/data/processed/sec/sec_chunks.jsonl')
    p.add_argument('--ollama-host',default='http://140.113.87.164:11434');p.add_argument('--llm',default='qwen2.5:3b');p.add_argument('--embed',default='bge-m3:latest')
    p.add_argument('--device',default='auto');p.add_argument('--max-new-tokens',type=int,default=48)
    p.add_argument('--top-k',type=int,default=3);p.add_argument('--max-search-steps',type=int,default=3)
    p.add_argument('--agents',nargs='+',default=['A2C','PPO','DDPG'],choices=['A2C','PPO','DDPG'])
    p.add_argument('--modes',nargs='+',default=['ACTIVE','NEVER','ALWAYS_BOTH'],choices=['ACTIVE','NEVER','ALWAYS_BOTH'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,123,2026,7,999]);p.add_argument('--timesteps',type=int,default=50000)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--transaction-cost',type=float,default=.001)
    p.add_argument('--reward-scaling',type=float,default=1.0);p.add_argument('--tier',default='conservative')
    p.add_argument('--cost-scale',type=float,default=1.0);p.add_argument('--out',default='results/aegis_rl_v1/flare_raggym_live')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();mandatory(args.max_search_steps>=1,'max-search-steps must be >=1')
    out=(ROOT/args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    for path in [args.kgt,args.kg,args.sec]:mandatory(Path(path).exists(),f'Missing input: {path}')
    _,_,schema,market,_,_=v2.read_original_data(args)
    base_cols=schema['FULL_DYNAMIC'];market=market.copy();market['date']=pd.to_datetime(market.date)
    dates=sorted(market.date.unique())
    kgt,cols=legacy.load_kgt(args.kgt,dates)
    mandatory(len(cols)==200,'Expected 200 KGT dimensions')
    kg=legacy.load_kg(args.kg);sec=legacy.load_sec(args.sec)
    engine=LLMEngine(args.llm,args.embed,out/'llm_cache.sqlite',args.device,args.max_new_tokens,args.ollama_host)
    retriever=TemporalRetrieval(kg,sec,engine,args.top_k)
    print('LOADED:',len(kg),'dated KG rows,',len(sec),'dated SEC chunks,',len(cols),'KGT dimensions,',engine.dim,'evidence dimensions per source',flush=True)
    def env(split,mode):return MultiStepEnv(market[market.split.eq(split)],base_cols,kgt,retriever,engine,mode,args.max_search_steps,args.transaction_cost,args.cost_scale,split)
    if args.smoke:
        e=env('val','ACTIVE');obs,_=e.reset()
        for source_idx in [1,2]:
            action=np.zeros(18,dtype=np.float32);action[source_idx]=1;action[8]=1;action[9]=1
            obs,_,_,_,_=e.step(action)
            q=e.day_queries[-1]
            print('SMOKE',q['source'],'query=',q['query'],'token_confidence=',q['generation_token_confidence'],'hits=',len(q['evidence']))
            for hit in q['evidence'][:2]:print(' ',hit['source'],hit['available_at'],hit['text_excerpt'][:90])
        print('SMOKE: PASS (retrieval executed; zero hits must be investigated if present)');return
    config={'kgt':str(Path(args.kgt).resolve()),'kg':str(Path(args.kg).resolve()),'sec':str(Path(args.sec).resolve()),'hashes':{k:sha(pth) for k,pth in [('kgt',args.kgt),('kg',args.kg),('sec',args.sec)]},'ollama_host':args.ollama_host,'llm':args.llm,'embed':args.embed,'timesteps':args.timesteps,'seeds':args.seeds,'modes':args.modes,'agents':args.agents,'cost_scale':args.cost_scale,'search_steps':args.max_search_steps,'strict_pit_certified':False,'notes':['The local LLM may know future market events from pretraining','KGT representation training provenance not certified','No token-calibrated ground-truth knowledge guarantee','This is FLARE/RAG-Gym-inspired; not original framework reproduction']}
    cfg=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();config['config_hash']=cfg
    manifest=out/'manifest.json'
    if manifest.exists():mandatory(json.loads(manifest.read_text()).get('config_hash')==cfg,'Output configuration mismatch; change --out')
    manifest.write_text(json.dumps(config,indent=2))
    allrows=[]
    for mode in args.modes:
        for agent in args.agents:
            for seed in args.seeds:
                name=f'{mode}__{agent}__seed{seed}';folder=out/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
                result_path=folder/'result.json'
                if result_path.exists():
                    prior=json.loads(result_path.read_text());mandatory(prior.get('config_hash')==cfg,'Resume hash mismatch');allrows.extend(prior['rows']);print('[RESUME]',name,flush=True);continue
                print('[TRAIN]',name,flush=True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                model=build_model(agent,env('train',mode),seed,args.learning_rate,'cpu' if args.device=='auto' else args.device)
                model.learn(total_timesteps=args.timesteps,progress_bar=False);model.save(str(folder/'model'))
                runrows=[]
                for split in ['val','test']:
                    metrics,hist=evaluate(model,env(split,mode));hist.to_csv(folder/f'{split}_queries.csv',index=False)
                    row={'mode':mode,'agent':agent,'seed':seed,'split':split,**metrics};runrows.append(row);allrows.append(row)
                result_path.write_text(json.dumps({'config_hash':cfg,'rows':runrows},indent=2))
                pd.DataFrame(allrows).to_csv(out/'all_results.csv',index=False)
    df=pd.DataFrame(allrows);df.to_csv(out/'all_results.csv',index=False)
    summary=df[df.split=='test'].groupby(['mode','agent']).agg(return_mean=('total_return','mean'),sharpe_mean=('sharpe','mean'),mdd_mean=('max_drawdown','mean'),query_rate=('query_rate','mean'),kg_hits=('KG_hits','mean'),sec_hits=('SEC_hits','mean')).reset_index()
    summary.to_csv(out/'test_summary.csv',index=False);print(summary.to_string(index=False));print('FINRL-X LIVE MULTI-STEP RAG: PASS')

if __name__=='__main__':main()
