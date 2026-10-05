# FinDKG + SEC-RAG Four-Way Ablation

## Feature Sets

- BASE = 9 technical features
- BASE + SEC = 9 technical + 7 SEC features
- BASE + FinDKG = 9 technical + 200 FinDKG embeddings
- BASE + SEC + FinDKG = 9 technical + 7 SEC + 200 FinDKG

## Protocol

- Train: 2018-01-08 to 2020-12-31
- Held-out test: 2021-01-01 to 2022-12-30
- Agents: A2C, PPO, DDPG
- Seeds: 0,1,2,3,4
- Training steps: 50,000

## FinDKG Pipeline

Released FinDKG temporal KG
-> weekly dynamic graphs
-> KGTransformer
-> 200-d entity embeddings
-> AAPL / MSFT / NVDA / AMZN / GOOGL
-> strict backward as-of daily merge
-> FinRL state
-> daily portfolio weights

## Main Finding

FinDKG provides the most consistent incremental improvement.

Adding FinDKG to BASE improves return and Sharpe across most seeds.
Adding FinDKG on top of SEC improves return and Sharpe in all
5 seeds for all three DRL agents.

SEC alone is more agent-dependent, but becomes strongly complementary
to FinDKG for DDPG.
