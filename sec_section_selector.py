#!/usr/bin/env python3
"""Build a dated, section-aware SEC JSONL corpus for LightRAG V4.

Only prefilters the original chunks. It does NOT claim to recover SEC sections
perfectly and it does NOT perform semantic retrieval. Standard library only.
"""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

SECTION_PATTERNS = {
    "risk_factors": re.compile(r"\b(?:item\s*1a\s*[.:-]?\s*risk\s*factors|risk\s+factors|cybersecurity|supply\s+chain|material\s+adverse|regulatory\s+risk)\b", re.I),
    "mda": re.compile(r"\b(?:item\s*2\s*[.:-]?\s*management|management.s\s+discussion|results\s+of\s+operations|liquidity\s+and\s+capital\s+resources|net\s+sales|gross\s+margin)\b", re.I),
    "legal": re.compile(r"\b(?:legal\s+proceedings|litigation|antitrust|class.action|lawsuit|app\s+store|epic\s+games|regulatory\s+proceedings)\b", re.I),
    "business": re.compile(r"\b(?:competition|competitive\s+pressure|product\s+demand|services\s+revenue|greater\s+china|component\s+shortage|manufacturing\s+capacity)\b", re.I),
}
COVER = re.compile(r"(?:united\s+states\s+securities\s+and\s+exchange\s+commission|indicate\s+by\s+check\s+mark|commission\s+file\s+number|table\s+of\s+contents|registrant\s+has\s+filed\s+all\s+reports)", re.I)


def score_chunk(text):
    # Evaluate complete chunk, not only its first characters.
    matches = {k: len(p.findall(text)) for k, p in SECTION_PATTERNS.items()}
    score = (4 * min(matches['risk_factors'], 3) +
             4 * min(matches['legal'], 3) +
             3 * min(matches['mda'], 3) +
             2 * min(matches['business'], 3))
    if COVER.search(text[:1200]):
        score -= 8
    return score, [k for k, n in matches.items() if n]


def select(args):
    as_of = date.fromisoformat(args.as_of)
    groups = defaultdict(list)
    stats = Counter()
    with Path(args.input).open(encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get('ticker', '')).upper() != args.ticker:
                continue
            stamp = str(r.get('published_at') or '')[:10]
            try:
                published = date.fromisoformat(stamp)
            except ValueError:
                stats['missing_or_bad_date'] += 1
                continue
            if published > as_of:
                stats['future_excluded'] += 1
                continue
            text = str(r.get('text') or '').strip()
            if len(text) < args.min_chars:
                continue
            if not r.get('doc_id') or not r.get('chunk_id'):
                stats['missing_id'] += 1
                continue
            score, tags = score_chunk(text)
            if score < args.min_score:
                continue
            normalized = re.sub(r'\s+', ' ', text).lower()
            fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
            groups[(stamp, str(r['doc_id']))].append((score, r, tags, fingerprint))
            stats['candidates'] += 1
    # At most N latest eligible filings; choose high-signal chunks *within* each.
    filings = sorted(groups.keys(), reverse=True)[:args.max_docs]
    selected = []
    fingerprints = set()
    for filing in filings:
        candidates = sorted(groups[filing], key=lambda x: (-x[0], int(x[1].get('chunk_index', 0))))
        kept = 0
        for score, r, tags, fingerprint in candidates:
            if fingerprint in fingerprints:
                stats['duplicate_excluded'] += 1
                continue
            if kept >= args.chunks_per_doc:
                break
            r = dict(r)
            # Keep original source schema compatible with late_fusion_v4.py.
            r['selector_score'] = score
            r['selector_tags'] = tags
            r['selector_version'] = 'section-aware-v1'
            selected.append(r)
            fingerprints.add(fingerprint)
            kept += 1
    selected.sort(key=lambda r: (r['published_at'], r['doc_id'], int(r.get('chunk_index', 0))), reverse=True)
    target = Path(args.output)
    if target.resolve() == Path(args.input).resolve():
        raise SystemExit('Refusing to overwrite original input')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    report = {'ticker': args.ticker, 'as_of': args.as_of, 'input': args.input,
              'output': str(target), 'selected_chunks': len(selected),
              'selected_filings': len(set(r['doc_id'] for r in selected)),
              'tags': dict(Counter(tag for r in selected for tag in r['selector_tags'])),
              'stats': dict(stats),
              'warning': 'Heuristic chunk scoring is not verified SEC section parsing. Dates rely on local published_at metadata.'}
    report_path = target.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not selected:
        raise SystemExit('No selected chunks; adjust filters or inspect corpus')
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='rag/data/processed/sec/sec_chunks.jsonl')
    p.add_argument('--output', default='rag/data/processed/sec/sec_sections_AAPL_2021-06-01.jsonl')
    p.add_argument('--ticker', default='AAPL')
    p.add_argument('--as-of', default='2021-06-01')
    p.add_argument('--max-docs', type=int, default=3)
    p.add_argument('--chunks-per-doc', type=int, default=3)
    p.add_argument('--min-score', type=int, default=2)
    p.add_argument('--min-chars', type=int, default=150)
    a = p.parse_args()
    a.ticker = a.ticker.upper()
    if min(a.max_docs, a.chunks_per_doc, a.min_chars) < 1:
        p.error('limits must be positive')
    return a


if __name__ == '__main__':
    select(parser())
