# Copyright 2025 ellipse2v
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Before/after benchmark of the CPE matcher.

Runs every case of cpe_benchmark_cases.csv through whatever version of
src/cpe_matcher/cpe_matcher.py is currently checked out, reproducing what the
Excel mode writes in the 'CPE' column, and stores accuracy + timings in a JSON
file. Two JSON files can then be compared.

    python benchmark/run_benchmark.py --label after
    python benchmark/run_benchmark.py --compare benchmark/results_before.json benchmark/results_after.json

The script only relies on the public API shared by the old and new versions
(CPEMatcher(), load_data(), search()), so it works on both.
"""

import argparse
import csv
import importlib
import json
import os
import re
import sys
import time
import zlib
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CATEGORIES = ('exact', 'substituted', 'variant', 'variant_substituted', 'typo', 'negative')


def split_cpe(cpe):
    return re.split(r'(?<!\\):', cpe)


def version_key(v):
    v = re.sub(r'\s+', '_', str(v).strip().lower())
    while v.endswith('.0') and len(v) > 2:
        v = v[:-2]
    return v


def load_cases(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


class HashingEncoder:
    """Offline stand-in for a sentence-transformer (character trigrams)."""
    name = 'hashing-trigram-benchmark'

    def __init__(self, dim=512):
        self.dim = dim

    def encode(self, texts, **_kwargs):
        import numpy as np
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = f"  {t.lower()} "
            for j in range(len(t) - 2):
                out[i, zlib.crc32(t[j:j + 3].encode()) % self.dim] += 1.0
        return out

    def get_sentence_embedding_dimension(self):
        return self.dim


def evaluate(case, results, threshold, mod):
    """Score one case the way the Excel mode would fill the CPE column."""
    expected = case['expected_product']
    accepted_products = set(expected.split('|')) if expected else set()
    top = results[0] if results else None
    final_cpe, accepted = None, False
    if top is not None and top['score'] > threshold:
        accepted = True
        if 'version_in_dictionary' in top:
            final_cpe = top['cpe']  # new matcher: version already resolved
        else:
            final_cpe = mod.adjust_cpe_version(top['cpe'], case['version'])  # old process_excel()

    def pvp(cpe):
        p = split_cpe(cpe)
        return ':'.join(p[2:5]) if len(p) > 5 else ''

    ranked = [pvp(r['cpe']) for r in results]
    row = OrderedDict(
        category=case['category'], vendor=case['vendor'], product=case['product'],
        version=case['version'], expected_product=expected,
        top1_cpe=top['cpe'] if top else None,
        top1_score=round(float(top['score']), 4) if top else None,
        final_cpe=final_cpe, accepted=accepted,
    )
    if not expected:
        row['product_hit@1'] = None
        row['product_hit@5'] = None
        row['correct'] = not accepted
        return row
    row['product_hit@1'] = bool(ranked) and ranked[0] in accepted_products
    row['product_hit@5'] = bool(accepted_products & set(ranked[:5]))
    version_ok = False
    if final_cpe:
        p = split_cpe(final_cpe)
        version_ok = len(p) > 5 and version_key(p[5].replace('\\', '')) == version_key(case['version'])
    row['correct'] = bool(final_cpe) and pvp(final_cpe) in accepted_products and version_ok
    return row


def summarize(rows):
    def pct(vals):
        vals = [v for v in vals if v is not None]
        return round(100.0 * sum(vals) / len(vals), 1) if vals else None

    summary = OrderedDict()
    for cat in CATEGORIES + ('ALL',):
        sub = [r for r in rows if cat == 'ALL' or r['category'] == cat]
        if not sub:
            continue
        summary[cat] = OrderedDict(
            n=len(sub),
            correct_pct=pct([r['correct'] for r in sub]),
            product_hit1_pct=pct([r['product_hit@1'] for r in sub]),
            product_hit5_pct=pct([r['product_hit@5'] for r in sub]),
            accepted_pct=pct([r['accepted'] for r in sub]),
        )
    return summary


def print_summary(label, summary, timings):
    print(f"\n=== {label} ===")
    print(f"{'category':<21}{'n':>4}{'correct%':>10}{'hit@1%':>9}{'hit@5%':>9}{'accepted%':>11}")
    for cat, s in summary.items():
        print(f"{cat:<21}{s['n']:>4}{s['correct_pct']!s:>10}{s['product_hit1_pct']!s:>9}"
              f"{s['product_hit5_pct']!s:>9}{s['accepted_pct']!s:>11}")
    print("timings: " + ", ".join(f"{k}={v}" for k, v in timings.items()))


def run(args):
    sys.path.insert(0, PROJECT_ROOT)
    mod = importlib.import_module('src.cpe_matcher.cpe_matcher')
    cases = load_cases(args.cases)
    if args.category:
        cases = [c for c in cases if c['category'] in args.category]

    kwargs = {}
    if args.config:
        kwargs['config_path'] = args.config
    if args.use_mini_llm:
        kwargs['use_mini'] = True
    if args.fake_model:
        kwargs['model'] = HashingEncoder()

    t0 = time.time()
    matcher = mod.CPEMatcher(**kwargs)
    t_model = time.time() - t0
    t0 = time.time()
    matcher.load_data()
    t_load = time.time() - t0
    threshold = matcher.config['min_score_threshold']

    # Pass 1: one search() per row (interactive mode path) -> accuracy + latency.
    rows, per_query = [], []
    for case in cases:
        t0 = time.time()
        results = matcher.search(case['vendor'], case['product'], case['version'], num_results=5)
        per_query.append(time.time() - t0)
        rows.append(evaluate(case, results, threshold, mod))

    # Pass 2: batch processing of cases * repeat rows (Excel mode path).
    batch = [(c['vendor'], c['product'], c['version']) for c in cases] * args.repeat
    t0 = time.time()
    if hasattr(matcher, 'search_batch'):
        matcher.search_batch(batch, num_results=1)
    else:
        if (getattr(mod, '_FAISS_AVAILABLE', False) and getattr(matcher, '_faiss_index', 1) is None
                and len(batch) >= matcher.config.get('faiss_min_rows', 10 ** 9)):
            matcher._build_faiss_index()  # what the old process_excel() does
        for q in batch:
            matcher.search(*q, num_results=1)
    t_batch = time.time() - t0

    timings = OrderedDict(
        model_load_s=round(t_model, 2),
        data_load_s=round(t_load, 2),
        mean_query_ms=round(1000 * sum(per_query) / max(len(per_query), 1), 1),
        max_query_ms=round(1000 * max(per_query or [0]), 1),
        batch_rows=len(batch),
        batch_total_s=round(t_batch, 2),
        batch_rows_per_s=round(len(batch) / t_batch, 1) if t_batch else None,
    )
    summary = summarize(rows)
    label = args.label or 'run'
    print_summary(label, summary, timings)

    out = args.output or os.path.join(HERE, f'results_{label}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'label': label, 'threshold': threshold, 'summary': summary,
                   'timings': timings, 'cases': rows}, f, indent=2)
    print(f"\nDetailed results written to {out}")


def compare(path_a, path_b):
    with open(path_a, encoding='utf-8') as f:
        a = json.load(f)
    with open(path_b, encoding='utf-8') as f:
        b = json.load(f)
    print(f"\n=== {a['label']}  ->  {b['label']} ===")
    print(f"{'category':<21}{'correct%':>20}{'hit@1%':>20}{'accepted%':>20}")
    for cat, sb in b['summary'].items():
        sa = a['summary'].get(cat, {})

        def cell(key):
            return f"{sa.get(key)!s} -> {sb.get(key)!s}"
        print(f"{cat:<21}{cell('correct_pct'):>20}{cell('product_hit1_pct'):>20}{cell('accepted_pct'):>20}")
    print("\ntimings:")
    for k, vb in b['timings'].items():
        print(f"  {k:<18}{a['timings'].get(k)!s:>12} -> {vb!s}")

    key = lambda r: (r['vendor'], r['product'], r['version'])
    before = {key(r): r for r in a['cases']}
    fixed, broken = [], []
    for r in b['cases']:
        old = before.get(key(r))
        if old is None:
            continue
        if r['correct'] and not old['correct']:
            fixed.append((old, r))
        elif old['correct'] and not r['correct']:
            broken.append((old, r))
    for title, items in (("Fixed", fixed), ("Regressed", broken)):
        print(f"\n{title} ({len(items)}):")
        for old, new in items:
            print(f"  [{new['category']}] {new['vendor']} | {new['product']} | {new['version']}")
            print(f"      {a['label']}: {old['final_cpe']}")
            print(f"      {b['label']}: {new['final_cpe']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--label', help="Name of this run (e.g. before / after)")
    parser.add_argument('--cases', default=os.path.join(HERE, 'cpe_benchmark_cases.csv'))
    parser.add_argument('--category', action='append', choices=CATEGORIES,
                        help="Only run this category (repeatable)")
    parser.add_argument('--config', help="Path to the matcher config.ini")
    parser.add_argument('--use-mini-llm', action='store_true')
    parser.add_argument('--repeat', type=int, default=10,
                        help="Batch timing runs the cases N times (default 10)")
    parser.add_argument('--output', help="Results JSON path (default benchmark/results_<label>.json)")
    parser.add_argument('--fake-model', action='store_true',
                        help="Offline sanity run with a hashing encoder (new matcher only)")
    parser.add_argument('--compare', nargs=2, metavar=('BEFORE_JSON', 'AFTER_JSON'))
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == '__main__':
    main()
