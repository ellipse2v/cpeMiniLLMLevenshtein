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

import os
import gzip
import json
import hashlib
import numpy as np
import re
import socket
import sys
import time
import argparse
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import Levenshtein

try:
    import torch
    from sentence_transformers import SentenceTransformer
    import lxml.etree as ET
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm
    import pandas as pd
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure helper functions — no side effects, safe to import
# ---------------------------------------------------------------------------

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


def parse_cpe_name(cpe_string):
    """Return (vendor, product, version) from a CPE 2.3 string."""
    parts = cpe_string.split(':')
    if len(parts) >= 5:
        vendor = parts[3] if parts[3] != '*' else ""
        product = parts[4] if parts[4] != '*' else ""
        version = parts[5] if len(parts) > 5 and parts[5] != '*' else ""
        return vendor, product, version
    return "", "", ""


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def levenshtein_similarity(str1, str2):
    if not str1 or not str2:
        return 0.0
    distance = Levenshtein.distance(str1.lower(), str2.lower())
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)


def version_similarity(v1, v2):
    """
    Numeric-aware version comparison.
    Parses X.Y.Z components and weighs major differences more heavily.
    Falls back to Levenshtein for non-numeric strings.
    Returns 0.5 for wildcard '*' (no version info, neutral score).
    """
    if not v1 or not v2:
        return 0.0
    if v1 == '*' or v2 == '*':
        return 0.5

    def _parse(v):
        parts = v.strip().split('.')
        result = []
        for p in parts:
            try:
                result.append(int(p))
            except ValueError:
                return None
        return result

    p1, p2 = _parse(v1), _parse(v2)
    if p1 is None or p2 is None:
        return levenshtein_similarity(v1, v2)

    max_len = max(len(p1), len(p2))
    p1 += [0] * (max_len - len(p1))
    p2 += [0] * (max_len - len(p2))

    total_weight = 0.0
    total_score = 0.0
    for i, (a, b) in enumerate(zip(p1, p2)):
        weight = 1.0 / (2 ** i)
        max_val = max(a, b, 1)
        total_score += (1.0 - abs(a - b) / max_val) * weight
        total_weight += weight

    return total_score / total_weight


def get_canonical_cpe(cpe_string):
    """Strip architecture-specific suffixes for deduplication (keep first 10 components)."""
    return ':'.join(cpe_string.split(':')[:10])


def adjust_cpe_version(cpe_string, source_version):
    """Replace the version component of a CPE string with source_version, or '*' if empty."""
    parts = cpe_string.split(':')
    if len(parts) > 5:
        if source_version and str(source_version).strip():
            parts[5] = str(source_version).strip().lower().replace(' ', '_')
        else:
            parts[5] = '*'
    return ':'.join(parts)


# ---------------------------------------------------------------------------
# CPEMatcher class
# ---------------------------------------------------------------------------

class CPEMatcher:
    """
    Encapsulates CPE matching: config, model, embeddings, search, and batch processing.
    Instantiate once; call load_data() before search() or process_excel().
    """

    def __init__(self, config_path=None, use_mini=False):
        self.config = self._load_config(config_path)
        self.use_mini = use_mini
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        _t = time.time()
        self.model = self._load_model_with_fallback()
        print(f"Model loaded on {self.device} ({time.time()-_t:.2f}s)")
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        self.cpe_items = None
        self.titles = None
        self.embeddings = None
        self.product_map = None
        self._faiss_index = None
        self._embeddings_normalized = None

    def _load_config(self, config_path=None):
        config = configparser.ConfigParser()
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        config.read(config_path)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        return {
            'default_model': config.get('Models', 'DEFAULT_MODEL'),
            'fallback_model': config.get('Models', 'FALLBACK_MODEL'),
            'default_model_path': os.path.join(project_root, config.get('Paths', 'DEFAULT_MODEL_PATH')),
            'fallback_model_path': os.path.join(project_root, config.get('Paths', 'FALLBACK_MODEL_PATH')),
            'json_filepath': os.path.join(project_root, config.get('Paths', 'CPE_DATA_JSON', fallback='cpe_data.json.gz')),
            'embeddings_filepath': os.path.join(project_root, config.get('Paths', 'CPE_EMBEDDINGS_NUMPY')),
            'xml_filepath': os.path.join(project_root, config.get('Paths', 'CPE_DICTIONARY_XML')),
            'batch_size': config.getint('Settings', 'BATCH_SIZE'),
            'num_results': config.getint('Settings', 'NUM_RESULTS'),
            'force_regenerate': config.getboolean('Settings', 'FORCE_REGENERATE'),
            'semantic_weight': config.getfloat('Settings', 'SEMANTIC_SCORE_WEIGHT'),
            'vendor_weight': config.getfloat('Settings', 'VENDOR_SCORE_WEIGHT'),
            'product_weight': config.getfloat('Settings', 'PRODUCT_SCORE_WEIGHT'),
            'version_weight': config.getfloat('Settings', 'VERSION_SCORE_WEIGHT'),
            'min_score_threshold': config.getfloat('Settings', 'MIN_SCORE_THRESHOLD', fallback=0.7),
            'max_workers': config.getint('Settings', 'MAX_WORKERS', fallback=1),
            'faiss_min_rows': config.getint('Settings', 'FAISS_MIN_ROWS', fallback=100),
        }

    def _load_model_with_fallback(self):
        if self.use_mini:
            primary_path = self.config['fallback_model_path']
            primary_name = self.config['fallback_model']
            secondary_path = self.config['default_model_path']
            secondary_name = self.config['default_model']
        else:
            primary_path = self.config['default_model_path']
            primary_name = self.config['default_model']
            secondary_path = self.config['fallback_model_path']
            secondary_name = self.config['fallback_model']

        if os.path.exists(primary_path):
            print(f"Loading model from {primary_path}")
            try:
                return SentenceTransformer(primary_path, device=self.device)
            except Exception as e:
                print(f"Error loading primary model: {e}")

        print(f"Downloading {primary_name}...")
        if check_internet_connection():
            try:
                os.makedirs(os.path.dirname(primary_path), exist_ok=True)
                model = SentenceTransformer(primary_name, device=self.device)
                model.save(primary_path)
                return model
            except Exception as e:
                print(f"Download failed: {e}")
        else:
            print("No internet connection.")

        if os.path.exists(secondary_path):
            try:
                return SentenceTransformer(secondary_path, device=self.device)
            except Exception as e:
                print(f"Error loading fallback model: {e}")

        print("FATAL: Could not load any model.")
        sys.exit(1)

    def load_data(self, force_regenerate=False):
        """Load or generate CPE data and embeddings. Must be called before search()."""
        _t_total = time.time()
        cfg = self.config
        if self.use_mini:
            cfg = dict(cfg)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            cfg['embeddings_filepath'] = os.path.join(project_root, 'cpe_embeddings_minilm.npy')
            self.config = cfg

        should_regenerate = (
            force_regenerate
            or cfg['force_regenerate']
            or not os.path.exists(cfg['json_filepath'])
            or not os.path.exists(cfg['embeddings_filepath'])
        )

        if not should_regenerate:
            self.cpe_items, self.titles, self.embeddings, self.product_map = self._load_cpe_data()
            if not self.product_map:
                print("Product map missing. Forcing regeneration.")
                should_regenerate = True

        if should_regenerate:
            self.cpe_items, self.titles, self.embeddings, self.product_map = self._prepare_cpe_data()
            if self.cpe_items is not None:
                self._save_cpe_data()

        print(f"Total startup time: {time.time()-_t_total:.2f}s")

    def _load_cpe_data(self):
        cfg = self.config
        json_path = cfg['json_filepath']
        hash_path = json_path + '.sha256'
        try:
            _t = time.time()
            print(f"Loading CPE data from {json_path}...")
            with gzip.open(json_path, 'rb') as f:
                raw = f.read()

            if os.path.exists(hash_path):
                with open(hash_path) as f:
                    expected = f.read().strip()
                if hashlib.sha256(raw).hexdigest() != expected:
                    print("WARNING: Cache integrity check failed. Forcing regeneration.")
                    return None, None, None, None

            data = json.loads(raw.decode('utf-8'))
            cpe_items = data.get('cpe_items', [])
            titles = data.get('titles', [])
            product_map = data.get('product_map', {})
            print(f"  JSON: {len(cpe_items)} items in {time.time()-_t:.2f}s")

            _t = time.time()
            print(f"Loading embeddings from {cfg['embeddings_filepath']}...")
            embeddings = np.load(cfg['embeddings_filepath'], mmap_mode='r')
            print(f"  Embeddings: {embeddings.shape} mapped in {time.time()-_t:.2f}s")
            return cpe_items, titles, embeddings, product_map
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None, None

    def _save_cpe_data(self):
        cfg = self.config
        json_path = cfg['json_filepath']
        hash_path = json_path + '.sha256'
        try:
            data = {
                'cpe_items': self.cpe_items,
                'titles': self.titles,
                'product_map': self.product_map,
            }
            payload = json.dumps(data).encode('utf-8')
            with gzip.open(json_path, 'wb') as f:
                f.write(payload)
            digest = hashlib.sha256(payload).hexdigest()
            with open(hash_path, 'w') as f:
                f.write(digest)
            np.save(cfg['embeddings_filepath'], self.embeddings)
            print(f"Data saved ({json_path}, sha256: {digest[:12]}...)")
        except Exception as e:
            print(f"Error saving data: {e}")

    def _prepare_cpe_data(self):
        cfg = self.config
        root = self._load_cpe_xml(cfg['xml_filepath'])
        if root is None:
            return None, None, None, None

        print("Extracting CPE items...")
        cpe_items, titles = self._extract_cpe_items(root)
        print(f"Extracted {len(cpe_items)} items.")

        product_map = {}
        for i, cpe_string in enumerate(tqdm(cpe_items, desc="Mapping products")):
            _, product, _ = parse_cpe_name(cpe_string)
            if product:
                key = product.lower().replace('_', ' ')
                if key not in product_map:
                    product_map[key] = []
                product_map[key].append(i)

        texts = []
        for cpe, title in zip(cpe_items, titles):
            vendor, product, version = parse_cpe_name(cpe)
            text = f"{clean_text(vendor.replace('_', ' '))} {clean_text(product.replace('_', ' '))}"
            if version:
                text += f" {clean_text(version.replace('_', ' '))}"
            if title:
                text += f" {clean_text(title)}"
            texts.append(text.strip())

        embeddings = self._create_embeddings(texts)
        return cpe_items, titles, embeddings, product_map

    def _load_cpe_xml(self, filepath):
        abs_path = os.path.abspath(filepath)
        if not os.path.exists(abs_path):
            print(f"Error: XML not found: {abs_path}")
            return None
        try:
            return ET.parse(filepath).getroot()
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return None

    def _extract_cpe_items(self, root):
        ns = {
            'cpe': 'http://cpe.mitre.org/dictionary/2.0',
            'cpe-23': 'http://scap.nist.gov/schema/cpe-extension/2.3',
        }
        cpe_items, titles = [], []
        for item in root.findall(".//cpe:cpe-item", namespaces=ns):
            cpe23 = item.find(".//cpe-23:cpe23-item", namespaces=ns)
            if cpe23 is not None and cpe23.get("name"):
                title_el = item.find(".//cpe:title", namespaces=ns)
                cpe_items.append(cpe23.get("name"))
                titles.append(title_el.text if title_el is not None else "")
        return cpe_items, titles

    def _create_embeddings(self, texts):
        dim = self.model.get_sentence_embedding_dimension()
        embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
            for i in range(0, len(texts), self.config['batch_size']):
                batch = texts[i:i + self.config['batch_size']]
                if i > 0 and i % (self.config['batch_size'] * 10) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    batch_emb = self.model.encode(batch, convert_to_numpy=True)
                    end = min(i + self.config['batch_size'], len(texts))
                    embeddings[i:end] = batch_emb
                except Exception as e:
                    print(f"Error on batch {i}: {e}")
                pbar.update(len(batch))
        return embeddings

    def _build_faiss_index(self):
        _t = time.time()
        print("Building FAISS index...")
        emb = np.array(self.embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings_normalized = emb / norms
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(self._embeddings_normalized)
        self._faiss_index = index
        print(f"FAISS index ready ({index.ntotal} vectors, {time.time()-_t:.2f}s).")

    def search(self, vendor, product, version, num_results=None):
        """
        Search for matching CPEs.

        Returns list of dicts:
          {score, cpe, title, is_exact_product_match,
           score_breakdown: {semantic, vendor, product, version}}
        """
        if num_results is None:
            num_results = self.config['num_results']

        query = " ".join(filter(None, [
            clean_text(vendor), clean_text(product), clean_text(version)
        ])).strip()
        if not query:
            return []

        _t = time.time()
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        print(f"  Query embedding: {time.time()-_t:.3f}s")
        _t = time.time()

        exact_indices = set()
        if product:
            normalized_product = product.lower().replace('_', ' ')
            exact_indices = set(self.product_map.get(normalized_product, []))

        k = min(num_results * 20, len(self.cpe_items))

        if _FAISS_AVAILABLE and self._faiss_index is not None:
            q_norm = query_embedding / (np.linalg.norm(query_embedding) or 1.0)
            distances, faiss_ids = self._faiss_index.search(q_norm.astype(np.float32), k)
            sim_map = {int(i): float(d) for i, d in zip(faiss_ids[0], distances[0]) if i >= 0}
            top_semantic_indices = set(sim_map.keys())
            extra = exact_indices - top_semantic_indices
            if extra:
                extra_list = list(extra)
                extra_sims = (q_norm @ self._embeddings_normalized[extra_list].T)[0]
                for idx, sim in zip(extra_list, extra_sims):
                    sim_map[idx] = float(sim)
            candidates = exact_indices.union(top_semantic_indices)
            def _get_sim(idx): return sim_map.get(idx, 0.0)
        else:
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_semantic_indices = set(int(i) for i in similarities.argsort()[-k:][::-1])
            candidates = exact_indices.union(top_semantic_indices)
            def _get_sim(idx): return float(similarities[idx])

        results = []
        for idx in candidates:
            cpe = self.cpe_items[idx]
            title = self.titles[idx] if idx < len(self.titles) else ""
            cpe_vendor, cpe_product, cpe_version = parse_cpe_name(cpe)

            is_exact = (
                product.lower().replace('_', ' ') == cpe_product.lower().replace('_', ' ')
                if product and cpe_product else False
            )

            v_score = levenshtein_similarity(vendor, cpe_vendor.replace('_', ' ')) if vendor and cpe_vendor else 0.5
            if product and cpe_product:
                cpe_prod_norm = cpe_product.replace('_', ' ').lower()
                prod_norm = product.lower()
                p_score = levenshtein_similarity(product, cpe_product.replace('_', ' '))
                if prod_norm in cpe_prod_norm or cpe_prod_norm in prod_norm:
                    p_score = max(p_score, 0.8)
            else:
                p_score = 0.0
            ver_score = version_similarity(version, cpe_version.replace('_', ' ')) if version else 0.5
            sem_score = _get_sim(idx)

            combined = (
                sem_score * self.config['semantic_weight']
                + v_score * self.config['vendor_weight']
                + p_score * self.config['product_weight']
                + ver_score * self.config['version_weight']
            )

            results.append({
                'score': combined,
                'cpe': cpe,
                'title': title,
                'is_exact_product_match': is_exact,
                'score_breakdown': {
                    'semantic': sem_score,
                    'vendor': v_score,
                    'product': p_score,
                    'version': ver_score,
                },
            })

        print(f"  Scored {len(candidates)} candidates in {time.time()-_t:.3f}s")
        results.sort(key=lambda x: (x['is_exact_product_match'], x['score']), reverse=True)

        seen = set()
        deduped = []
        for r in results:
            canonical = get_canonical_cpe(r['cpe'])
            if canonical not in seen:
                deduped.append(r)
                seen.add(canonical)

        return deduped[:num_results]

    def process_excel(self, excel_path, output_path=None):
        """Batch-process an Excel file using parallel workers. Returns True on success."""
        try:
            df = pd.read_excel(excel_path)
            required = ['Vendor', 'Product', 'Version', 'CPE', 'Levenshtein score']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"Missing columns: {', '.join(missing)}")
                return False

            threshold = self.config['min_score_threshold']
            max_workers = self.config.get('max_workers', 1)
            faiss_min_rows = self.config['faiss_min_rows']
            row_count = len(df)

            if _FAISS_AVAILABLE and self._faiss_index is None and row_count >= faiss_min_rows:
                print(f"{row_count} rows >= threshold {faiss_min_rows}: building FAISS index...")
                self._build_faiss_index()
            elif row_count < faiss_min_rows:
                print(f"{row_count} rows < threshold {faiss_min_rows}: using brute-force cosine.")

            print(f"Processing {row_count} rows (threshold={threshold}, workers={max_workers})...")

            def _process_row(args):
                idx, row = args
                vendor = str(row['Vendor']) if not pd.isna(row['Vendor']) else ""
                product = str(row['Product']) if not pd.isna(row['Product']) else ""
                version = str(row['Version']) if not pd.isna(row['Version']) else ""
                if not product:
                    return idx, None, None
                results = self.search(vendor, product, version, num_results=1)
                if results and results[0]['score'] > threshold:
                    r = results[0]
                    return idx, adjust_cpe_version(r['cpe'], version), r['score']
                return idx, None, None

            rows = list(df.iterrows())
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_process_row, row): row[0] for row in rows}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    idx, cpe_val, score_val = future.result()
                    if cpe_val is not None:
                        df.at[idx, 'CPE'] = cpe_val
                        df.at[idx, 'Levenshtein score'] = score_val

            if not output_path:
                base, _ = os.path.splitext(excel_path)
                output_path = f"{base}_updated.xlsx"

            df.to_excel(output_path, index=False)
            print(f"Saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error processing Excel: {e}")
            return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _interactive_mode(matcher):
    while True:
        print("\n=== Search for CPE codes ===")
        vendor = input("Vendor (empty=unknown, 'q'=quit): ")
        if vendor.lower() == 'q':
            break
        product = input("Product ('q'=quit): ")
        if product.lower() == 'q':
            break
        if not product:
            print("Product name is required.")
            continue
        version = input("Version: ")

        t0 = time.time()
        results = matcher.search(vendor, product, version)
        elapsed = time.time() - t0

        if results:
            print(f"\nResults for '{vendor} {product} {version}' ({elapsed:.2f}s):")
            for r in results:
                v, p, ver = parse_cpe_name(r['cpe'])
                bd = r['score_breakdown']
                print(f"- {r['cpe']}")
                print(f"  Score: {r['score']:.4f}  "
                      f"(semantic={bd['semantic']:.3f}, vendor={bd['vendor']:.3f}, "
                      f"product={bd['product']:.3f}, version={bd['version']:.3f})")
                print(f"  Vendor: {v.replace('_', ' ')}  "
                      f"Product: {p.replace('_', ' ')}  "
                      f"Version: {ver.replace('_', ' ')}")
                if r['title']:
                    print(f"  Title: {r['title']}")
                print()
        else:
            print(f"No results for '{vendor} {product} {version}'.")


def main():
    parser = argparse.ArgumentParser(description="CPE Matcher")
    parser.add_argument("-data", help="Excel file path", type=str, default=None)
    parser.add_argument("-output", help="Excel output path", type=str, default=None)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--use-mini-llm", action="store_true")
    args = parser.parse_args()

    print("\n=== CPE Matcher ===")
    matcher = CPEMatcher(use_mini=args.use_mini_llm)
    matcher.load_data(force_regenerate=args.force_regenerate)

    if not matcher.cpe_items or matcher.embeddings is None:
        print("Cannot continue without valid CPE data.")
        return

    if args.data:
        if not os.path.exists(args.data):
            print(f"Error: file not found: {args.data}")
            return
        matcher.process_excel(args.data, args.output)
    else:
        _interactive_mode(matcher)


if __name__ == "__main__":
    main()
