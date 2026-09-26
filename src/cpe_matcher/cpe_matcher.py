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
CPE matcher.

The NVD dictionary holds ~1.5 M CPE names but only ~145 k distinct
(part, vendor, product) triples: most entries are the same product with a
different version. Matching is therefore done in two steps:

1. Identify the product: semantic similarity (sentence-transformers) on one
   embedding per product, combined with Levenshtein vendor/product scores.
2. Resolve the version: if the requested version exists in the dictionary the
   official CPE is returned; otherwise a valid CPE is built from the product's
   template by substituting the requested version (e.g. a query for Internet
   Explorer 3 yields cpe:2.3:a:microsoft:internet_explorer:3:*:*:*:*:*:*:*
   even though NVD only lists versions 5.01 and later).
"""

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
import gc
from collections import Counter
import Levenshtein

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:  # tqdm is only cosmetic
    def tqdm(iterable=None, **_kwargs):
        return iterable


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DEFAULT_CONFIG = {
    'Models': {
        'DEFAULT_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
        'FALLBACK_MODEL': 'sentence-transformers/all-mpnet-base-v2',
    },
    'Paths': {
        'DEFAULT_MODEL_PATH': 'models/all-MiniLM-L6-v2',
        'FALLBACK_MODEL_PATH': 'models/all-mpnet-base-v2',
        'CPE_DATA_JSON': 'cpe_data.json.gz',
        'CPE_DICTIONARY_XML': 'official-cpe-dictionary_v2.3.xml',
        'EMBEDDINGS_DIR': '.',
    },
    'Settings': {
        'BATCH_SIZE': '256',
        'NUM_RESULTS': '5',
        'FORCE_REGENERATE': 'false',
        'SEMANTIC_SCORE_WEIGHT': '0.5',
        'VENDOR_SCORE_WEIGHT': '0.2',
        'PRODUCT_SCORE_WEIGHT': '0.2',
        'VERSION_SCORE_WEIGHT': '0.1',
        'MIN_SCORE_THRESHOLD': '0.6',
        'SEMANTIC_TOP_K': '50',
    },
}

CPE_FIELDS = ('part', 'vendor', 'product', 'version', 'update', 'edition',
              'language', 'sw_edition', 'target_sw', 'target_hw', 'other')

_TOKENS = re.compile(r'[a-z0-9]+')
_SAFE_VERSION = re.compile(r'[a-z0-9._\-]+')

_VENDOR_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'ltd',
    'limited', 'llc', 'gmbh', 'ag', 'sa', 'sas', 'srl', 'bv', 'plc', 'the',
    'software', 'technologies', 'technology', 'foundation', 'project',
}


# ---------------------------------------------------------------------------
# Pure helper functions — no side effects, safe to import
# ---------------------------------------------------------------------------

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def split_cpe(cpe_string):
    """Split a CPE 2.3 formatted string on ':' while honouring backslash escapes."""
    if '\\' not in cpe_string:
        return cpe_string.split(':')
    parts, buf, escaped = [], [], False
    for ch in cpe_string:
        if escaped:
            buf.append(ch)
            escaped = False
        elif ch == '\\':
            buf.append(ch)
            escaped = True
        elif ch == ':':
            parts.append(''.join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append(''.join(buf))
    return parts


def unescape_cpe_value(value):
    if '\\' not in value:
        return value
    return re.sub(r'\\(.)', r'\1', value)


def escape_cpe_value(value):
    """
    Turn free text into a CPE 2.3 formatted-string component: lower case,
    whitespace -> '_', and every character other than [a-z0-9._-] escaped.
    Returns '*' for empty input.
    """
    value = re.sub(r'\s+', '_', str(value).strip().lower())
    if not value:
        return '*'
    return ''.join(ch if re.match(r'[a-z0-9._\-]', ch) else '\\' + ch for ch in value)


def parse_cpe_name(cpe_string):
    """Return (vendor, product, version) from a CPE 2.3 string ('' for '*')."""
    parts = split_cpe(cpe_string)
    if len(parts) >= 5:
        vendor = parts[3] if parts[3] != '*' else ""
        product = parts[4] if parts[4] != '*' else ""
        version = parts[5] if len(parts) > 5 and parts[5] != '*' else ""
        return vendor, product, version
    return "", "", ""


def build_cpe(part, vendor, product, version='*'):
    """Build a CPE 2.3 name with every field after the version set to '*'."""
    return ':'.join(['cpe', '2.3', part, vendor, product, version or '*'] + ['*'] * 7)


def clean_text(text):
    if not text:
        return ""
    return ' '.join(_TOKENS.findall(str(text).lower()))


def compact_key(text):
    """'Internet-Explorer', 'internet_explorer', 'InternetExplorer' -> 'internetexplorer'."""
    return re.sub(r'[^a-z0-9+#]', '', unescape_cpe_value(str(text)).lower())


def product_key(text):
    """'Internet Explorer' / 'internet_explorer' -> 'internet explorer'; keeps 'g++', 'c#'."""
    return re.sub(r'\s+', ' ', unescape_cpe_value(str(text)).lower().replace('_', ' ')).strip()


def normalize_version_key(version):
    """
    Key used to look a version up in the dictionary: '4.0' and '4' are the
    same release, so trailing '.0' groups are dropped.
    """
    v = str(version).strip().lower()
    if not _SAFE_VERSION.fullmatch(v):
        v = escape_cpe_value(v)
    while v.endswith('.0') and len(v) > 2:
        v = v[:-2]
    return v


def levenshtein_similarity(str1, str2):
    if not str1 or not str2:
        return 0.0
    distance = Levenshtein.distance(str1.lower(), str2.lower())
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)


def _strip_vendor_suffixes(vendor):
    tokens = clean_text(unescape_cpe_value(vendor).replace('_', ' ')).split()
    kept = [t for t in tokens if t not in _VENDOR_SUFFIXES]
    return ''.join(kept or tokens)


def vendor_similarity(query_vendor, cpe_vendor):
    """Levenshtein on vendor names after removing 'Inc', 'Corporation', ..."""
    a, b = _strip_vendor_suffixes(query_vendor), _strip_vendor_suffixes(cpe_vendor)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    score = levenshtein_similarity(a, b)
    if len(a) >= 3 and len(b) >= 3 and (a in b or b in a):
        score = max(score, 0.85)
    return score


def product_similarity(query_product, cpe_product):
    a = clean_text(unescape_cpe_value(query_product).replace('_', ' '))
    b = clean_text(unescape_cpe_value(cpe_product).replace('_', ' '))
    if not a or not b:
        return 0.0
    if a.replace(' ', '') == b.replace(' ', ''):
        return 1.0
    score = levenshtein_similarity(a, b)
    if a in b or b in a:
        score = max(score, 0.8)
    return score


def product_variants(vendor, product):
    """
    The product as given plus, when it starts with the vendor name, the product
    without it: ('Microsoft', 'Microsoft SQL Server') -> ['Microsoft SQL Server', 'SQL Server'].
    """
    variants = [product]
    vendor_tokens = set(clean_text(vendor).split()) - _VENDOR_SUFFIXES if vendor else set()
    words = str(product).split()
    i = 0
    while i < len(words) - 1 and clean_text(words[i]) in vendor_tokens:
        i += 1
    if i:
        variants.append(' '.join(words[i:]))
    return variants


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
    return ':'.join(split_cpe(cpe_string)[:10])


def adjust_cpe_version(cpe_string, source_version):
    """Replace the version component of a CPE string with source_version, or '*' if empty."""
    parts = split_cpe(cpe_string)
    if len(parts) > 5:
        if source_version and str(source_version).strip():
            parts[5] = escape_cpe_value(source_version)
        else:
            parts[5] = '*'
    return ':'.join(parts)


def strip_version_from_title(title, fields):
    """
    'Microsoft Internet Explorer 6 SP1' with version '6' / update 'sp1'
    -> 'Microsoft Internet Explorer'. Used to build a version-less product text.
    """
    drop = set()
    for f in fields:
        if f and f not in ('*', '-'):
            drop.update(_TOKENS.findall(f.lower()))
    if not drop:
        return str(title or '')
    words = []
    for w in str(title or '').split():
        toks = _TOKENS.findall(w.lower())
        if toks and all(t in drop for t in toks):
            continue
        words.append(w)
    return ' '.join(words)


def _wildcard_count(parts):
    return sum(1 for p in parts[6:] if p in ('*', '-'))


class ProductIndex:
    """
    Groups CPE names by (part, vendor, product).

    Attributes (parallel lists, one entry per product):
      keys       -> (part, vendor, product) as they appear in the CPE names
      texts      -> version-less text used for the embedding
      versions   -> {normalized_version: [cpe indices]}
    plus lookup tables by product name and compact product/vendor names.
    """

    def __init__(self, cpe_items, titles):
        # Millions of small allocations: the cyclic GC only slows this down.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            self._build(cpe_items, titles)
        finally:
            if gc_was_enabled:
                gc.enable()

    def _build(self, cpe_items, titles):
        self.cpe_items = cpe_items
        self.titles = titles
        key_to_id = {}
        self.keys = []
        self.versions = []
        title_votes = []
        n_titles = len(titles)

        for i, cpe in enumerate(cpe_items):
            parts = split_cpe(cpe)
            if len(parts) < 6:
                continue
            key = (parts[2], parts[3], parts[4])
            pid = key_to_id.get(key)
            if pid is None:
                pid = len(self.keys)
                key_to_id[key] = pid
                self.keys.append(key)
                self.versions.append({})
                title_votes.append(Counter())
            ver = parts[5]
            vkey = ver if ver in ('*', '-') else normalize_version_key(unescape_cpe_value(ver))
            bucket = self.versions[pid].get(vkey)
            if bucket is None:
                self.versions[pid][vkey] = [i]
            else:
                bucket.append(i)
            votes = title_votes[pid]
            if i < n_titles and titles[i] and sum(votes.values()) < 5:
                stripped = strip_version_from_title(titles[i], parts[5:])
                if stripped:
                    votes[stripped] += 1

        self.texts = []
        self.display_titles = []
        for pid, (part, vendor, product) in enumerate(self.keys):
            title = title_votes[pid].most_common(1)[0][0] if title_votes[pid] else ''
            self.display_titles.append(title)
            words = []
            for w in (clean_text(unescape_cpe_value(vendor).replace('_', ' ')) + ' '
                      + clean_text(unescape_cpe_value(product).replace('_', ' ')) + ' '
                      + clean_text(title)).split():
                if w not in words:
                    words.append(w)
            self.texts.append(' '.join(words))

        self.by_product = {}
        self.by_compact_product = {}
        self.by_vendor = {}
        for pid, (_, vendor, product) in enumerate(self.keys):
            self.by_product.setdefault(product_key(product), []).append(pid)
            self.by_compact_product.setdefault(compact_key(product), []).append(pid)
            self.by_vendor.setdefault(_strip_vendor_suffixes(vendor), []).append(pid)

    def __len__(self):
        return len(self.keys)

    def lexical_candidates(self, vendor, product):
        """Products whose name matches the query exactly (modulo separators/case)."""
        cands = set()
        if product:
            for variant in product_variants(vendor, product)[1:]:
                cands.update(self.by_product.get(product_key(variant), []))
                cands.update(self.by_compact_product.get(compact_key(variant), []))
            cands.update(self.by_product.get(product_key(product), []))
            ck = compact_key(product)
            cands.update(self.by_compact_product.get(ck, []))
            if vendor and len(ck) >= 3:
                # Vendor's products whose name contains the query (e.g. 'office' -> 'office_2019')
                for pid in self.by_vendor.get(_strip_vendor_suffixes(vendor), []):
                    if ck and ck in compact_key(self.keys[pid][2]):
                        cands.add(pid)
        return cands

    def resolve_version(self, pid, version):
        """
        Return (cpe, version_score, version_in_dictionary, reference_cpe).

        * version found in the dictionary -> the official CPE (the least
          specific variant, e.g. '6:*' rather than '6:sp1').
        * version unknown -> a valid CPE built from the product with the
          requested version substituted; scored by closeness to the nearest
          known version.
        * no version requested -> product CPE with version '*'.
        """
        part, vendor, product = self.keys[pid]
        versions = self.versions[pid]
        if not version or not str(version).strip():
            return build_cpe(part, vendor, product, '*'), 0.5, False, None

        vkey = normalize_version_key(version)
        hits = versions.get(vkey)
        if hits:
            best = max(hits, key=lambda i: _wildcard_count(split_cpe(self.cpe_items[i])))
            best_cpe = self.cpe_items[best]
            if _wildcard_count(split_cpe(best_cpe)) == len(split_cpe(best_cpe)) - 6:
                return best_cpe, 1.0, True, best_cpe
            # Only platform-specific variants exist (e.g. target_sw=chrome):
            # return a generic CPE for that exact version.
            return build_cpe(part, vendor, product, escape_cpe_value(version)), 1.0, True, best_cpe

        closest, closest_score = None, 0.0
        for known, idxs in versions.items():
            if known in ('*', '-'):
                continue
            s = version_similarity(vkey, known)
            if s > closest_score:
                closest, closest_score = idxs[0], s
        ref = self.cpe_items[closest] if closest is not None else None
        cpe = build_cpe(part, vendor, product, escape_cpe_value(version))
        return cpe, 0.5 + 0.5 * closest_score, False, ref


# ---------------------------------------------------------------------------
# CPEMatcher class
# ---------------------------------------------------------------------------

class CPEMatcher:
    """
    Encapsulates CPE matching: config, model, embeddings, search, and batch processing.
    Instantiate once; call load_data() before search() or process_excel().

    `model` may be injected (any object exposing encode(list[str]) -> ndarray);
    this is what the unit tests do to avoid downloading a transformer.
    """

    def __init__(self, config_path=None, use_mini=False, model=None):
        self.config = self._load_config(config_path)
        self.use_mini = use_mini
        self.device = None
        if model is not None:
            self.model = model
            self.model_name = getattr(model, 'name', type(model).__name__)
        else:
            self.model, self.model_name = self._load_model_with_fallback()

        self.cpe_items = None
        self.titles = None
        self.index = None
        self.embeddings = None  # one normalized row per product
        self._faiss_index = None

    # -- configuration / model ---------------------------------------------

    def _load_config(self, config_path=None):
        config = configparser.ConfigParser()
        config.read_dict(DEFAULT_CONFIG)
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        if os.path.exists(config_path):
            config.read(config_path)
        elif config_path != os.path.join(os.path.dirname(__file__), 'config.ini'):
            raise FileNotFoundError(f"Config not found: {config_path}")

        def _path(section, key):
            return os.path.join(PROJECT_ROOT, config.get(section, key))

        return {
            'default_model': config.get('Models', 'DEFAULT_MODEL'),
            'fallback_model': config.get('Models', 'FALLBACK_MODEL'),
            'default_model_path': _path('Paths', 'DEFAULT_MODEL_PATH'),
            'fallback_model_path': _path('Paths', 'FALLBACK_MODEL_PATH'),
            'json_filepath': _path('Paths', 'CPE_DATA_JSON'),
            'xml_filepath': _path('Paths', 'CPE_DICTIONARY_XML'),
            'embeddings_dir': _path('Paths', 'EMBEDDINGS_DIR'),
            'batch_size': config.getint('Settings', 'BATCH_SIZE'),
            'num_results': config.getint('Settings', 'NUM_RESULTS'),
            'force_regenerate': config.getboolean('Settings', 'FORCE_REGENERATE'),
            'semantic_weight': config.getfloat('Settings', 'SEMANTIC_SCORE_WEIGHT'),
            'vendor_weight': config.getfloat('Settings', 'VENDOR_SCORE_WEIGHT'),
            'product_weight': config.getfloat('Settings', 'PRODUCT_SCORE_WEIGHT'),
            'version_weight': config.getfloat('Settings', 'VERSION_SCORE_WEIGHT'),
            'min_score_threshold': config.getfloat('Settings', 'MIN_SCORE_THRESHOLD'),
            'semantic_top_k': config.getint('Settings', 'SEMANTIC_TOP_K'),
        }

    def _load_model_with_fallback(self):
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        candidates = [
            (self.config['default_model_path'], self.config['default_model']),
            (self.config['fallback_model_path'], self.config['fallback_model']),
        ]
        if self.use_mini:
            candidates.reverse()

        _t = time.time()
        for path, name in candidates:
            if os.path.exists(path):
                try:
                    print(f"Loading model from {path}")
                    model = SentenceTransformer(path, device=self.device)
                    print(f"Model loaded ({time.time()-_t:.2f}s)")
                    return model, name
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            if check_internet_connection():
                try:
                    print(f"Downloading {name}...")
                    model = SentenceTransformer(name, device=self.device)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    model.save(path)
                    print(f"Model loaded ({time.time()-_t:.2f}s)")
                    return model, name
                except Exception as e:
                    print(f"Download failed: {e}")
            else:
                print(f"No internet connection, cannot download {name}.")

        print("FATAL: Could not load any model.")
        sys.exit(1)

    def _encode(self, texts):
        """Encode texts to L2-normalized float32 vectors."""
        vecs = self.model.encode(texts, batch_size=self.config['batch_size'],
                                 convert_to_numpy=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    # -- data loading ------------------------------------------------------

    def _embeddings_paths(self):
        safe = re.sub(r'[^A-Za-z0-9_.-]', '_', self.model_name.split('/')[-1])
        base = os.path.join(self.config['embeddings_dir'], f'cpe_product_embeddings_{safe}')
        return base + '.npy', base + '.texts.json.gz'

    def load_data(self, force_regenerate=False):
        """Load CPE data and product embeddings. Must be called before search()."""
        _t_total = time.time()
        cfg = self.config
        force = force_regenerate or cfg['force_regenerate']

        xml_path, json_path = cfg['xml_filepath'], cfg['json_filepath']
        xml_is_newer = (os.path.exists(xml_path) and os.path.exists(json_path)
                        and os.path.getmtime(xml_path) > os.path.getmtime(json_path))
        if xml_is_newer:
            # generate_cpe_dictionary.py produced a fresher dictionary
            print(f"{xml_path} is newer than the cache: rebuilding CPE data.")
            self.cpe_items, self.titles = None, None
        else:
            self.cpe_items, self.titles = self._load_cpe_data()
        if self.cpe_items is None:
            print("Parsing XML dictionary...")
            self.cpe_items, self.titles = self._load_cpe_xml(cfg['xml_filepath'])
            if self.cpe_items is None:
                return
            self._save_cpe_data()

        _t = time.time()
        self.index = ProductIndex(self.cpe_items, self.titles)
        print(f"  Product index: {len(self.index)} products from {len(self.cpe_items)} CPEs "
              f"({time.time()-_t:.2f}s)")

        self.embeddings = self._load_or_build_embeddings(force)
        if _FAISS_AVAILABLE:
            _t = time.time()
            self._top_k(self.embeddings[:1], 1)  # builds the FAISS index
            print(f"  FAISS index: {self._faiss_index.ntotal} vectors ({time.time()-_t:.2f}s)")
        print(f"Total startup time: {time.time()-_t_total:.2f}s")

    def _load_cpe_data(self):
        json_path = self.config['json_filepath']
        hash_path = json_path + '.sha256'
        if not os.path.exists(json_path):
            return None, None
        try:
            _t = time.time()
            print(f"Loading CPE data from {json_path}...")
            with gzip.open(json_path, 'rb') as f:
                raw = f.read()
            if os.path.exists(hash_path):
                with open(hash_path) as f:
                    expected = f.read().strip()
                if hashlib.sha256(raw).hexdigest() != expected:
                    print("WARNING: Cache integrity check failed.")
                    return None, None
            data = json.loads(raw.decode('utf-8'))
            cpe_items = data.get('cpe_items', [])
            titles = data.get('titles', [])
            info = f", NVD data up to {data['last_modified']}" if data.get('last_modified') else ""
            print(f"  JSON: {len(cpe_items)} items in {time.time()-_t:.2f}s{info}")
            return cpe_items, titles
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def _save_cpe_data(self):
        json_path = self.config['json_filepath']
        try:
            payload = json.dumps({'cpe_items': self.cpe_items, 'titles': self.titles}).encode('utf-8')
            with gzip.open(json_path, 'wb') as f:
                f.write(payload)
            digest = hashlib.sha256(payload).hexdigest()
            with open(json_path + '.sha256', 'w') as f:
                f.write(digest)
            print(f"Data saved ({json_path}, sha256: {digest[:12]}...)")
        except Exception as e:
            print(f"Error saving data: {e}")

    def _load_cpe_xml(self, filepath):
        if not os.path.exists(filepath):
            print(f"Error: XML not found: {filepath}")
            return None, None
        import xml.etree.ElementTree as ET
        ns_item = '{http://cpe.mitre.org/dictionary/2.0}cpe-item'
        ns_title = '{http://cpe.mitre.org/dictionary/2.0}title'
        ns_23 = '{http://scap.nist.gov/schema/cpe-extension/2.3}cpe23-item'
        cpe_items, titles = [], []
        try:
            for _, elem in ET.iterparse(filepath, events=('end',)):
                if elem.tag != ns_item:
                    continue
                if elem.get('deprecated') != 'true':
                    cpe23 = elem.find(ns_23)
                    if cpe23 is not None and cpe23.get('name'):
                        title_el = elem.find(ns_title)
                        cpe_items.append(cpe23.get('name'))
                        titles.append(title_el.text if title_el is not None else "")
                elem.clear()
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return None, None
        print(f"Extracted {len(cpe_items)} items.")
        return cpe_items, titles

    def _load_or_build_embeddings(self, force=False):
        """
        One embedding per product. The texts are stored next to the vectors so
        that, after an NVD update, only new products need to be encoded.
        """
        emb_path, texts_path = self._embeddings_paths()
        texts = self.index.texts
        old = {}
        if not force and os.path.exists(emb_path) and os.path.exists(texts_path):
            try:
                _t = time.time()
                with gzip.open(texts_path, 'rt', encoding='utf-8') as f:
                    old_texts = json.load(f)
                old_emb = np.load(emb_path)
                if old_texts == texts and old_emb.shape[0] == len(texts):
                    print(f"  Embeddings: {old_emb.shape} loaded in {time.time()-_t:.2f}s")
                    return old_emb
                if old_emb.shape[0] == len(old_texts):
                    old = {t: i for i, t in enumerate(old_texts)}
            except Exception as e:
                print(f"Could not reuse embeddings: {e}")
                old = {}

        missing = [i for i, t in enumerate(texts) if t not in old]
        print(f"Encoding {len(missing)} new product texts ({len(texts) - len(missing)} reused)...")
        dim = None
        new_vecs = {}
        bs = max(self.config['batch_size'] * 8, 1)
        for start in tqdm(range(0, len(missing), bs), desc="Generating embeddings"):
            chunk = missing[start:start + bs]
            vecs = self._encode([texts[i] for i in chunk])
            dim = vecs.shape[1]
            for i, v in zip(chunk, vecs):
                new_vecs[i] = v
        if old:
            old_emb = np.load(emb_path)
            dim = old_emb.shape[1]
        if dim is None:
            dim = self._encode(['x']).shape[1]
        emb = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            emb[i] = new_vecs[i] if i in new_vecs else old_emb[old[t]]

        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        np.save(emb_path, emb)
        with gzip.open(texts_path, 'wt', encoding='utf-8') as f:
            json.dump(texts, f)
        print(f"Embeddings saved to {emb_path}")
        return emb

    # -- search ------------------------------------------------------------

    def search(self, vendor, product, version, num_results=None):
        """
        Search for matching CPEs for one query. See search_batch() for the
        structure of the returned dicts.
        """
        return self.search_batch([(vendor, product, version)], num_results)[0]

    def search_batch(self, queries, num_results=None):
        """
        Search many (vendor, product, version) queries at once: the queries
        are encoded in batches and compared to all products with one matrix
        product per chunk, which is much faster than one query at a time.

        Returns, per query, a list of dicts (best first, one per product):
          {score, cpe, title, is_exact_product_match, version_in_dictionary,
           reference_cpe, score_breakdown: {semantic, vendor, product, version}}
        `cpe` always carries the requested version; `reference_cpe` is the
        dictionary entry it was derived from (None if the product has none).
        """
        if num_results is None:
            num_results = self.config['num_results']
        queries = [tuple('' if x is None else str(x).strip() for x in q) for q in queries]
        texts = [" ".join(filter(None, [clean_text(v), clean_text(p.replace('_', ' '))])) for v, p, _ in queries]
        todo = [i for i, t in enumerate(texts) if t and queries[i][1]]
        results = [[] for _ in queries]
        if not todo:
            return results

        _t = time.time()
        q_emb = self._encode([texts[i] for i in todo])
        t_enc = time.time() - _t

        _t = time.time()
        k = min(self.config['semantic_top_k'], len(self.index))
        chunk = 256
        for start in range(0, len(todo), chunk):
            q = q_emb[start:start + chunk]
            top_ids, top_sims = self._top_k(q, k)
            for row in range(q.shape[0]):
                qi = todo[start + row]
                vendor, product, version = queries[qi]
                sims = dict(zip(top_ids[row].tolist(), top_sims[row].tolist()))
                extra = [pid for pid in self.index.lexical_candidates(vendor, product) if pid not in sims]
                if extra:
                    sims.update(zip(extra, (self.embeddings[extra] @ q[row]).tolist()))
                sims.pop(-1, None)
                results[qi] = self._rank(vendor, product, version, sims, num_results)
        if len(todo) == 1:
            print(f"  Query embedding: {t_enc:.3f}s, scoring: {time.time()-_t:.3f}s")
        else:
            print(f"  {len(todo)} queries: embedding {t_enc:.2f}s, scoring {time.time()-_t:.2f}s")
        return results

    def _top_k(self, q, k):
        """Top-k products by cosine similarity (FAISS when installed, numpy otherwise)."""
        if _FAISS_AVAILABLE:
            if self._faiss_index is None or self._faiss_index.ntotal != len(self.embeddings):
                self._faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self._faiss_index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))
            sims, ids = self._faiss_index.search(np.ascontiguousarray(q, dtype=np.float32), k)
            return ids, sims
        sims = q @ self.embeddings.T
        if k < sims.shape[1]:
            ids = np.argpartition(sims, sims.shape[1] - k, axis=1)[:, -k:]
        else:
            ids = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
        return ids, np.take_along_axis(sims, ids, axis=1)

    def _rank(self, vendor, product, version, sims, num_results):
        """sims: {product id: cosine similarity} for every candidate product."""
        cfg = self.config
        variants = product_variants(vendor, product)
        norm_products = {compact_key(v) for v in variants}
        scored = []
        for pid, sem in sims.items():
            part, c_vendor, c_product = self.index.keys[pid]
            sem = float(sem)
            v_score = vendor_similarity(vendor, c_vendor) if vendor else 0.5
            p_score = max(product_similarity(v, c_product) for v in variants)
            identity = (sem * cfg['semantic_weight'] + v_score * cfg['vendor_weight']
                        + p_score * cfg['product_weight'])
            is_exact = compact_key(c_product) in norm_products
            scored.append((is_exact, identity, pid, sem, v_score, p_score))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out = []
        for is_exact, identity, pid, sem, v_score, p_score in scored[:num_results]:
            cpe, ver_score, in_dict, ref = self.index.resolve_version(pid, version)
            out.append({
                'score': identity + ver_score * cfg['version_weight'],
                'cpe': cpe,
                'title': self.index.display_titles[pid],
                'is_exact_product_match': is_exact,
                'version_in_dictionary': in_dict,
                'reference_cpe': ref,
                'score_breakdown': {
                    'semantic': sem,
                    'vendor': float(v_score),
                    'product': float(p_score),
                    'version': float(ver_score),
                },
            })
        out.sort(key=lambda r: (r['is_exact_product_match'], r['score']), reverse=True)
        return out

    # -- batch Excel -------------------------------------------------------

    def process_excel(self, excel_path, output_path=None):
        """Batch-process an Excel file. Returns True on success."""
        import pandas as pd
        try:
            df = pd.read_excel(excel_path)
            required = ['Vendor', 'Product', 'Version', 'CPE', 'Levenshtein score']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"Missing columns: {', '.join(missing)}")
                return False
            df['CPE'] = df['CPE'].astype(object)

            threshold = self.config['min_score_threshold']
            print(f"Processing {len(df)} rows (threshold={threshold})...")

            def _cell(v):
                return "" if pd.isna(v) else str(v)

            queries = [(_cell(r['Vendor']), _cell(r['Product']), _cell(r['Version']))
                       for _, r in df.iterrows()]
            all_results = self.search_batch(queries, num_results=1)
            found = 0
            for idx, res in zip(df.index, all_results):
                if res and res[0]['score'] > threshold:
                    df.at[idx, 'CPE'] = res[0]['cpe']
                    df.at[idx, 'Levenshtein score'] = res[0]['score']
                    found += 1
            print(f"Matched {found}/{len(df)} rows.")

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
                bd = r['score_breakdown']
                origin = "in NVD dictionary" if r['version_in_dictionary'] else "version substituted"
                print(f"- {r['cpe']}  [{origin}]")
                print(f"  Score: {r['score']:.4f}  "
                      f"(semantic={bd['semantic']:.3f}, vendor={bd['vendor']:.3f}, "
                      f"product={bd['product']:.3f}, version={bd['version']:.3f})")
                if r['reference_cpe'] and r['reference_cpe'] != r['cpe']:
                    print(f"  Based on: {r['reference_cpe']}")
                if r['title']:
                    print(f"  Title: {r['title']}")
                print()
        else:
            print(f"No results for '{vendor} {product} {version}'.")


def main():
    parser = argparse.ArgumentParser(description="CPE Matcher")
    parser.add_argument("-data", "--input", dest="data", help="Excel file path", type=str, default=None)
    parser.add_argument("-output", "--output", dest="output", help="Excel output path", type=str, default=None)
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Recompute all product embeddings")
    parser.add_argument("--use-mini-llm", action="store_true",
                        help="Prefer the fallback model over the default one")
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
