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

import unittest
import os
import sys
import gzip
import json
import hashlib
import tempfile
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


class TestModuleImport(unittest.TestCase):
    """Importing cpe_matcher must not trigger model loading."""

    def test_import_does_not_load_model(self):
        try:
            import torch
            _has_torch = True
        except ImportError:
            _has_torch = False

        for key in list(sys.modules.keys()):
            if 'cpe_matcher' in key:
                del sys.modules[key]

        mem_before = torch.cuda.memory_allocated() if (_has_torch and torch.cuda.is_available()) else 0
        import src.cpe_matcher.cpe_matcher as m
        mem_after = torch.cuda.memory_allocated() if (_has_torch and torch.cuda.is_available()) else 0

        self.assertFalse(hasattr(m, 'model'), "Module must not have a top-level 'model'")
        self.assertEqual(mem_before, mem_after, "GPU memory must not change on import")


class TestJSONCache(unittest.TestCase):
    """JSON+gzip+sha256 round-trip and integrity detection."""

    def test_round_trip(self):
        cpe_items = ["cpe:2.3:a:test:product:1.0:*:*:*:*:*:*:*"]
        titles = ["Test Product 1.0"]
        product_map = {"product": [0]}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, 'cpe_data.json.gz')
            hash_path = json_path + '.sha256'

            data = {'cpe_items': cpe_items, 'titles': titles, 'product_map': product_map}
            payload = json.dumps(data).encode('utf-8')
            with gzip.open(json_path, 'wb') as f:
                f.write(payload)
            with open(hash_path, 'w') as f:
                f.write(hashlib.sha256(payload).hexdigest())
            np.save(os.path.join(tmpdir, 'emb.npy'), np.zeros((1, 4), dtype=np.float32))

            with gzip.open(json_path, 'rb') as f:
                raw = f.read()
            with open(hash_path) as f:
                expected = f.read().strip()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
            loaded = json.loads(raw.decode('utf-8'))
            self.assertEqual(loaded['cpe_items'], cpe_items)

    def test_tampered_cache_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, 'cpe_data.json.gz')
            hash_path = json_path + '.sha256'

            data = {'cpe_items': ['cpe:2.3:a:x:y:1:*:*:*:*:*:*:*'], 'titles': [''], 'product_map': {}}
            payload = json.dumps(data).encode('utf-8')
            with gzip.open(json_path, 'wb') as f:
                f.write(payload)
            with open(hash_path, 'w') as f:
                f.write("deadbeef" * 8)

            with gzip.open(json_path, 'rb') as f:
                raw = f.read()
            with open(hash_path) as f:
                stored = f.read().strip()
            self.assertNotEqual(hashlib.sha256(raw).hexdigest(), stored)


class TestVersionSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.cpe_matcher.cpe_matcher import version_similarity
        cls.version_similarity = staticmethod(version_similarity)

    def test_identical(self):
        self.assertAlmostEqual(self.version_similarity("1.2.3", "1.2.3"), 1.0)

    def test_one_patch_off(self):
        score = self.version_similarity("10.0.1", "10.0.2")
        self.assertGreater(score, 0.8)

    def test_major_diff(self):
        score = self.version_similarity("1.0", "10.0")
        self.assertLess(score, 0.5)

    def test_empty(self):
        self.assertEqual(self.version_similarity("", "1.0"), 0.0)
        self.assertEqual(self.version_similarity("1.0", ""), 0.0)

    def test_non_numeric_fallback(self):
        self.assertAlmostEqual(self.version_similarity("sp1", "sp1"), 1.0)

    def test_wildcard(self):
        self.assertEqual(self.version_similarity("10.0.1", "*"), 0.5)


class TestCPEMatcher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.cpe_matcher.cpe_matcher import CPEMatcher
        cls.matcher = CPEMatcher()
        cls.matcher.load_data()
        if not cls.matcher.cpe_items or cls.matcher.embeddings is None:
            raise Exception("Could not load CPE data.")

    def test_windows11_is_prioritized_and_found(self):
        from src.cpe_matcher.cpe_matcher import parse_cpe_name
        results = self.matcher.search('microsoft', 'windows_11', '22000', num_results=10)
        self.assertTrue(results)
        for r in results[:5]:
            _, p, _ = parse_cpe_name(r['cpe'])
            self.assertEqual(p, 'windows_11', f"Expected windows_11, got {p}")
        _, top_product, _ = parse_cpe_name(results[0]['cpe'])
        self.assertEqual(top_product, 'windows_11')

    def test_search_returns_score_breakdown(self):
        results = self.matcher.search('microsoft', 'windows_11', '22000', num_results=1)
        self.assertTrue(results)
        bd = results[0]['score_breakdown']
        for key in ('semantic', 'vendor', 'product', 'version'):
            self.assertIn(key, bd)
            self.assertIsInstance(bd[key], float)

    def test_min_score_threshold_in_config(self):
        t = self.matcher.config['min_score_threshold']
        self.assertIsInstance(t, float)
        self.assertGreater(t, 0.0)
        self.assertLessEqual(t, 1.0)


class TestFAISSSearch(unittest.TestCase):

    def test_faiss_top1_matches_brute_force(self):
        """FAISS inner-product top-1 must agree with brute-force cosine."""
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss-cpu not installed")

        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos

        np.random.seed(42)
        n, dim = 200, 64
        vecs = np.random.rand(n, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_norm = vecs / norms

        query = np.random.rand(1, dim).astype(np.float32)
        query_norm = query / np.linalg.norm(query)

        bf_top = int(sklearn_cos(query_norm, vecs_norm)[0].argmax())

        index = faiss.IndexFlatIP(dim)
        index.add(vecs_norm)
        _, faiss_ids = index.search(query_norm, 1)
        faiss_top = int(faiss_ids[0][0])

        self.assertEqual(bf_top, faiss_top)


if __name__ == '__main__':
    unittest.main()
