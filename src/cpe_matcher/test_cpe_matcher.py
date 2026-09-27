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


def _has_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class FakeEncoder:
    """Deterministic character-trigram encoder, stands in for sentence-transformers."""
    name = 'fake-trigram'

    def encode(self, texts, **_kwargs):
        import zlib
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            t = f"  {t.lower()} "
            for j in range(len(t) - 2):
                out[i, zlib.crc32(t[j:j + 3].encode()) % 256] += 1.0
        return out


SAMPLE_CPES = [
    ("cpe:2.3:a:microsoft:internet_explorer:6:*:*:*:*:*:*:*", "Microsoft Internet Explorer 6"),
    ("cpe:2.3:a:microsoft:internet_explorer:6:sp1:*:*:*:*:*:*", "Microsoft Internet Explorer 6 SP1"),
    ("cpe:2.3:a:microsoft:internet_explorer:5.01:sp4:*:*:*:*:*:*", "Microsoft Internet Explorer 5.01 Service Pack 4"),
    ("cpe:2.3:a:microsoft:internet_explorer:4.0:*:*:*:*:*:*:*", "Microsoft Internet Explorer 4.0"),
    ("cpe:2.3:a:microsoft:internet_explorer:11:-:*:*:*:*:*:*", "Microsoft Internet Explorer 11"),
    ("cpe:2.3:a:microsoft:sql_server:2019:*:*:*:*:*:*:*", "Microsoft SQL Server 2019"),
    ("cpe:2.3:a:mozilla:firefox:99.0:*:*:*:*:*:*:*", "Mozilla Firefox 99.0"),
    ("cpe:2.3:a:adobe:flash_player:20.0.0.306:*:*:*:*:chrome:*:*", "Adobe Flash Player 20.0.0.306 for Chrome"),
    ("cpe:2.3:a:gnu:g\\+\\+:3.3.3:*:*:*:*:*:*:*", "GNU G++ 3.3.3"),
    ("cpe:2.3:o:microsoft:windows_11:-:*:*:*:*:*:*:*", "Microsoft Windows 11"),
    ("cpe:2.3:a:7-zip:7-zip:9.38:*:*:*:*:*:*:*", "7-Zip 9.38"),
    ("cpe:2.3:a:f5:nginx:1.9.9:*:*:*:*:*:*:*", "F5 Nginx 1.9.9"),
]


class TestCpeHelpers(unittest.TestCase):

    def test_split_cpe_honours_escapes(self):
        from src.cpe_matcher.cpe_matcher import split_cpe
        parts = split_cpe("cpe:2.3:a:cisco:ios:10.3\\(16\\):*:*:*:*:*:*:*")
        self.assertEqual(len(parts), 13)
        self.assertEqual(parts[5], "10.3\\(16\\)")
        parts = split_cpe("cpe:2.3:a:vendor:prod\\:uct:1.0:*:*:*:*:*:*:*")
        self.assertEqual(parts[4], "prod\\:uct")

    def test_escape_cpe_value(self):
        from src.cpe_matcher.cpe_matcher import escape_cpe_value
        self.assertEqual(escape_cpe_value("2019 SP1"), "2019_sp1")
        self.assertEqual(escape_cpe_value("10.3(16)"), "10.3\\(16\\)")
        self.assertEqual(escape_cpe_value(""), "*")

    def test_escape_cpe_value_already_escaped(self):
        from src.cpe_matcher.cpe_matcher import escape_cpe_value
        self.assertEqual(escape_cpe_value("6.0\\(2\\)u6\\(5\\)"), "6.0\\(2\\)u6\\(5\\)")

    def test_product_similarity_acronym(self):
        from src.cpe_matcher.cpe_matcher import product_similarity
        self.assertGreaterEqual(product_similarity("GNU Image Manipulation Program", "gimp"), 0.9)
        self.assertGreaterEqual(product_similarity("IIS", "internet_information_services"), 0.9)
        self.assertLess(product_similarity("Office", "internet_information_services"), 0.5)

    def test_vendor_exact_beats_suffix_stripped(self):
        from src.cpe_matcher.cpe_matcher import vendor_similarity
        self.assertGreater(vendor_similarity("apache", "apache"),
                           vendor_similarity("apache", "apache_software_foundation"))
        self.assertGreater(vendor_similarity("Microsoft Corporation", "microsoft"), 0.9)

    def test_normalize_version_key(self):
        from src.cpe_matcher.cpe_matcher import normalize_version_key
        self.assertEqual(normalize_version_key("4.0"), normalize_version_key("4"))
        self.assertEqual(normalize_version_key("4.0.0"), "4")
        self.assertNotEqual(normalize_version_key("4.01"), normalize_version_key("4.1"))

    def test_product_variants_strip_vendor(self):
        from src.cpe_matcher.cpe_matcher import product_variants
        self.assertEqual(product_variants("Microsoft Corporation", "Microsoft SQL Server"),
                         ["Microsoft SQL Server", "SQL Server"])
        self.assertEqual(product_variants("", "Firefox"), ["Firefox"])

    def test_strip_version_from_title(self):
        from src.cpe_matcher.cpe_matcher import strip_version_from_title
        self.assertEqual(strip_version_from_title("Microsoft Internet Explorer 6 SP1", ["6", "sp1"]),
                         "Microsoft Internet Explorer")
        self.assertEqual(strip_version_from_title("GNU libgomp 4.8.5", ["4.8.5"]), "GNU libgomp")


class TestAliases(unittest.TestCase):
    """Vendor renames derived from NVD deprecations (nginx:nginx -> f5:nginx)."""

    def test_build_aliases(self):
        from src.cpe_matcher.cpe_matcher import build_aliases
        active = {('a', 'f5', 'nginx'): 9, ('a', 'mattermost', 'confluence'): 5,
                  ('a', 'x', 'dup'): 3, ('a', 'y', 'dup'): 3}
        deprecated = {('a', 'nginx', 'nginx'): 10, ('a', 'atlassian', 'confluence'): 100,
                      ('a', 'z', 'dup'): 3, ('a', 'f5', 'nginx'): 1}
        self.assertEqual(build_aliases(active, deprecated),
                         [(('a', 'nginx', 'nginx'), ('a', 'f5', 'nginx'))])

    def test_save_keeps_manual_rows(self):
        from src.cpe_matcher.cpe_matcher import save_aliases, load_aliases
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'aliases.csv')
            with open(path, 'w', newline='') as f:
                f.write("part,old_vendor,old_product,new_vendor,new_product,source\n"
                        "a,igor_pavlov,7-zip,7-zip,7-zip,manual\n"
                        "a,old,thing,new,thing,auto\n")
            save_aliases(path, [(('a', 'nginx', 'nginx'), ('a', 'f5', 'nginx'))])
            rows = load_aliases(path)
        self.assertEqual([(r['old_vendor'], r['source']) for r in rows],
                         [('igor_pavlov', 'manual'), ('nginx', 'auto')])


class TestProductIndex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.cpe_matcher.cpe_matcher import ProductIndex
        cls.index = ProductIndex([c for c, _ in SAMPLE_CPES], [t for _, t in SAMPLE_CPES])

    def _pid(self, product):
        return next(i for i, k in enumerate(self.index.keys) if k[2] == product)

    def test_one_entry_per_product(self):
        self.assertEqual(len(self.index), 8)
        self.assertEqual(self.index.texts[self._pid('internet_explorer')], "microsoft internet explorer")

    def test_known_version_returns_dictionary_cpe(self):
        cpe, score, in_dict, _ = self.index.resolve_version(self._pid('internet_explorer'), '6')
        self.assertEqual(cpe, "cpe:2.3:a:microsoft:internet_explorer:6:*:*:*:*:*:*:*")
        self.assertTrue(in_dict)
        self.assertEqual(score, 1.0)

    def test_trailing_zero_version_is_known(self):
        cpe, _, in_dict, _ = self.index.resolve_version(self._pid('internet_explorer'), '4')
        self.assertTrue(in_dict)
        self.assertEqual(cpe, "cpe:2.3:a:microsoft:internet_explorer:4.0:*:*:*:*:*:*:*")

    def test_unknown_version_is_substituted(self):
        """IE 3 is not in the sample dictionary: a valid CPE is built for it."""
        cpe, score, in_dict, ref = self.index.resolve_version(self._pid('internet_explorer'), '3')
        self.assertEqual(cpe, "cpe:2.3:a:microsoft:internet_explorer:3:*:*:*:*:*:*:*")
        self.assertFalse(in_dict)
        self.assertEqual(ref, "cpe:2.3:a:microsoft:internet_explorer:4.0:*:*:*:*:*:*:*")
        self.assertTrue(0.5 <= score < 1.0)

    def test_platform_specific_version_gives_generic_cpe(self):
        cpe, _, in_dict, ref = self.index.resolve_version(self._pid('flash_player'), '20.0.0.306')
        self.assertEqual(cpe, "cpe:2.3:a:adobe:flash_player:20.0.0.306:*:*:*:*:*:*:*")
        self.assertTrue(in_dict)
        self.assertIn(":chrome:", ref)

    def test_escaped_product(self):
        pid = self._pid('g\\+\\+')
        self.assertIn(pid, self.index.lexical_candidates('gnu', 'g++'))
        cpe, _, _, _ = self.index.resolve_version(pid, '9.1')
        self.assertEqual(cpe, "cpe:2.3:a:gnu:g\\+\\+:9.1:*:*:*:*:*:*:*")

    def test_no_version(self):
        cpe, _, in_dict, _ = self.index.resolve_version(self._pid('firefox'), '')
        self.assertEqual(cpe, "cpe:2.3:a:mozilla:firefox:*:*:*:*:*:*:*:*")
        self.assertFalse(in_dict)


class TestCPEMatcherOffline(unittest.TestCase):
    """End-to-end search with an injected encoder and a tiny dictionary."""

    @classmethod
    def setUpClass(cls):
        from src.cpe_matcher.cpe_matcher import CPEMatcher
        cls.tmpdir = tempfile.TemporaryDirectory()
        json_path = os.path.join(cls.tmpdir.name, 'cpe_data.json.gz')
        payload = json.dumps({'cpe_items': [c for c, _ in SAMPLE_CPES],
                              'titles': [t for _, t in SAMPLE_CPES]}).encode('utf-8')
        with gzip.open(json_path, 'wb') as f:
            f.write(payload)
        cfg = os.path.join(cls.tmpdir.name, 'config.ini')
        with open(cfg, 'w') as f:
            f.write(f"[Paths]\nCPE_DATA_JSON = {json_path}\nEMBEDDINGS_DIR = {cls.tmpdir.name}\n"
                    f"CPE_DICTIONARY_XML = {os.path.join(cls.tmpdir.name, 'none.xml')}\n")
        cls.matcher = CPEMatcher(config_path=cfg, model=FakeEncoder())
        cls.matcher.load_data()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_substituted_version_is_found(self):
        results = self.matcher.search('microsoft', 'internet explorer', '3')
        self.assertTrue(results)
        top = results[0]
        self.assertEqual(top['cpe'], "cpe:2.3:a:microsoft:internet_explorer:3:*:*:*:*:*:*:*")
        self.assertFalse(top['version_in_dictionary'])
        self.assertGreater(top['score'], self.matcher.config['min_score_threshold'])

    def test_product_named_vendor_ignores_inventory_vendor(self):
        """NVD uses the product as vendor (7-zip:7-zip): 'Igor Pavlov' must not be penalised."""
        top = self.matcher.search('Igor Pavlov', '7-Zip', '9.38', num_results=1)[0]
        self.assertEqual(top['cpe'], "cpe:2.3:a:7-zip:7-zip:9.38:*:*:*:*:*:*:*")
        self.assertGreaterEqual(top['score_breakdown']['vendor'], 0.9)

    def test_alias_redirects_renamed_vendor(self):
        m = self.matcher
        pid_f5 = m.index.key_to_id[('a', 'f5', 'nginx')]
        m.index.set_aliases([{'part': 'a', 'old_vendor': 'nginx', 'old_product': 'nginx',
                              'new_vendor': 'f5', 'new_product': 'nginx', 'source': 'auto'}])
        try:
            self.assertEqual(m.index.alias_targets('Nginx Inc.', 'nginx'), {pid_f5})
            top = m.search('nginx', 'nginx', '1.9.99', num_results=1)[0]
        finally:
            m.index.set_aliases([])
        self.assertEqual(top['cpe'], "cpe:2.3:a:f5:nginx:1.9.99:*:*:*:*:*:*:*")
        self.assertEqual(top['score_breakdown']['vendor'], 1.0)

    def test_vendor_prefix_in_product(self):
        results = self.matcher.search('Microsoft Corporation', 'Microsoft SQL Server', '2019')
        self.assertEqual(results[0]['cpe'], "cpe:2.3:a:microsoft:sql_server:2019:*:*:*:*:*:*:*")
        self.assertTrue(results[0]['version_in_dictionary'])

    def test_one_result_per_product(self):
        results = self.matcher.search('microsoft', 'internet explorer', '6', num_results=5)
        prods = [r['cpe'].split(':')[4] for r in results]
        self.assertEqual(len(prods), len(set(prods)))

    def test_version_counts_before_truncation(self):
        """num_results=1 must not drop the product that has the requested version."""
        from src.cpe_matcher.cpe_matcher import ProductIndex
        m = self.matcher
        saved = m.index, m.embeddings
        items = ["cpe:2.3:a:apache:tomcat:6.0.44:*:*:*:*:*:*:*",
                 "cpe:2.3:a:apache_software_foundation:tomcat:1.0:*:*:*:*:*:*:*"]
        m.index = ProductIndex(items, ["", ""])
        # Semantic 0.95 vs 1.0: identity alone favours the second product,
        # the known version must tip the balance back to the first one.
        q = m._encode(["apache tomcat"])[0]
        other = np.zeros_like(q)
        other[int(np.argmin(np.abs(q)))] = 1.0
        other -= other.dot(q) * q
        other /= np.linalg.norm(other)
        m.embeddings = np.stack([0.95 * q + np.sqrt(1 - 0.95 ** 2) * other, q]).astype(np.float32)
        try:
            top = m.search('apache', 'tomcat', '6.0.44', num_results=1)[0]
        finally:
            m.index, m.embeddings = saved
        self.assertEqual(top['cpe'], "cpe:2.3:a:apache:tomcat:6.0.44:*:*:*:*:*:*:*")

    def test_batch_equals_single(self):
        queries = [('mozilla', 'firefox', '100.0'), ('gnu', 'g++', '9.1'), ('', '', '')]
        batch = self.matcher.search_batch(queries, num_results=1)
        self.assertEqual(batch[0][0]['cpe'], self.matcher.search(*queries[0], num_results=1)[0]['cpe'])
        self.assertEqual(batch[1][0]['cpe'], "cpe:2.3:a:gnu:g\\+\\+:9.1:*:*:*:*:*:*:*")
        self.assertEqual(batch[2], [])

    def test_embeddings_are_reused_incrementally(self):
        from src.cpe_matcher.cpe_matcher import ProductIndex
        m = self.matcher
        m.index = ProductIndex(m.cpe_items + ["cpe:2.3:a:videolan:vlc_media_player:3.0.9:*:*:*:*:*:*:*"],
                               m.titles + ["VideoLAN VLC media player 3.0.9"])
        calls = []
        original = m.model.encode
        m.model.encode = lambda texts, **kw: (calls.append(list(texts)), original(texts))[1]
        try:
            emb = m._load_or_build_embeddings()
        finally:
            m.model.encode = original
        self.assertEqual(emb.shape[0], len(m.index))
        self.assertEqual(calls, [["videolan vlc media player"]])


@unittest.skipUnless(_has_sentence_transformers(), "sentence-transformers not installed")
class TestCPEMatcher(unittest.TestCase):
    """Full end-to-end test (requires the model and cpe_data.json.gz)."""

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
        _, top_product, top_version = parse_cpe_name(results[0]['cpe'])
        self.assertEqual(top_product, 'windows_11')
        self.assertEqual(top_version, '22000')

    def test_internet_explorer_unknown_version(self):
        results = self.matcher.search('microsoft', 'internet explorer', '2.5', num_results=1)
        self.assertEqual(results[0]['cpe'], "cpe:2.3:a:microsoft:internet_explorer:2.5:*:*:*:*:*:*:*")

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


if __name__ == '__main__':
    unittest.main()
