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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.cpe_matcher.cpe_matcher import CPEMatcher, parse_cpe_name


def _has_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_sentence_transformers(), "sentence-transformers not installed")
class TestUserCPERequest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matcher = CPEMatcher()
        cls.matcher.load_data()
        if not cls.matcher.cpe_items or cls.matcher.embeddings is None:
            raise Exception("Could not load CPE data.")

    def test_user_specific_query(self):
        results = self.matcher.search('microsoft', 'windows_11', '2147562', num_results=10)
        self.assertTrue(results)
        for i, r in enumerate(results[:10]):
            _, p, _ = parse_cpe_name(r['cpe'])
            bd = r['score_breakdown']
            print(f"{i+1}. {r['cpe']}  score={r['score']:.4f} "
                  f"(sem={bd['semantic']:.3f}, v={bd['vendor']:.3f}, "
                  f"p={bd['product']:.3f}, ver={bd['version']:.3f})")


if __name__ == '__main__':
    unittest.main()
