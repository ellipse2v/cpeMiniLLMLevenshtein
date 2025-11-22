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

# Add src to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.cpe_matcher.cpe_matcher import (
    load_cpe_data,
    prepare_cpe_data,
    save_cpe_data,
    find_closest_cpes,
    parse_cpe_name,
    config,
    device,
    model
)

class TestUserCPERequest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load all necessary data once for all tests."""
        print("Setting up test class...")
        should_regenerate = config['force_regenerate'] or not os.path.exists(config['pickle_filepath']) or not os.path.exists(config['embeddings_filepath'])

        if not should_regenerate:
            print("Loading existing data for tests...")
            cls.cpe_items, cls.titles, cls.embeddings, cls.product_map = load_cpe_data(config['pickle_filepath'], config['embeddings_filepath'])
            if not cls.product_map:
                print("Product map missing. Forcing regeneration for tests.")
                should_regenerate = True
        
        if should_regenerate:
            print("Preparing new CPE data for tests...")
            cls.cpe_items, cls.titles, cls.embeddings, cls.product_map = prepare_cpe_data(config['xml_filepath'], model, device)
            if cls.cpe_items is not None:
                save_cpe_data(cls.cpe_items, cls.titles, cls.embeddings, cls.product_map, config['pickle_filepath'], config['embeddings_filepath'])

        if not cls.cpe_items or cls.embeddings is None or not cls.product_map:
            raise Exception("Could not load or prepare CPE data for tests.")

        print("Test class setup complete.")

    def test_user_specific_query(self):
        """
        Tests with the user's specific query to check architecture handling.
        """
        vendor = 'microsoft'
        product = 'windows_11'
        version = '2147562'

        print(f"\n--- Running test: test_user_specific_query with query: {vendor} {product} {version} ---")
        
        results = find_closest_cpes(
            vendor=vendor,
            product=product,
            version=version,
            cpe_items=self.cpe_items,
            titles=self.titles,
            embeddings=self.embeddings,
            model=model,
            device=device,
            product_map=self.product_map,
            num_results=10
        )

        self.assertTrue(results, "Should return at least one result")

        print("\nTop 10 results from the user's test query:")
        for i, (score, cpe, title) in enumerate(results[:10]):
            _, p_name, _ = parse_cpe_name(cpe)
            p_name_normalized = p_name.replace('_', ' ')
            print(f"{i+1}. CPE: {cpe}, Product: {p_name_normalized}, Score: {score:.4f}")

if __name__ == '__main__':
    unittest.main()
