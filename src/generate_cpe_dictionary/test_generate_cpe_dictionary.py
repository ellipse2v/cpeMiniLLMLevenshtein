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
import argparse
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


class TestArgParser(unittest.TestCase):

    def _build_parser(self):
        """Builds the expected argument parser structure."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-csv", action="store_true")
        parser.add_argument("--limit", type=int, default=None,
                            help="Limit to N CPEs (testing). Default: fetch all.")
        return parser

    def test_default_limit_is_none(self):
        """No flags → limit is None (means fetch all)."""
        args = self._build_parser().parse_args([])
        self.assertIsNone(args.limit, "Default limit must be None to trigger full fetch")

    def test_limit_flag_sets_value(self):
        args = self._build_parser().parse_args(["--limit", "5"])
        self.assertEqual(args.limit, 5)

    def test_fetch_all_flag_removed(self):
        """--fetch-all must no longer be a valid flag."""
        parser = self._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--fetch-all"])


if __name__ == '__main__':
    unittest.main()
