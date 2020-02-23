# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Tests for cloud_lib."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import unittest
#
# import mock
# import requests
#
# from official.utils.logs import cloud_lib
#
#
# class CloudLibTest(unittest.TestCase):
#
#   @mock.patch("requests.get")
#   def test_on_gcp(self, mock_requests_get):
#     mock_response = mock.MagicMock()
#     mock_requests_get.return_value = mock_response
#     mock_response.status_code = 200
#
#     self.assertEqual(cloud_lib.on_gcp(), True)
#
#   @mock.patch("requests.get")
#   def test_not_on_gcp(self, mock_requests_get):
#     mock_requests_get.side_effect = requests.exceptions.ConnectionError()
#
#     self.assertEqual(cloud_lib.on_gcp(), False)
#
#
# if __name__ == "__main__":
#   unittest.main()
