"""Unit tests for topic name helpers."""

import unittest

from cooperative_link.topic_names import partner_topic


class TestTopicNames(unittest.TestCase):
    def test_partner_topic(self) -> None:
        t = partner_topic("/cooperative_target/{id}/relative_pose_nav", 2)
        self.assertEqual(t, "/cooperative_target/2/relative_pose_nav")


if __name__ == "__main__":
    unittest.main()
