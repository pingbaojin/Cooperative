"""ROS config topic defaults for single vs multi partner."""

import unittest

from cooperative_link.ros_config import load_ros_link_config


class TestRosConfigTopics(unittest.TestCase):
    def test_single_partner_flat_topics(self) -> None:
        cfg = {
            "cooperative_link_multi": {"enabled": False},
            "cooperative_link_ros": {},
        }
        ros = load_ros_link_config(cfg)
        self.assertEqual(ros.pub_relative_pose_nav, "/cooperative_target/relative_pose_nav")
        self.assertNotIn("{id}", ros.pub_relative_pose_nav)

    def test_multi_partner_id_topics(self) -> None:
        cfg = {
            "cooperative_link_multi": {
                "enabled": True,
                "partners": [{"id": 1, "nav_path": "a.txt"}],
            },
            "cooperative_link_ros": {},
        }
        ros = load_ros_link_config(cfg)
        self.assertEqual(
            ros.pub_relative_pose_nav, "/cooperative_target/{id}/relative_pose_nav"
        )
        self.assertEqual(ros.pub_link_state, "/cooperative_target/{id}/link_state")


if __name__ == "__main__":
    unittest.main()
