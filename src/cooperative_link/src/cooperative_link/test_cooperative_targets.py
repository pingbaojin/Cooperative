"""Unit tests for cooperative target registry."""

import unittest

from cooperative_link.targets_registry import (
    build_cooperative_targets,
    nav_valid_from_poses,
    partner_ids_from_multi,
)
from cooperative_link.partner_config import PartnerSpec


class TestCooperativeTargets(unittest.TestCase):
    def test_single_partner(self) -> None:
        host_id, count, ids, nav_valid, link_id = build_cooperative_targets(
            host_id=0,
            link_target_id=1,
            partner_ids=[1],
            nav_valid_by_id={1: True},
        )
        self.assertEqual(host_id, 0)
        self.assertEqual(count, 1)
        self.assertEqual(ids, [1])
        self.assertEqual(nav_valid, [True])
        self.assertEqual(link_id, 1)

    def test_two_partners_sorted(self) -> None:
        host_id, count, ids, nav_valid, link_id = build_cooperative_targets(
            host_id=0,
            link_target_id=1,
            partner_ids=[2, 1],
            nav_valid_by_id={1: True, 2: False},
        )
        self.assertEqual(count, 2)
        self.assertEqual(ids, [1, 2])
        self.assertEqual(nav_valid, [True, False])
        self.assertEqual(link_id, 1)
        self.assertEqual(host_id, 0)

    def test_nav_valid_from_poses(self) -> None:
        poses = {1: {"lat_deg": 30.0}, 2: None}
        valid = nav_valid_from_poses([1, 2], poses)
        self.assertEqual(valid, {1: True, 2: False})

    def test_partner_ids_from_multi(self) -> None:
        partners = [
            PartnerSpec(partner_id=2, name="b"),
            PartnerSpec(partner_id=1, name="a"),
        ]
        self.assertEqual(partner_ids_from_multi(partners), [2, 1])


if __name__ == "__main__":
    unittest.main()
