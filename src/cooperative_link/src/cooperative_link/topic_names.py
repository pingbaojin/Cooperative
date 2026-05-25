"""ROS topic name helpers for per-partner cooperative link outputs."""

from __future__ import annotations


def partner_topic(template: str, partner_id: int) -> str:
    """Format a topic template containing ``{id}`` for one partner."""
    return template.format(id=int(partner_id))
