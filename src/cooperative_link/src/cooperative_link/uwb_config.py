"""Parse cooperative_link_uwb section from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from cooperative_link.partner_config import load_multi_link_config


@dataclass
class UwbConfig:
    enabled: bool = True
    host_agent_id: int = 0
    from_agent_id: int = 0
    to_agent_id: int = 1
    pub_uwb_range: str = "/cooperative_uwb/range"
    pub_uwb_ranges: str = "/cooperative_uwb/ranges"
    pub_uwb_range_per_partner: str = "/cooperative_uwb/{id}/range"
    publish_per_partner_topics: bool = True
    publish_legacy_single: bool = True
    range_noise_std: float = 0.0
    publish_only_when_nav_ok: bool = True
    skip_invalid: bool = False
    source_label: str = "nav_sim"


def load_uwb_config(cfg: Dict[str, Any]) -> UwbConfig:
    raw = cfg.get("cooperative_link_uwb") or {}
    multi = load_multi_link_config(cfg)
    host_id = int(raw.get("host_agent_id", multi.host_agent_id))
    to_id = int(raw.get("to_agent_id", 1))
    if multi.enabled and multi.partners:
        to_id = multi.partners[0].partner_id
    return UwbConfig(
        enabled=bool(raw.get("enabled", True)),
        host_agent_id=host_id,
        from_agent_id=int(raw.get("from_agent_id", host_id)),
        to_agent_id=to_id,
        pub_uwb_range=str(raw.get("pub_uwb_range", "/cooperative_uwb/range")),
        pub_uwb_ranges=str(raw.get("pub_uwb_ranges", "/cooperative_uwb/ranges")),
        pub_uwb_range_per_partner=str(
            raw.get("pub_uwb_range_per_partner", "/cooperative_uwb/{id}/range")
        ),
        publish_per_partner_topics=bool(raw.get("publish_per_partner_topics", True)),
        publish_legacy_single=bool(
            raw.get(
                "publish_legacy_single",
                not (multi.enabled and len(multi.partners) > 0),
            )
        ),
        range_noise_std=float(raw.get("range_noise_std", 0.0)),
        publish_only_when_nav_ok=bool(raw.get("publish_only_when_nav_ok", True)),
        skip_invalid=bool(raw.get("skip_invalid", False)),
        source_label=str(raw.get("source_label", "nav_sim")),
    )
