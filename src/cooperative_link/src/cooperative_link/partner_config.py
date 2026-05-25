"""Multi-partner configuration for cooperative link / UWB."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PartnerSpec:
    partner_id: int
    name: str
    nav_path: str = ""
    navsat_topic: str = ""
    yaw_topic: str = ""


@dataclass
class MultiLinkConfig:
    enabled: bool = False
    host_agent_id: int = 0
    partners: List[PartnerSpec] = field(default_factory=list)


def load_multi_link_config(cfg: Dict[str, Any]) -> MultiLinkConfig:
    raw = cfg.get("cooperative_link_multi") or {}
    partners_raw = raw.get("partners") or []
    partners: List[PartnerSpec] = []
    for p in partners_raw:
        pid = int(p.get("id", p.get("partner_id", 0)))
        if pid <= 0:
            continue
        partners.append(
            PartnerSpec(
                partner_id=pid,
                name=str(p.get("name", f"partner_{pid}")),
                nav_path=str(p.get("nav_path", "")),
                navsat_topic=str(
                    p.get("navsat_topic", f"/cooperative_partner/{pid}/navsat")
                ),
                yaw_topic=str(
                    p.get("yaw_topic", f"/cooperative_partner/{pid}/yaw_deg")
                ),
            )
        )
    return MultiLinkConfig(
        enabled=bool(raw.get("enabled", False)) and len(partners) > 0,
        host_agent_id=int(raw.get("host_agent_id", 0)),
        partners=partners,
    )
