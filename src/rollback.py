"""Rollback / split mechanism for identity merges.

Every merge is recorded as an append-only event.
Rollback appends a split event, never deletes merge events.

All state is recomputable from the event log + source embeddings.
"""
import json
import time
from pathlib import Path


def record_merge(
    from_entities: list,
    to_person_id: str,
    reason: str,
    evidence_ids: list,
    confidence: float,
    created_by: str = "auto_resolver_v1",
) -> dict:
    """Record an identity merge event."""
    return {
        "event_type": "merge",
        "event_id": f"merge_{int(time.time() * 1000)}",
        "from_entities": from_entities,
        "to_person_id": to_person_id,
        "reason": reason,
        "evidence_ids": evidence_ids,
        "confidence": confidence,
        "created_by": created_by,
        "created_at": int(time.time() * 1000),
    }


def record_split(
    from_person_id: str,
    to_entities: list,
    reason: str,
    triggered_by_event_id: str = None,
    created_by: str = "human_review",
) -> dict:
    """Record an identity split event (rollback)."""
    return {
        "event_type": "split",
        "event_id": f"split_{int(time.time() * 1000)}",
        "from_person_id": from_person_id,
        "to_entities": to_entities,
        "reason": reason,
        "triggered_by_event_id": triggered_by_event_id,
        "created_by": created_by,
        "created_at": int(time.time() * 1000),
    }


def rollback_merge(merge_event: dict, reason: str = "error_correction") -> dict:
    """Create a split event that reverses a merge."""
    return record_split(
        from_person_id=merge_event["to_person_id"],
        to_entities=merge_event["from_entities"],
        reason=reason,
        triggered_by_event_id=merge_event["event_id"],
    )


if __name__ == "__main__":
    # Demo: record merge then rollback
    merge = record_merge(
        from_entities=["person_018", "gvoice_034"],
        to_person_id="person_002",
        reason="face_voice_binding",
        evidence_ids=["bev_001", "bev_017"],
        confidence=0.91,
    )
    print("Merge event:", json.dumps(merge, indent=2))
    
    split = rollback_merge(merge)
    print("\nSplit event (rollback):", json.dumps(split, indent=2))
