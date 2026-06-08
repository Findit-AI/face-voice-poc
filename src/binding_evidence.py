"""BindingEvidence — refactored face-voice binding evidence.

Evidence types:
- temporal_overlap: face visible while voice present (weak)
- only_visible_face_candidate: only one face visible (weak, NOT high confidence)
- active_speaker_detection: ASD score from TalkNet/lip-sync (strong, future)

Rules:
- All evidence has asserts_identity = False
- Only GlobalIdentityResolver can create Person links
- single_visible_face ≠ high confidence binding
"""
import json
from pathlib import Path


EVIDENCE_TYPES = {
    "temporal_overlap": "weak",
    "only_visible_face_candidate": "weak",
    "active_speaker_detection": "strong",
}

BINDING_STATUS = ["Unspecified", "Candidate", "Accepted", "Rejected", "Uncertain"]


def create_binding_evidence(
    face_cluster_id: str,
    speaker_cluster_id: str,
    video_id: str,
    evidence_type: str,
    segment_start_ms: int,
    segment_end_ms: int,
    overlap_duration_ms: int,
    visible_face_fraction: float,
    num_visible_faces: int,
    raw_score: float = None,
    score_margin: float = None,
    calibrated_probability: float = None,
) -> dict:
    """Create a BindingEvidence record.
    
    Always sets asserts_identity = False.
    Always starts as Candidate status.
    """
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(f"Unknown evidence type: {evidence_type}")
    
    return {
        "face_cluster_id": face_cluster_id,
        "speaker_cluster_id": speaker_cluster_id,
        "video_id": video_id,
        "evidence_type": evidence_type,
        "segment_start_ms": segment_start_ms,
        "segment_end_ms": segment_end_ms,
        "duration_ms": overlap_duration_ms,
        "raw_score": raw_score,
        "score_margin": score_margin,
        "calibrated_probability": calibrated_probability,
        "visible_face_fraction": visible_face_fraction,
        "num_visible_faces": num_visible_faces,
        "status": "Candidate",
        "asserts_identity": False,
        "strength": EVIDENCE_TYPES[evidence_type],
    }


def refactor_candidates_to_evidence(candidates_path: str, output_path: str,
                                      video_id: str = "unknown") -> list:
    """Convert old candidate format to new BindingEvidence format.
    
    Key changes:
    - single_face → only_visible_face_candidate (weak, NOT high confidence)
    - multi_face_needs_asd → temporal_overlap (weak, needs ASD to strengthen)
    - All have asserts_identity = False
    """
    with open(candidates_path) as f:
        candidates = json.load(f)
    
    evidence = []
    for c in candidates:
        # Determine evidence type
        if c.get("method") == "single_face":
            ev_type = "only_visible_face_candidate"
        elif c.get("method") == "multi_face_needs_asd":
            ev_type = "temporal_overlap"
        else:
            ev_type = "temporal_overlap"
        
        ev = create_binding_evidence(
            face_cluster_id=c.get("face_cluster_id", ""),
            speaker_cluster_id=c.get("speaker_cluster_id", c.get("speaker_id", "")),
            video_id=video_id,
            evidence_type=ev_type,
            segment_start_ms=int(c.get("segment_start", 0) * 1000),
            segment_end_ms=int(c.get("segment_end", 0) * 1000),
            overlap_duration_ms=c.get("overlap_duration_ms", c.get("duration_ms", 0)),
            visible_face_fraction=c.get("visible_face_fraction", 0.0),
            num_visible_faces=1 if c.get("method") == "single_face" else 2,
            raw_score=c.get("asd_score"),
            score_margin=c.get("asd_margin"),
        )
        evidence.append(ev)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(evidence, f, indent=2)
    
    # Summary
    by_type = {}
    for ev in evidence:
        t = ev["evidence_type"]
        by_type[t] = by_type.get(t, 0) + 1
    
    print(f"BindingEvidence: {len(evidence)} records")
    for t, count in by_type.items():
        strength = EVIDENCE_TYPES[t]
        print(f"  {t} ({strength}): {count}")
    print(f"  All asserts_identity = False")
    
    return evidence


if __name__ == "__main__":
    import sys
    
    candidates = sys.argv[1] if len(sys.argv) > 1 else "output/candidates.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "output/binding_evidence_v2.json"
    
    refactor_candidates_to_evidence(candidates, output)
