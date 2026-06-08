"""Conflict records for face-voice binding evidence.

When evidence contradicts, record the conflict instead of hard-excluding.
Uses evidence graph, not hard exclusion.

Conflict types:
- voice_to_multiple_faces: same voice bound to multiple face clusters
- face_to_multiple_voices: same face bound to multiple voice clusters
- asd_voice_mismatch: ASD evidence contradicts voice similarity
- face_voice_contradiction: face match says one person, voice match says another
"""
import json
from pathlib import Path


CONFLICT_TYPES = [
    "voice_to_multiple_faces",
    "face_to_multiple_voices",
    "asd_voice_mismatch",
    "face_voice_contradiction",
]

SEVERITY_LEVELS = ["low", "medium", "high"]


def detect_conflicts(evidence_path: str, output_path: str = None) -> list:
    """Detect conflicts in binding evidence.
    
    Scans evidence for:
    - Same face_cluster bound to multiple speaker_clusters
    - Same speaker_cluster bound to multiple face_clusters
    """
    with open(evidence_path) as f:
        evidence = json.load(f)
    
    conflicts = []
    conflict_id = 0
    
    # Build face → speakers mapping
    face_to_speakers = {}
    speaker_to_faces = {}
    
    for ev in evidence:
        fc = ev.get("face_cluster_id", "")
        sc = ev.get("speaker_cluster_id", "")
        
        if fc and sc:
            face_to_speakers.setdefault(fc, set()).add(sc)
            speaker_to_faces.setdefault(sc, set()).add(fc)
    
    # Detect voice_to_multiple_faces
    for sc, faces in speaker_to_faces.items():
        if len(faces) > 1:
            conflicts.append({
                "conflict_id": f"conf_{conflict_id}",
                "type": "voice_to_multiple_faces",
                "entities": {
                    "speaker_cluster_id": sc,
                    "candidate_face_cluster_ids": list(faces),
                },
                "severity": "high" if len(faces) > 2 else "medium",
                "status": "needs_review",
            })
            conflict_id += 1
    
    # Detect face_to_multiple_voices
    for fc, speakers in face_to_speakers.items():
        if len(speakers) > 1:
            conflicts.append({
                "conflict_id": f"conf_{conflict_id}",
                "type": "face_to_multiple_voices",
                "entities": {
                    "face_cluster_id": fc,
                    "candidate_speaker_cluster_ids": list(speakers),
                },
                "severity": "high" if len(speakers) > 2 else "medium",
                "status": "needs_review",
            })
            conflict_id += 1
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(conflicts, f, indent=2)
    
    print(f"Conflicts detected: {len(conflicts)}")
    for c in conflicts:
        print(f"  {c['conflict_id']}: {c['type']} (severity={c['severity']})")
    
    return conflicts


if __name__ == "__main__":
    import sys
    evidence = sys.argv[1] if len(sys.argv) > 1 else "output/binding_evidence_v2.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "output/conflicts.json"
    detect_conflicts(evidence, output)
