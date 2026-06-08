"""Human review queue for face-voice binding decisions.

Cases requiring human review:
- top1/top2 margin too small
- voice match high but face binding weak
- face match high but voice match weak
- only visible face, no ASD
- multi-face scene, ASD scores close
- global merge would affect many historical records
- same Person has conflicting evidence
"""
import json
from pathlib import Path


def create_review_task(
    task_type: str,
    face_cluster_id: str = None,
    speaker_cluster_id: str = None,
    candidate_person_id: str = None,
    evidence_summary: dict = None,
    conflict_id: str = None,
) -> dict:
    """Create a human review task."""
    return {
        "type": task_type,
        "face_cluster_id": face_cluster_id,
        "speaker_cluster_id": speaker_cluster_id,
        "candidate_person_id": candidate_person_id,
        "evidence_summary": evidence_summary or {},
        "conflict_id": conflict_id,
        "status": "pending",
        "reviewer": None,
        "decision": None,
        "notes": None,
    }


def generate_review_tasks(conflicts_path: str = None,
                           evidence_path: str = None,
                           output_path: str = None) -> list:
    """Generate review tasks from conflicts and evidence."""
    tasks = []
    
    # From conflicts
    if conflicts_path and Path(conflicts_path).exists():
        with open(conflicts_path) as f:
            conflicts = json.load(f)
        for c in conflicts:
            task = create_review_task(
                task_type="resolve_conflict",
                conflict_id=c["conflict_id"],
                evidence_summary={"conflict_type": c["type"], "severity": c["severity"]},
            )
            tasks.append(task)
    
    # From ambiguous evidence
    if evidence_path and Path(evidence_path).exists():
        with open(evidence_path) as f:
            evidence = json.load(f)
        for ev in evidence:
            if ev.get("status") == "Uncertain":
                task = create_review_task(
                    task_type="confirm_binding",
                    face_cluster_id=ev.get("face_cluster_id"),
                    speaker_cluster_id=ev.get("speaker_cluster_id"),
                    evidence_summary={
                        "evidence_type": ev.get("evidence_type"),
                        "duration_ms": ev.get("duration_ms"),
                        "visible_face_fraction": ev.get("visible_face_fraction"),
                    },
                )
                tasks.append(task)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(tasks, f, indent=2)
    
    print(f"Review tasks: {len(tasks)}")
    for t in tasks:
        print(f"  {t['type']}: {t['status']}")
    
    return tasks


if __name__ == "__main__":
    import sys
    conflicts = sys.argv[1] if len(sys.argv) > 1 else "output/conflicts.json"
    evidence = sys.argv[2] if len(sys.argv) > 2 else "output/binding_evidence_v2.json"
    output = sys.argv[3] if len(sys.argv) > 3 else "output/review_tasks.json"
    generate_review_tasks(conflicts, evidence, output)
