"""GlobalIdentityResolver — resolves person identity from evidence.

Proposal-only mode: generates merge proposals, never auto-merges.
Requires human review or measured false-merge rate before auto-merge.

Resolver policy:
- MIN_EVIDENCE_FOR_AUTO_MERGE = 3
- MIN_CUMULATIVE_DURATION = 10s
- MAX_CONFLICT_SEVERITY = "medium"
- MONOTONICITY = true (human-confirmed Person can't be auto-split)
- PROPOSAL_ONLY = true (until false-merge rate < threshold)

Forbidden auto-merge cases:
- Single evidence source only
- Conflicting face and voice matches
- Low-quality embeddings
- First-ever merge for a global template
- Merge would affect >10 existing records
"""
import json
from pathlib import Path
from src.global_index import GlobalIndex, create_global_face_index, create_global_voice_index


# Resolver policy constants
MIN_EVIDENCE_FOR_AUTO_MERGE = 3
MIN_CUMULATIVE_DURATION_MS = 10000  # 10 seconds
MAX_CONFLICT_SEVERITY = "medium"
MONOTONICITY = True
PROPOSAL_ONLY = True  # Set to False only after measured false-merge rate is acceptable


class GlobalIdentityResolver:
    """Resolves person identity from face/voice evidence."""
    
    def __init__(self):
        self.face_index = create_global_face_index()
        self.voice_index = create_global_voice_index()
        self.merge_proposals = []
        self.conflicts = []
        self.persons = {}  # person_id → {"face_template": ..., "voice_template": ...}
    
    def process_video(self, face_clusters: list, voice_clusters: list,
                       binding_evidence: list, video_id: str) -> dict:
        """Process a video's clusters and evidence.
        
        Returns:
            {
                "face_matches": [...],
                "voice_matches": [...],
                "merge_proposals": [...],
                "conflicts": [...],
            }
        """
        face_matches = []
        voice_matches = []
        
        # Match face clusters to global face index
        for fc in face_clusters:
            centroid = fc.get("centroid")
            if centroid:
                import numpy as np
                result = self.face_index.match(
                    np.array(centroid), video_id=video_id,
                    quality_score=fc.get("quality_score", 1.0),
                )
                face_matches.append({
                    "local_cluster_id": fc.get("cluster_id"),
                    **result,
                })
        
        # Match voice clusters to global voice index
        for vc in voice_clusters:
            centroid = vc.get("centroid")
            if centroid:
                import numpy as np
                result = self.voice_index.match(
                    np.array(centroid), video_id=video_id,
                    quality_score=vc.get("quality_score", 1.0),
                )
                voice_matches.append({
                    "local_cluster_id": vc.get("speaker_id", vc.get("cluster_id")),
                    **result,
                })
        
        # Generate merge proposals from evidence
        proposals = self._generate_proposals(
            face_matches, voice_matches, binding_evidence, video_id
        )
        
        return {
            "face_matches": face_matches,
            "voice_matches": voice_matches,
            "merge_proposals": proposals,
            "conflicts": self.conflicts,
        }
    
    def _generate_proposals(self, face_matches, voice_matches, evidence, video_id):
        """Generate merge proposals from matched clusters and evidence."""
        proposals = []
        
        # Find face-voice bindings
        for ev in evidence:
            if ev.get("asserts_identity"):
                continue  # Skip (shouldn't happen, but safety)
            
            face_id = ev.get("face_cluster_id")
            voice_id = ev.get("speaker_cluster_id", ev.get("speaker_id"))
            
            # Find corresponding global templates
            face_template = None
            voice_template = None
            
            for fm in face_matches:
                if str(fm.get("local_cluster_id")) == str(face_id):
                    face_template = fm.get("template_id")
                    break
            
            for vm in voice_matches:
                if str(vm.get("local_cluster_id")) == str(voice_id):
                    voice_template = vm.get("template_id")
                    break
            
            if face_template and voice_template:
                proposal = {
                    "proposal_id": f"prop_{len(proposals)}",
                    "face_template_id": face_template,
                    "voice_template_id": voice_template,
                    "evidence_type": ev.get("evidence_type"),
                    "confidence": ev.get("calibrated_probability"),
                    "video_id": video_id,
                    "status": "pending_review" if PROPOSAL_ONLY else "auto_merge",
                }
                proposals.append(proposal)
        
        self.merge_proposals.extend(proposals)
        return proposals
    
    def summary(self) -> dict:
        """Return resolver summary."""
        return {
            "face_templates": self.face_index.summary(),
            "voice_templates": self.voice_index.summary(),
            "merge_proposals": len(self.merge_proposals),
            "conflicts": len(self.conflicts),
            "proposal_only": PROPOSAL_ONLY,
        }


if __name__ == "__main__":
    import numpy as np
    
    resolver = GlobalIdentityResolver()
    
    # Simulate processing two videos
    np.random.seed(42)
    person_a = np.random.randn(512); person_a /= np.linalg.norm(person_a)
    person_b = np.random.randn(512); person_b /= np.linalg.norm(person_b)
    
    # Video 1
    result1 = resolver.process_video(
        face_clusters=[
            {"cluster_id": 0, "centroid": person_a.tolist(), "quality_score": 0.8},
            {"cluster_id": 1, "centroid": person_b.tolist(), "quality_score": 0.7},
        ],
        voice_clusters=[
            {"speaker_id": "SPEAKER_00", "centroid": np.random.randn(256).tolist()},
        ],
        binding_evidence=[
            {"face_cluster_id": 0, "speaker_id": "SPEAKER_00", "evidence_type": "temporal_overlap"},
        ],
        video_id="video_1",
    )
    
    print("Video 1 results:")
    print(f"  Face matches: {len(result1['face_matches'])}")
    print(f"  Voice matches: {len(result1['voice_matches'])}")
    print(f"  Merge proposals: {len(result1['merge_proposals'])}")
    
    print(f"\nResolver summary: {resolver.summary()}")
