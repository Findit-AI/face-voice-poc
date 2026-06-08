"""Global identity index for cross-video face and voice matching.

Implements DIA-style online clustering:
- Maintain centroids in memory
- For each new embedding, find best match
- If similarity >= threshold → assign to existing template
- Else → create new template

Three-zone matching: candidate_existing / candidate_new / ambiguous
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances


class GlobalIndex:
    """Online clustering index for cross-video identity matching."""
    
    def __init__(self, modality: str, dimension: int,
                 accept_threshold: float = 0.3,
                 reject_threshold: float = 0.6,
                 margin_threshold: float = 0.1):
        """
        Args:
            modality: "face" or "voice"
            dimension: embedding dimension (512 for face, 256 for voice)
            accept_threshold: max distance to accept as existing match
            reject_threshold: min distance to create new template
            margin_threshold: min gap between top1 and top2 for confident match
        """
        self.modality = modality
        self.dimension = dimension
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.margin_threshold = margin_threshold
        
        # Template storage
        self.templates = []  # list of {"id": str, "centroid": np.array, "count": int, "video_ids": set}
    
    def match(self, embedding: np.ndarray, video_id: str = None,
              quality_score: float = 1.0) -> dict:
        """Match an embedding against existing templates.
        
        Returns:
            {
                "decision": "candidate_existing" | "candidate_new" | "ambiguous",
                "template_id": str | None,
                "distance": float,
                "top2_distance": float | None,
                "margin": float | None,
            }
        """
        if not self.templates:
            # No templates yet — create new
            tid = f"{self.modality}_0"
            self.templates.append({
                "id": tid,
                "centroid": embedding.copy(),
                "count": 1,
                "video_ids": {video_id} if video_id else set(),
            })
            return {
                "decision": "candidate_new",
                "template_id": tid,
                "distance": 0.0,
                "top2_distance": None,
                "margin": None,
            }
        
        # Compute distances to all templates
        centroids = np.array([t["centroid"] for t in self.templates])
        distances = cosine_distances(embedding.reshape(1, -1), centroids)[0]
        
        # Find top-2
        sorted_idx = np.argsort(distances)
        top1_idx = sorted_idx[0]
        top1_dist = distances[top1_idx]
        top2_dist = distances[sorted_idx[1]] if len(sorted_idx) > 1 else None
        margin = (top2_dist - top1_dist) if top2_dist is not None else None
        
        # Three-zone decision
        if top1_dist <= self.accept_threshold:
            # Check margin
            if margin is None or margin >= self.margin_threshold:
                # Accept: update centroid
                t = self.templates[top1_idx]
                alpha = 0.1  # EMA weight
                t["centroid"] = (1 - alpha) * t["centroid"] + alpha * embedding
                t["centroid"] = t["centroid"] / np.linalg.norm(t["centroid"])
                t["count"] += 1
                if video_id:
                    t["video_ids"].add(video_id)
                return {
                    "decision": "candidate_existing",
                    "template_id": t["id"],
                    "distance": float(top1_dist),
                    "top2_distance": float(top2_dist) if top2_dist is not None else None,
                    "margin": float(margin) if margin is not None else None,
                }
            else:
                # Ambiguous: margin too small
                return {
                    "decision": "ambiguous",
                    "template_id": self.templates[top1_idx]["id"],
                    "distance": float(top1_dist),
                    "top2_distance": float(top2_dist),
                    "margin": float(margin),
                }
        elif top1_dist >= self.reject_threshold:
            # Create new template
            tid = f"{self.modality}_{len(self.templates)}"
            self.templates.append({
                "id": tid,
                "centroid": embedding.copy(),
                "count": 1,
                "video_ids": {video_id} if video_id else set(),
            })
            return {
                "decision": "candidate_new",
                "template_id": tid,
                "distance": float(top1_dist),
                "top2_distance": float(top2_dist) if top2_dist is not None else None,
                "margin": float(margin) if margin is not None else None,
            }
        else:
            # Ambiguous zone
            return {
                "decision": "ambiguous",
                "template_id": self.templates[top1_idx]["id"],
                "distance": float(top1_dist),
                "top2_distance": float(top2_dist) if top2_dist is not None else None,
                "margin": float(margin) if margin is not None else None,
            }
    
    def summary(self) -> dict:
        """Return summary of all templates."""
        return {
            "modality": self.modality,
            "num_templates": len(self.templates),
            "templates": [{
                "id": t["id"],
                "count": t["count"],
                "video_ids": list(t["video_ids"]),
            } for t in self.templates],
        }


def create_global_face_index(accept_threshold: float = 0.3,
                              reject_threshold: float = 0.6) -> GlobalIndex:
    """Create a GlobalIndex for face embeddings (512D)."""
    return GlobalIndex("face", 512, accept_threshold, reject_threshold)


def create_global_voice_index(accept_threshold: float = 0.4,
                               reject_threshold: float = 0.7) -> GlobalIndex:
    """Create a GlobalIndex for voice embeddings (256D)."""
    return GlobalIndex("voice", 256, accept_threshold, reject_threshold)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    index = create_global_face_index()
    
    # Simulate 3 people across 2 videos
    person_a_v1 = np.random.randn(512); person_a_v1 /= np.linalg.norm(person_a_v1)
    person_a_v2 = person_a_v1 + np.random.randn(512) * 0.1; person_a_v2 /= np.linalg.norm(person_a_v2)
    person_b_v1 = np.random.randn(512); person_b_v1 /= np.linalg.norm(person_b_v1)
    
    # Video 1
    r1 = index.match(person_a_v1, "video_1")
    r2 = index.match(person_b_v1, "video_1")
    print(f"V1 Person A: {r1['decision']} ({r1['template_id']})")
    print(f"V1 Person B: {r2['decision']} ({r2['template_id']})")
    
    # Video 2
    r3 = index.match(person_a_v2, "video_2")
    print(f"V2 Person A: {r3['decision']} ({r3['template_id']}, dist={r3['distance']:.3f})")
    
    print(f"\nSummary: {index.summary()}")
