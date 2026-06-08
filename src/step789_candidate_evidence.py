"""Step 7-9: Candidate windows, binding evidence assembly.

Step 7: Find overlapping face-track / speaker-segment windows
Step 8: (TalkNet ASD — placeholder, run separately)
Step 9: Assemble BindingEvidence
"""
import json
import numpy as np
from pathlib import Path
from src.binding_evidence import refactor_candidates_to_evidence


def find_candidates(face_clusters_path: str, speaker_segments_path: str,
                    face_tracks_path: str, output_path: str,
                    min_duration_ms: int = 300) -> list[dict]:
    """Find candidate (face_cluster, speaker_segment) pairs where they overlap.
    
    A candidate exists when a speaker segment overlaps in time with a face cluster.
    """
    with open(face_clusters_path) as f:
        clusters = json.load(f)
    with open(speaker_segments_path) as f:
        segments = json.load(f)
    with open(face_tracks_path) as f:
        tracks = json.load(f)
    
    # Build track_id → cluster mapping
    track_to_cluster = {}
    for c in clusters:
        for tid in c["track_ids"]:
            track_to_cluster[tid] = c["cluster_id"]
    
    candidates = []
    candidate_id = 0
    
    for seg in segments:
        if seg["duration_ms"] < min_duration_ms:
            continue  # skip short segments
        
        seg_start = seg["start_time"]
        seg_end = seg["end_time"]
        
        # Find face clusters with tracks overlapping this segment
        overlapping_clusters = {}
        for track in tracks:
            if track["track_id"] not in track_to_cluster:
                continue
            cid = track_to_cluster[track["track_id"]]
            t_start, t_end = track["time_range"]
            
            # Check temporal overlap
            overlap_start = max(seg_start, t_start)
            overlap_end = min(seg_end, t_end)
            overlap_dur = overlap_end - overlap_start
            
            if overlap_dur > 0:
                if cid not in overlapping_clusters:
                    overlapping_clusters[cid] = {
                        "overlap_duration": 0,
                        "track_ids": [],
                        "avg_confidence": [],
                        "avg_face_size": [],
                    }
                overlapping_clusters[cid]["overlap_duration"] += overlap_dur
                overlapping_clusters[cid]["track_ids"].append(track["track_id"])
                # Average confidence from frames overlapping this segment
                for frame in track["frames"]:
                    if seg_start <= frame["time"] <= seg_end:
                        overlapping_clusters[cid]["avg_confidence"].append(frame["confidence"])
                overlapping_clusters[cid]["avg_face_size"].append(track["avg_face_size"])
        
        for cid, info in overlapping_clusters.items():
            avg_conf = np.mean(info["avg_confidence"]) if info["avg_confidence"] else 0
            avg_size = np.mean(info["avg_face_size"]) if info["avg_face_size"] else 0
            
            # Determine initial status
            if len(overlapping_clusters) == 1:
                heuristic = "single_face"
            else:
                heuristic = "multi_face_needs_asd"
            
            candidates.append({
                "candidate_id": candidate_id,
                "face_cluster_id": cid,
                "speaker_id": seg["speaker_id"],
                "segment_start": seg["start_time"],
                "segment_end": seg["end_time"],
                "segment_duration_ms": seg["duration_ms"],
                "overlap_duration_ms": round(info["overlap_duration"] * 1000),
                "visible_face_fraction": round(info["overlap_duration"] / (seg["duration_ms"]/1000), 3),
                "face_track_ids": info["track_ids"],
                "avg_face_confidence": round(float(avg_conf), 4),
                "avg_face_size": round(float(avg_size), 1),
                "method": heuristic,
                "status": "candidate",
                # These will be filled by TalkNet ASD
                "asd_score": None,
                "asd_margin": None,
                "asd_frames": None,
            })
            candidate_id += 1
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2)
    
    print(f"Binding candidates: {len(candidates)}")
    print(f"  Single-face: {sum(1 for c in candidates if c['method']=='single_face')}")
    print(f"  Multi-face (needs ASD): {sum(1 for c in candidates if c['method']=='multi_face_needs_asd')}")
    
    return candidates


def assemble_evidence(candidates_path: str, output_path: str,
                      video_id: str = "unknown"):
    """Assemble BindingEvidence from candidates.
    
    For now: convert candidates to evidence format.
    TalkNet scores will be merged in later.
    """
    evidence = refactor_candidates_to_evidence(candidates_path, output_path,
                                                video_id=video_id)
    return evidence


if __name__ == "__main__":
    import sys
    clusters = sys.argv[1] if len(sys.argv) > 1 else "output/face_clusters.json"
    segments = sys.argv[2] if len(sys.argv) > 2 else "output/speaker_segments.json"
    tracks = sys.argv[3] if len(sys.argv) > 3 else "output/face_tracks.json"
    candidates_out = sys.argv[4] if len(sys.argv) > 4 else "output/candidates.json"
    evidence_out = sys.argv[5] if len(sys.argv) > 5 else "output/binding_evidence.json"
    
    find_candidates(clusters, segments, tracks, candidates_out)
    assemble_evidence(candidates_out, evidence_out)
