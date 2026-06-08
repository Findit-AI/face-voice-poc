"""Step 4: Face clustering using DBSCAN with optional LanceDB centroid computation."""
import json
import numpy as np
from pathlib import Path
from typing import Optional, List
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

try:
    from quality_filter import filter_face_embeddings
    HAS_QUALITY_FILTER = True
except ImportError:
    HAS_QUALITY_FILTER = False

try:
    from src.embedding_store import get_db, compute_centroid
    HAS_LANCEDB = True
except ImportError:
    HAS_LANCEDB = False


def cluster_faces(tracks_path: str, output_path: str, eps: float = 0.8, min_samples: int = 1,
                  use_lancedb: bool = True):
    """Cluster face tracks into LocalFaceClusters using DBSCAN.
    
    Args:
        tracks_path: path to face_tracks.json
        output_path: path to save clusters
        eps: DBSCAN epsilon (cosine distance threshold)
        min_samples: DBSCAN min_samples
        use_lancedb: if True, compute centroids from LanceDB embeddings
    """
    with open(tracks_path) as f:
        tracks = json.load(f)
    
    # Extract embeddings
    embeddings = []
    valid_tracks = []
    for t in tracks:
        if t.get("embedding") is not None:
            embeddings.append(t["embedding"])
            valid_tracks.append(t)
    
    # Apply quality filter before clustering
    if HAS_QUALITY_FILTER:
        pre_filter_count = len(valid_tracks)
        valid_tracks = filter_face_embeddings(valid_tracks)
        embeddings = [t["embedding"] for t in valid_tracks]
        print(f"Quality filter: {pre_filter_count} → {len(valid_tracks)} tracks")
    
    if not embeddings:
        print("No embeddings found!")
        return []
    
    embs = np.array(embeddings)
    dist = cosine_distances(embs)
    
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(dist)
    
    # Build clusters
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(valid_tracks[i])
    
    # Optionally load LanceDB for centroid computation
    db = None
    if use_lancedb and HAS_LANCEDB:
        try:
            db = get_db()  # type: ignore[possibly-unbound]
            print("Using LanceDB for centroid computation")
        except Exception as e:
            print(f"LanceDB not available ({e}), falling back to JSON centroids")
            db = None
    elif use_lancedb:
        print("LanceDB module not available, falling back to JSON centroids")
    
    # Build output
    result = []
    for cid, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
        # Collect all vector_ids from member tracks
        all_vector_ids = []
        for m in members:
            vids = m.get("vector_ids", [])
            all_vector_ids.extend(vids)
        
        # Compute centroid — prefer LanceDB if available
        if db and all_vector_ids:
            try:
                # Convert hex strings back to bytes
                vid_bytes = [bytes.fromhex(v) for v in all_vector_ids]
                centroid = compute_centroid(db, "face_embeddings", vid_bytes)
            except Exception as e:
                print(f"  LanceDB centroid failed for cluster {cid}: {e}, using JSON fallback")
                # Fallback: compute from track centroids
                member_embs = np.array([m["embedding"] for m in members])
                centroid = np.mean(member_embs, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
        else:
            # Fallback: compute from track centroids in JSON
            member_embs = np.array([m["embedding"] for m in members])
            centroid = np.mean(member_embs, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
        
        # Time range
        all_times = []
        for m in members:
            all_times.extend(m["time_range"])
        
        result.append({
            "cluster_id": int(cid),
            "track_ids": [m["track_id"] for m in members],
            "track_count": len(members),
            "total_faces": sum(m["face_count"] for m in members),
            "avg_face_size": round(np.mean([m["avg_face_size"] for m in members]), 1),
            "time_range": [round(min(all_times), 3), round(max(all_times), 3)],
            "centroid": centroid.tolist(),
            "vector_ids": all_vector_ids,
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Face clustering: {len(valid_tracks)} tracks → {len(result)} clusters (eps={eps})")
    for c in result:
        vid_count = len(c.get("vector_ids", []))
        print(f"  F{c['cluster_id']}: {c['track_count']} tracks, "
              f"{c['total_faces']} faces, "
              f"avg_size={c['avg_face_size']:.0f}px, "
              f"{c['time_range'][0]:.1f}s-{c['time_range'][1]:.1f}s, "
              f"{vid_count} LanceDB vectors")
    
    return result


if __name__ == "__main__":
    import sys
    tracks = sys.argv[1] if len(sys.argv) > 1 else "output/face_tracks.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "output/face_clusters.json"
    cluster_faces(tracks, output)
