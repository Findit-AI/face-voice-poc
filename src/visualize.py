"""Generate visual report with annotated frames for each binding candidate."""
import cv2
import json
import numpy as np
from pathlib import Path


def visualize(video_path: str, candidates_path: str, face_tracks_path: str,
              face_clusters_path: str, output_dir: str):
    """Generate annotated images for each binding candidate."""
    with open(candidates_path) as f:
        candidates = json.load(f)
    with open(face_tracks_path) as f:
        tracks = json.load(f)
    with open(face_clusters_path) as f:
        clusters = json.load(f)
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Color palette for clusters
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
    
    # Build track_id → cluster mapping
    track_to_cluster = {}
    for c in clusters:
        for tid in c["track_ids"]:
            track_to_cluster[tid] = c["cluster_id"]
    
    # Generate per-candidate annotated frames
    for cand in candidates:
        cid = cand["face_cluster_id"]
        speaker = cand["speaker_id"]
        seg_start = cand["segment_start"]
        seg_end = cand["segment_end"]
        mid_time = (seg_start + seg_end) / 2
        
        # Get frame at middle of segment
        frame_idx = int(mid_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Draw all face tracks active at this time
        for track in tracks:
            t_start, t_end = track["time_range"]
            if t_start <= mid_time <= t_end:
                tcid = track_to_cluster.get(track["track_id"], -1)
                # Find the closest frame in this track
                closest = min(track["frames"], key=lambda f: abs(f["time"] - mid_time))
                x, y, w, h = closest["bbox"]
                color = colors[tcid % len(colors)] if tcid >= 0 else (128,128,128)
                
                is_binding = (tcid == cid)
                thickness = 3 if is_binding else 1
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                label = f"F{tcid}" + (" BOUND" if is_binding else "")
                cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add text overlay
        cv2.putText(frame, f"{speaker} @ {seg_start:.1f}s-{seg_end:.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"F{cid} bound to {speaker} (overlap={cand['overlap_duration_ms']}ms)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        
        fname = f"candidate_{cand['candidate_id']}_F{cid}_{speaker}.jpg"
        cv2.imwrite(str(out / fname), frame)
        print(f"  {fname}")
    
    # Generate timeline visualization
    timeline = np.zeros((200, 1200, 3), dtype=np.uint8)
    
    # Draw speaker segments
    total_dur = 105.0
    for seg_group in _group_speaker_segments(candidates):
        pass  # simplified
    
    # Draw face tracks
    for track in tracks:
        tcid = track_to_cluster.get(track["track_id"], -1)
        color = colors[tcid % len(colors)] if tcid >= 0 else (128,128,128)
        t_start, t_end = track["time_range"]
        x1 = int(t_start / total_dur * 1200)
        x2 = int(t_end / total_dur * 1200)
        y = 50 + tcid * 30
        cv2.rectangle(timeline, (x1, y), (x2, y+20), color, -1)
        cv2.putText(timeline, f"F{tcid}", (x1, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    cv2.imwrite(str(out / "timeline.jpg"), timeline)
    print(f"  timeline.jpg")
    
    cap.release()
    print(f"\nSaved to {out}")


def _group_speaker_segments(candidates):
    """Group candidates by speaker."""
    groups = {}
    for c in candidates:
        sid = c["speaker_id"]
        if sid not in groups:
            groups[sid] = []
        groups[sid].append(c)
    return groups


if __name__ == "__main__":
    visualize(
        str(Path.home() / "Desktop/Kino Demo Render.mp4"),
        "output/candidates.json",
        "output/face_tracks.json",
        "output/face_clusters.json",
        "output/visualizations",
    )
