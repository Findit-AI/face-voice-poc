"""Step 2-3: Face detection (Apple Vision) + tracking + embedding (InsightFace)."""
import cv2
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, List

import Vision
import Quartz
from Foundation import NSData


def detect_faces_apple_vision(frame_bgr: np.ndarray, confidence_threshold: float = 0.5) -> list[dict]:
    """Detect faces in a frame using Apple Vision framework.
    
    Returns list of: [{bbox: [x,y,w,h], confidence, landmarks: {...}}]
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    
    # Encode to JPEG for Vision framework
    _, jpeg_data = cv2.imencode('.jpg', frame_rgb)
    ns_data = NSData.dataWithBytes_length_(jpeg_data.tobytes(), len(jpeg_data.tobytes()))
    
    ci_image = Quartz.CIImage.imageWithData_(ns_data)
    
    # Create face detection request
    request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
    
    # Create handler and perform
    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
        ci_image, None
    )
    
    success = handler.performRequests_error_([request], None)
    if not success[0]:
        return []
    
    results = []
    for observation in request.results():
        conf = float(observation.confidence())
        if conf < confidence_threshold:
            continue
        
        bbox = observation.boundingBox()
        # Vision returns normalized coordinates (0-1), origin at bottom-left
        x = int(bbox.origin.x * w)
        y = int((1 - bbox.origin.y - bbox.size.height) * h)
        bw = int(bbox.size.width * w)
        bh = int(bbox.size.height * h)
        
        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        
        results.append({
            "bbox": [x, y, bw, bh],
            "confidence": round(conf, 4),
        })
    
    return results


def iou(b1, b2):
    """IoU between two bboxes [x, y, w, h]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    return inter / (area1 + area2 - inter + 1e-6)


def track_and_embed(video_path: str, shots_path: str, output_path: str,
                    detect_fps: float = 2.0, embed_per_track: int = 5):
    """Detect faces, track across frames, embed best crops.
    
    Pipeline:
    1. Apple Vision detection at detect_fps
    2. IoU tracking between detections
    3. InsightFace embedding on best crops per track
    """
    # Load shots
    with open(shots_path) as f:
        shots = json.load(f)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detect_interval = int(fps / detect_fps)
    
    print(f"Video: {fps:.1f}fps, {total_frames} frames")
    print(f"Detection interval: every {detect_interval} frames ({detect_fps}fps)")
    
    # Phase 1: Apple Vision detection + IoU tracking
    print("\n--- Phase 1: Face Detection + Tracking ---")
    tracks = []  # list of track dicts
    track_next_id = 0
    
    for shot in shots:
        start_f = shot["start_frame"]
        end_f = shot["end_frame"]
        shot_id = shot["shot_id"]
        
        # Reset tracks at shot boundary
        active_tracks = []
        
        for frame_idx in range(start_f, end_f, detect_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = detect_faces_apple_vision(frame)
            ts = frame_idx / fps
            
            # Match detections to active tracks via IoU
            matched_det = set()
            matched_track = set()
            for ti, (last_frame, last_bbox, tid) in enumerate(active_tracks):
                if frame_idx - last_frame > detect_interval * 3:
                    continue  # track too old
                best_iou, best_di = 0, -1
                for di, det in enumerate(detections):
                    if di in matched_det:
                        continue
                    i = iou(last_bbox, det["bbox"])
                    if i > best_iou:
                        best_iou, best_di = i, di
                if best_iou > 0.3 and best_di >= 0:
                    det = detections[best_di]
                    # Update track
                    tracks[tid]["frames"].append({
                        "frame": frame_idx, "time": round(ts, 3),
                        "bbox": det["bbox"], "confidence": det["confidence"],
                    })
                    active_tracks[ti] = (frame_idx, det["bbox"], tid)
                    matched_det.add(best_di)
                    matched_track.add(ti)
            
            # Create new tracks for unmatched detections
            for di, det in enumerate(detections):
                if di in matched_det:
                    continue
                tid = track_next_id
                track_next_id += 1
                tracks.append({
                    "track_id": tid, "shot_id": shot_id,
                    "frames": [{
                        "frame": frame_idx, "time": round(ts, 3),
                        "bbox": det["bbox"], "confidence": det["confidence"],
                    }],
                })
                active_tracks.append((frame_idx, det["bbox"], tid))
            
            # Remove stale tracks
            active_tracks = [(f, b, t) for f, b, t in active_tracks
                           if frame_idx - f <= detect_interval * 3]
        
        if (shot_id + 1) % 5 == 0:
            print(f"  Shot {shot_id}: {len(tracks)} total tracks")
    
    # Filter short tracks
    tracks = [t for t in tracks if len(t["frames"]) >= 3]
    print(f"\nTracks after filtering (≥3 detections): {len(tracks)}")
    
    # Phase 2: InsightFace embedding on best crops
    print("\n--- Phase 2: Face Embedding (InsightFace) ---")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # Initialize LanceDB for per-frame embedding storage
    from embedding_store import get_db, create_face_embeddings_table, write_face_embeddings
    db = get_db()
    create_face_embeddings_table(db)
    video_id = hashlib.md5(video_path.encode()).digest()  # 16 bytes
    
    total_stored = 0
    
    for track in tracks:
        # Select best frames by confidence
        sorted_frames = sorted(track["frames"], key=lambda f: f["confidence"], reverse=True)
        best_frames = sorted_frames[:embed_per_track]
        
        embeddings = []
        frame_embedding_records = []  # for LanceDB
        for bf in best_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, bf["frame"])
            ret, frame = cap.read()
            if not ret:
                continue
            x, y, w, h = bf["bbox"]
            fh, fw = frame.shape[:2]
            
            # Pad crop generously for InsightFace detector
            pad = int(max(w, h) * 0.5)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(fw, x + w + pad)
            y2 = min(fh, y + h + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or min(crop.shape[:2]) < 20:
                continue
            
            # Get embedding from InsightFace
            faces = app.get(crop)
            if faces and faces[0].normed_embedding is not None:
                emb_list = faces[0].normed_embedding.tolist()
                embeddings.append(emb_list)
                # Collect record for LanceDB
                track_id_bytes = track["track_id"].to_bytes(16, "big", signed=False)
                frame_embedding_records.append({
                    "track_id": track_id_bytes,
                    "frame_number": bf["frame"],
                    "vector": emb_list,
                    "bbox_size_px": w * h,
                    "vision_confidence": bf["confidence"],
                    "quality_score": bf["confidence"],  # use detection confidence as quality
                })
        
        # Store per-frame embeddings in LanceDB
        vector_ids = []
        if frame_embedding_records:
            vector_ids = write_face_embeddings(db, video_id, frame_embedding_records)
            total_stored += len(vector_ids)
        
        if embeddings:
            # Centroid of selected embeddings (backward compatible)
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            track["embedding"] = centroid.tolist()
            track["embedding_quality"] = len(embeddings) / embed_per_track
        else:
            track["embedding"] = None
            track["embedding_quality"] = 0.0
        
        # Store vector_ids for LanceDB linkage
        track["vector_ids"] = [v.hex() for v in vector_ids] if vector_ids else []
        
        # Summary stats
        bboxes = [f["bbox"] for f in track["frames"]]
        widths = [b[2] for b in bboxes]
        track["face_count"] = len(track["frames"])
        track["avg_face_size"] = round(np.mean(widths), 1)
        track["time_range"] = [
            round(track["frames"][0]["time"], 3),
            round(track["frames"][-1]["time"], 3),
        ]
    
    print(f"\nStored {total_stored} face embeddings in LanceDB (video_id={video_id.hex()[:8]}...)")
    
    cap.release()
    
    # Filter tracks without embeddings
    tracks_with_emb = [t for t in tracks if t.get("embedding") is not None]
    print(f"Tracks with embeddings: {len(tracks_with_emb)}")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Remove raw frames for compact output
    compact = []
    for t in tracks_with_emb:
        compact.append({
            "track_id": t["track_id"],
            "shot_id": t["shot_id"],
            "face_count": t["face_count"],
            "avg_face_size": t["avg_face_size"],
            "time_range": t["time_range"],
            "embedding": t["embedding"],
            "embedding_quality": t["embedding_quality"],
            "vector_ids": t.get("vector_ids", []),
            "frames": t["frames"],  # keep for ASD later
        })
    
    with open(output_path, "w") as f:
        json.dump(compact, f)
    
    print(f"Saved {len(compact)} tracks to {output_path}")
    return compact


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "Desktop/Kino Demo Render.mp4")
    shots = sys.argv[2] if len(sys.argv) > 2 else "output/shots.json"
    output = sys.argv[3] if len(sys.argv) > 3 else "output/face_tracks.json"
    track_and_embed(video, shots, output)
