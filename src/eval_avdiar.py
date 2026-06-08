"""Evaluate face-voice binding pipeline on AVDIAR dataset.

AVDIAR structure per sequence:
- Video/Seq*_CAM1.mp4 — 720×450, 25fps
- Audio/Seq*.wav — 6-channel, 48kHz
- GroundTruth/speakers.rttm — speaker diarization (person ID)
- GroundTruth/vad.rttm — voice activity detection
- GroundTruth/face_bb.txt — face bboxes per frame (frame_id, face_id, x, y, w, h)
"""
import json
import os
import sys
import subprocess
from pathlib import Path


def parse_rttm(rttm_path: str) -> list:
    """Parse RTTM file into segments.
    
    Format: SPEAKER file_id channel start_time duration <NA> <NA> speaker_id <NA>
    """
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                segments.append({
                    "speaker_id": parts[7],
                    "start": float(parts[3]),
                    "duration": float(parts[4]),
                    "end": float(parts[3]) + float(parts[4]),
                })
    return segments


def parse_face_bb(bb_path: str) -> dict:
    """Parse face bounding box file.
    
    Format: frame_id, face_id, x, y, width, height, ...
    Returns: {frame_id: [{face_id, x, y, w, h}, ...]}
    """
    frames = {}
    with open(bb_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                frame_id = int(parts[0])
                face_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                frames.setdefault(frame_id, []).append({
                    "face_id": face_id,
                    "x": x, "y": y, "w": w, "h": h,
                })
    return frames


def gt_to_binding_evidence(speakers_rttm: str, face_bb_path: str, fps: float = 25.0) -> list:
    """Convert ground truth to binding evidence for evaluation.
    
    For each speaker segment, find which face_ids are visible during that time.
    This creates the ground truth face-voice binding.
    """
    speakers = parse_rttm(speakers_rttm)
    faces = parse_face_bb(face_bb_path)
    
    evidence = []
    for seg in speakers:
        start_frame = int(seg["start"] * fps)
        end_frame = int(seg["end"] * fps)
        
        # Find face_ids visible during this segment
        visible_faces = {}
        for frame_id in range(start_frame, min(end_frame + 1, max(faces.keys()) + 1)):
            if frame_id in faces:
                for face in faces[frame_id]:
                    fid = face["face_id"]
                    visible_faces.setdefault(fid, 0)
                    visible_faces[fid] += 1
        
        # Each visible face is a binding candidate
        for face_id, count in visible_faces.items():
            evidence.append({
                "speaker_id": seg["speaker_id"],
                "face_id": face_id,
                "segment_start": seg["start"],
                "segment_end": seg["end"],
                "frames_visible": count,
                "is_binding": True,  # In GT, all speaker-face pairs are true bindings
            })
    
    return evidence


def run_pipeline_on_avdiar(seq_dir: str, output_dir: str) -> dict:
    """Run our pipeline on an AVDIAR sequence."""
    seq_name = os.path.basename(seq_dir)
    inner_dir = os.path.join(seq_dir, seq_name)
    
    video_path = os.path.join(inner_dir, "Video", f"{seq_name}_CAM1.mp4")
    audio_path = os.path.join(inner_dir, "Audio", f"{seq_name}.wav")
    
    if not os.path.exists(video_path):
        return {"error": f"Video not found: {video_path}"}
    
    # Extract mono audio from 6-channel
    mono_audio = os.path.join(output_dir, f"{seq_name}_mono.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", mono_audio
    ], capture_output=True)
    
    # Run our pipeline
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    results = {"sequence": seq_name}
    
    # Step 1: Shot detection
    try:
        from src.step1_shot_detection import detect_shots
        shots = detect_shots(video_path, os.path.join(output_dir, "shots.json"))
        results["shots"] = len(shots)
    except Exception as e:
        results["shots_error"] = str(e)
    
    # Step 2-3: Face detection + tracking + embedding
    try:
        from src.step23_detect_track_embed import detect_and_track
        tracks = detect_and_track(video_path, os.path.join(output_dir, "face_tracks.json"))
        results["face_tracks"] = len(tracks)
    except Exception as e:
        results["face_tracks_error"] = str(e)
    
    # Step 4: Face clustering
    try:
        from src.step4_face_cluster import cluster_faces
        clusters = cluster_faces(
            os.path.join(output_dir, "face_tracks.json"),
            os.path.join(output_dir, "face_clusters.json"),
        )
        results["face_clusters"] = len(clusters)
    except Exception as e:
        results["face_clusters_error"] = str(e)
    
    return results


def evaluate_diarization(pred_rttm: str, gt_rttm: str) -> dict:
    """Compare predicted speaker diarization with ground truth."""
    gt_segs = parse_rttm(gt_rttm)
    pred_segs = parse_rttm(pred_rttm) if os.path.exists(pred_rttm) else []
    
    gt_speakers = set(s["speaker_id"] for s in gt_segs)
    pred_speakers = set(s["speaker_id"] for s in pred_segs)
    
    return {
        "gt_speakers": sorted(gt_speakers),
        "pred_speakers": sorted(pred_speakers),
        "gt_segments": len(gt_segs),
        "pred_segments": len(pred_segs),
    }


if __name__ == "__main__":
    avdiar_dir = "datasets/avdiar"
    
    # Find all sequences
    seqs = sorted([d for d in os.listdir(avdiar_dir) if d.startswith("Seq") and os.path.isdir(os.path.join(avdiar_dir, d))])
    
    print(f"Found {len(seqs)} AVDIAR sequences")
    print("=" * 60)
    
    for seq_name in seqs:
        seq_dir = os.path.join(avdiar_dir, seq_name)
        inner_dir = os.path.join(seq_dir, seq_name)
        
        # Parse ground truth
        speakers_rttm = os.path.join(inner_dir, "GroundTruth", "speakers.rttm")
        face_bb = os.path.join(inner_dir, "GroundTruth", "face_bb.txt")
        
        if os.path.exists(speakers_rttm) and os.path.exists(face_bb):
            speakers = parse_rttm(speakers_rttm)
            faces = parse_face_bb(face_bb)
            
            speaker_ids = set(s["speaker_id"] for s in speakers)
            face_ids = set()
            for frame_faces in faces.values():
                for f in frame_faces:
                    face_ids.add(f["face_id"])
            
            total_speech = sum(s["duration"] for s in speakers)
            
            print(f"\n{seq_name}:")
            print(f"  Speakers: {sorted(speaker_ids)} ({len(speaker_ids)} total)")
            print(f"  Face IDs: {sorted(face_ids)} ({len(face_ids)} total)")
            print(f"  Speech duration: {total_speech:.1f}s")
            print(f"  Frames: {len(faces)}")
            
            # Create GT binding evidence
            gt_binding = gt_to_binding_evidence(speakers_rttm, face_bb)
            print(f"  GT bindings: {len(gt_binding)}")
            for b in gt_binding[:3]:
                print(f"    {b['speaker_id']} ↔ face_{b['face_id']} ({b['frames_visible']} frames)")
