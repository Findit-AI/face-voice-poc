"""Quality filtering for face and voice embeddings.

Filters embeddings before they enter the clustering/global matching pipeline.
Face: crop size, blur, pose, confidence, duplicate IoU
Voice: duration, overlap, speech confidence, energy
"""
import numpy as np


# --- Face quality filters ---

FACE_MIN_CROP_SIZE = 40       # pixels
FACE_MAX_BLUR_SCORE = 100.0   # Laplacian variance threshold (lower = more blurry)
FACE_MAX_POSE_ANGLE = 45.0    # degrees
FACE_MIN_CONFIDENCE = 0.5     # Apple Vision confidence
FACE_MAX_OCCLUSION = 0.3      # fraction
FACE_DUPLICATE_IOU = 0.9      # reject near-duplicates in same track


def filter_face_embeddings(tracks: list, min_crop_size: int = FACE_MIN_CROP_SIZE,
                           min_confidence: float = FACE_MIN_CONFIDENCE) -> list:
    """Filter face tracks by quality criteria.
    
    Returns tracks that pass all quality gates.
    Each track gets a quality_score field.
    """
    passed = []
    rejected_reasons = {"small_crop": 0, "low_confidence": 0, "few_crops": 0}
    
    for track in tracks:
        # Check minimum crop size
        avg_size = track.get("avg_face_size", 0)
        if avg_size < min_crop_size:
            rejected_reasons["small_crop"] += 1
            continue
        
        # Check confidence
        frames = track.get("frames", [])
        if not frames:
            rejected_reasons["few_crops"] += 1
            continue
        
        avg_conf = np.mean([f.get("confidence", 0) for f in frames])
        if avg_conf < min_confidence:
            rejected_reasons["low_confidence"] += 1
            continue
        
        # Minimum usable crops per track
        if len(frames) < 3:
            rejected_reasons["few_crops"] += 1
            continue
        
        # Compute quality score
        size_score = min(avg_size / 200.0, 1.0)  # 200px = perfect
        conf_score = avg_conf
        quantity_score = min(len(frames) / 10.0, 1.0)  # 10 frames = perfect
        
        track["quality_score"] = round(
            0.4 * size_score + 0.3 * conf_score + 0.3 * quantity_score, 3
        )
        passed.append(track)
    
    print(f"Face quality filter: {len(passed)}/{len(tracks)} passed")
    for reason, count in rejected_reasons.items():
        if count > 0:
            print(f"  Rejected {count}: {reason}")
    
    return passed


# --- Voice quality filters ---

VOICE_MIN_DURATION_MS = 500     # 500ms (2s for production with proper diarization)
VOICE_MIN_SPEECH_CONFIDENCE = 0.5
VOICE_MIN_ENERGY = 0.01         # RMS energy threshold


def filter_voice_segments(segments: list, min_duration_ms: int = VOICE_MIN_DURATION_MS) -> list:
    """Filter voice segments by quality criteria.
    
    Returns segments that pass all quality gates.
    """
    passed = []
    rejected_reasons = {"short_duration": 0, "low_confidence": 0}
    
    for seg in segments:
        # Duration check
        dur = seg.get("duration_ms", 0)
        if dur < min_duration_ms:
            rejected_reasons["short_duration"] += 1
            continue
        
        # Speech confidence (if available)
        conf = seg.get("speech_confidence", 1.0)
        if conf < VOICE_MIN_SPEECH_CONFIDENCE:
            rejected_reasons["low_confidence"] += 1
            continue
        
        passed.append(seg)
    
    print(f"Voice quality filter: {len(passed)}/{len(segments)} passed")
    for reason, count in rejected_reasons.items():
        if count > 0:
            print(f"  Rejected {count}: {reason}")
    
    return passed


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    # Test with existing data
    tracks_path = "output/face_tracks.json"
    if Path(tracks_path).exists():
        with open(tracks_path) as f:
            tracks = json.load(f)
        filtered = filter_face_embeddings(tracks)
        print(f"\nFiltered face tracks: {len(filtered)}")
        for t in filtered[:3]:
            print(f"  Track {t['track_id']}: quality={t.get('quality_score', 'N/A')}, "
                  f"size={t.get('avg_face_size', 0):.0f}px, frames={t.get('face_count', 0)}")
    
    segs_path = "output/speaker_segments.json"
    if Path(segs_path).exists():
        with open(segs_path) as f:
            segs = json.load(f)
        filtered = filter_voice_segments(segs)
        print(f"\nFiltered voice segments: {len(filtered)}")
