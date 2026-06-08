"""Active Speaker Detection (ASD) integration using TalkNet.

TalkNet uses cross-attention between face crops and audio MFCC features
to determine if a visible face is speaking. Score > 0 = speaking.

This is the CORRECT way to do face-voice binding:
- Temporal overlap only tells you "face visible while voice present"
- ASD tells you "this specific face is producing this specific voice"

Pipeline:
1. For each temporal overlap candidate (face track + speaker segment)
2. Extract face crops from the candidate window (224×224 grayscale)
3. Extract MFCC features from the audio of the same window
4. Run TalkNet to get per-frame ASD score
5. Average score → binding confidence
"""
import os
import sys
import math
import numpy as np
import cv2
from pathlib import Path

# Add TalkNet to path
TALKNET_DIR = str(Path.home() / "dev" / "TalkNet-ASD")
sys.path.insert(0, TALKNET_DIR)


def load_talknet(pretrain_path: str = None):
    """Load TalkNet model."""
    if pretrain_path is None:
        pretrain_path = os.path.join(TALKNET_DIR, "pretrain_TalkSet.model")
    
    from talkNet import talkNet
    model = talkNet()
    model.loadParameters(pretrain_path)
    model.eval()
    return model


def extract_face_crops(video_path: str, start_time: float, end_time: float,
                       face_tracks: list, fps: float = 25.0) -> np.ndarray:
    """Extract grayscale 224×224 face crops for a time window.
    
    Args:
        video_path: path to video file
        start_time: start of candidate window (seconds)
        end_time: end of candidate window (seconds)
        face_tracks: list of face track dicts with bbox info
        fps: video frame rate
    
    Returns:
        np.array of shape (N, 224, 224) grayscale face crops
    """
    cap = cv2.VideoCapture(video_path)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    crops = []
    for frame_id in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find face bbox for this frame
        bbox = None
        for track in face_tracks:
            for f in track.get("frames", []):
                if abs(f.get("frame", f.get("frame_number", -1)) - frame_id) <= 1:
                    bbox = f.get("bbox", None)
                    if bbox is None:
                        x, y = f.get("x", 0), f.get("y", 0)
                        w, h = f.get("w", 50), f.get("h", 50)
                        bbox = [int(x), int(y), int(w), int(h)]
                    break
            if bbox:
                break
        
        if bbox is None:
            continue
        
        x, y, w, h = [int(v) for v in bbox[:4]]
        
        # Pad bbox
        pad = int(max(w, h) * 0.4)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        # Convert to grayscale, resize to 224×224, then crop center 112×112
        # (TalkNet demo does: resize to 224, then crop center 112×112)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (224, 224))
        center_crop = resized[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
        crops.append(center_crop)
    
    cap.release()
    
    if not crops:
        return np.array([])
    
    return np.array(crops)


def extract_mfcc(audio_path: str, start_time: float, end_time: float,
                 sr: int = 16000) -> np.ndarray:
    """Extract MFCC features for a time window.
    
    Returns:
        np.array of shape (N, 13) MFCC features
    """
    import python_speech_features
    from scipy.io import wavfile
    
    sr_audio, audio = wavfile.read(audio_path)
    if sr_audio != sr:
        # Resample
        import librosa
        audio = librosa.resample(audio.astype(float), orig_sr=sr_audio, target_sr=sr)
        sr_audio = sr
    
    start_sample = int(start_time * sr_audio)
    end_sample = int(end_time * sr_audio)
    audio_segment = audio[start_sample:end_sample]
    
    if len(audio_segment) < sr_audio * 0.1:  # < 100ms
        return np.array([])
    
    mfcc = python_speech_features.mfcc(audio_segment, sr_audio, numcep=13, winlen=0.025, winstep=0.010)
    return mfcc


def run_asd(model, face_crops: np.ndarray, mfcc_features: np.ndarray) -> list:
    """Run TalkNet ASD on face crops + audio features.
    
    Args:
        model: loaded TalkNet model
        face_crops: (N, 224, 224) grayscale face crops
        mfcc_features: (M, 13) MFCC features
    
    Returns:
        list of per-frame ASD scores (>0 = speaking)
    """
    import torch
    
    if len(face_crops) == 0 or len(mfcc_features) == 0:
        return []
    
    # Align lengths: video at 25fps, audio at 100fps (10ms step)
    video_len = face_crops.shape[0]
    audio_len = mfcc_features.shape[0]
    
    # TalkNet expects: audio at 100fps, video at 25fps
    # So audio should be 4x video length
    expected_audio_len = video_len * 4
    if audio_len < expected_audio_len:
        # Pad audio
        pad_len = expected_audio_len - audio_len
        mfcc_features = np.pad(mfcc_features, ((0, pad_len), (0, 0)), mode='constant')
    elif audio_len > expected_audio_len:
        mfcc_features = mfcc_features[:expected_audio_len]
    
    # Run TalkNet with multiple duration windows for reliability
    durationSet = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
    all_scores = []
    
    length = video_len / 25.0  # duration in seconds
    
    for duration in durationSet:
        batch_size = int(math.ceil(length / duration))
        scores = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Audio: 100 frames per second
                audio_start = i * duration * 100
                audio_end = (i + 1) * duration * 100
                input_a = torch.FloatTensor(mfcc_features[int(audio_start):int(audio_end), :]).unsqueeze(0)
                
                # Video: 25 frames per second
                video_start = i * duration * 25
                video_end = (i + 1) * duration * 25
                v_start = int(video_start)
                v_end = min(int(video_end), video_len)
                
                if v_end <= v_start or input_a.shape[1] == 0:
                    continue
                
                input_v = torch.FloatTensor(face_crops[v_start:v_end, :, :]).unsqueeze(0)
                
                # Forward pass
                embed_a = model.model.forward_audio_frontend(input_a)
                embed_v = model.model.forward_visual_frontend(input_v)
                embed_a, embed_v = model.model.forward_cross_attention(embed_a, embed_v)
                out = model.model.forward_audio_visual_backend(embed_a, embed_v)
                score = model.lossAV.forward(out, labels=None)
                scores.extend(score)
        
        all_scores.append(scores)
    
    # Average across all duration windows
    if not all_scores or not all_scores[0]:
        return []
    
    # Pad to same length
    max_len = max(len(s) for s in all_scores)
    padded = []
    for s in all_scores:
        if len(s) < max_len:
            s.extend([s[-1]] * (max_len - len(s)))
        padded.append(s[:max_len])
    
    avg_scores = np.round(np.mean(np.array(padded), axis=0), 2).tolist()
    return avg_scores


def evaluate_binding_with_asd(video_path: str, audio_path: str,
                               face_tracks: list, speaker_segments: list,
                               model=None) -> list:
    """Evaluate face-voice binding using ASD.
    
    For each speaker segment:
    1. Find overlapping face tracks
    2. Run TalkNet on each face track during that segment
    3. The face with highest ASD score is the active speaker
    
    Returns:
        list of binding evidence with ASD scores
    """
    if model is None:
        model = load_talknet()
    
    import subprocess
    sr = 16000
    
    # Ensure audio is 16kHz mono WAV
    if not audio_path.endswith('.wav') or True:
        mono_path = audio_path.replace('.wav', '_mono.wav')
        subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-ac', '1', '-ar', str(sr), mono_path],
                       capture_output=True)
        audio_path = mono_path
    
    evidence = []
    
    for seg in speaker_segments:
        seg_start = seg.get("start", seg.get("start_time", 0))
        seg_end = seg.get("end", seg.get("end_time", 0))
        
        # Find overlapping face tracks
        overlapping = []
        for track in face_tracks:
            t_start = track.get("time_range", [0, 0])[0]
            t_end = track.get("time_range", [0, 0])[1]
            
            overlap_start = max(seg_start, t_start)
            overlap_end = min(seg_end, t_end)
            
            if overlap_end > overlap_start:
                overlapping.append(track)
        
        if not overlapping:
            continue
        
        # Run ASD for each overlapping face track
        for track in overlapping:
            face_crops = extract_face_crops(video_path, seg_start, seg_end, [track])
            mfcc = extract_mfcc(audio_path, seg_start, seg_end)
            
            if len(face_crops) == 0 or len(mfcc) == 0:
                continue
            
            scores = run_asd(model, face_crops, mfcc)
            
            if not scores:
                continue
            
            # Average ASD score for this face during this segment
            avg_score = float(np.mean(scores))
            
            # Smoothed score (±2 frames window)
            smoothed = []
            for i in range(len(scores)):
                window = scores[max(0, i-2):min(len(scores), i+3)]
                smoothed.append(float(np.mean(window)))
            avg_smoothed = float(np.mean(smoothed))
            
            evidence.append({
                "face_track_id": track.get("track_id"),
                "speaker_id": seg.get("speaker_id"),
                "segment_start": seg_start,
                "segment_end": seg_end,
                "duration_ms": int((seg_end - seg_start) * 1000),
                "asd_score": round(avg_score, 3),
                "asd_score_smoothed": round(avg_smoothed, 3),
                "num_frames": len(scores),
                "speaking_frames": sum(1 for s in scores if s > 0),
                "speaking_ratio": round(sum(1 for s in scores if s > 0) / len(scores), 3) if scores else 0,
                "evidence_type": "active_speaker_detection",
                "strength": "strong",
                "asserts_identity": False,
                "status": "Candidate",
            })
    
    return evidence


if __name__ == "__main__":
    import json
    
    # Test on AVDIAR Seq05
    base = str(Path.home() / "dev" / "face-voice-poc")
    seq_dir = f"{base}/datasets/avdiar/Seq05-2P-S1M0/Seq05-2P-S1M0"
    
    video = f"{seq_dir}/Video/Seq05-2P-S1M0_CAM1.mp4"
    audio = f"{base}/output/avdiar_seq05/audio.wav"
    
    # Parse GT face bboxes into tracks
    gt_faces = {}
    with open(f"{seq_dir}/GroundTruth/face_bb.txt") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                frame_id = int(parts[0])
                face_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                gt_faces.setdefault(face_id, []).append({"frame": frame_id, "bbox": [x, y, w, h]})
    
    # Build tracks
    tracks = []
    for face_id, frames in gt_faces.items():
        if len(frames) < 10:
            continue
        tracks.append({
            "track_id": face_id,
            "frames": frames,
            "time_range": [frames[0]["frame"] / 25.0, frames[-1]["frame"] / 25.0],
        })
    
    # Parse GT speakers
    speakers = []
    with open(f"{seq_dir}/GroundTruth/speakers.rttm") as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                speakers.append({
                    "speaker_id": parts[7],
                    "start": float(parts[3]),
                    "end": float(parts[3]) + float(parts[4]),
                })
    
    # Run ASD on first 5 speaker segments
    print("Loading TalkNet...")
    model = load_talknet()
    
    print(f"Running ASD on {min(5, len(speakers))} segments...")
    evidence = evaluate_binding_with_asd(video, audio, tracks, speakers[:5], model)
    
    print(f"\nASD Results: {len(evidence)} bindings")
    for ev in evidence:
        print(f"  {ev['speaker_id']} ↔ face_{ev['face_track_id']}: "
              f"score={ev['asd_score']:.3f} smoothed={ev['asd_score_smoothed']:.3f} "
              f"speaking={ev['speaking_ratio']:.0%}")
