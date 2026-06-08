"""Step 5-6: Speaker diarization (ASRY) + clustering + voice embeddings."""
import json
import hashlib
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List


def diarize_speakers(video_path: str, output_path: str) -> list[dict]:
    """Extract audio and run speaker diarization.
    
    Returns: [{speaker_id, start_ms, end_ms, duration_ms}]
    """
    # Extract audio
    audio_path = str(Path(output_path).parent / "audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ], capture_output=True, timeout=60)
    
    print(f"Audio extracted: {audio_path}")
    
    # Try ASRY first, fall back to pyannote directly
    try:
        segments = _diarize_asry(audio_path)
    except Exception as e:
        print(f"ASRY failed ({e}), trying pyannote directly...")
        segments = _diarize_pyannote(audio_path)
    
    # Extract and store voice embeddings in LanceDB
    try:
        n_stored = _store_voice_embeddings(video_path, audio_path, segments)
        print(f"Stored {n_stored} voice embeddings in LanceDB")
    except Exception as e:
        print(f"Warning: voice embedding storage failed: {e}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2)
    
    print(f"Speaker diarization: {len(segments)} segments")
    speakers = set(s["speaker_id"] for s in segments)
    print(f"  Unique speakers: {len(speakers)}")
    for sid in sorted(speakers):
        segs = [s for s in segments if s["speaker_id"] == sid]
        total_dur = sum(s["duration_ms"] for s in segs)
        print(f"  {sid}: {len(segs)} segments, {total_dur/1000:.1f}s total")
    
    return segments


def _extract_mel_segment(audio_path: str, start_ms: int, end_ms: int,
                         sr: int = 16000, n_mels: int = 80) -> np.ndarray:
    """Extract mel spectrogram segment for WeSpeaker.
    
    Returns: np.ndarray of shape (T, 80) — time-major for WeSpeaker input.
    """
    import librosa
    import soundfile as sf
    
    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0
    duration = end_sec - start_sec
    
    if duration < 0.05:
        return np.array([])
    
    # Load segment
    audio, _ = librosa.load(audio_path, sr=sr, offset=start_sec, duration=duration)
    
    if len(audio) < sr * 0.05:
        return np.array([])
    
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=512, hop_length=160, win_length=400
    )
    # mel shape: (n_mels, T) -> transpose to (T, n_mels)
    mel = mel.T
    
    # Log-mel
    mel = np.log(mel + 1e-6).astype(np.float32)
    
    # Normalize (mean-variance per frequency bin)
    mel = (mel - mel.mean(axis=0, keepdims=True)) / (mel.std(axis=0, keepdims=True) + 1e-6)
    
    return mel


def _embed_voice_wespeaker(audio_path: str, segments: list[dict],
                           model_path: Optional[str] = None) -> list:
    """Extract voice embeddings for each segment using WeSpeaker ONNX.
    
    Args:
        audio_path: path to 16kHz mono WAV
        segments: list of {start_ms, end_ms, ...}
        model_path: path to WeSpeaker ONNX model
    
    Returns:
        list of 256D embedding vectors (as lists of floats)
    """
    import onnxruntime as ort
    
    if model_path is None:
        model_path = str(Path.home() / "dev/dia/models/wespeaker_resnet34_lm.onnx")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"WeSpeaker model not found: {model_path}")
    
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    embeddings = []
    for seg in segments:
        mel = _extract_mel_segment(audio_path, seg["start_ms"], seg["end_ms"])
        if mel.size == 0:
            embeddings.append(None)
            continue
        
        # WeSpeaker expects (B, T, 80)
        mel_batch = mel[np.newaxis, :, :]  # (1, T, 80)
        
        # Run inference
        result = sess.run(None, {"input_features": mel_batch})
        emb = result[0][0]  # (256,)
        
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        
        embeddings.append(emb.tolist())
    
    return embeddings


def _store_voice_embeddings(video_path: str, audio_path: str,
                            segments: list[dict]) -> int:
    """Extract and store voice embeddings in LanceDB.
    
    Args:
        video_path: original video path (for video_id)
        audio_path: extracted audio WAV path
        segments: diarization segments with start_ms, end_ms
    
    Returns:
        number of embeddings stored
    """
    from src.embedding_store import (
        get_db, create_voice_embeddings_table, write_voice_embeddings
    )
    
    db = get_db()
    create_voice_embeddings_table(db)
    video_id = hashlib.md5(video_path.encode()).digest()  # 16 bytes
    
    # Extract embeddings
    embeddings = _embed_voice_wespeaker(audio_path, segments)
    
    # Build records for LanceDB
    records = []
    for i, (seg, emb) in enumerate(zip(segments, embeddings)):
        if emb is None:
            continue
        segment_id = i.to_bytes(16, "big", signed=False)
        records.append({
            "segment_id": segment_id,
            "start_ms": seg["start_ms"],
            "end_ms": seg["end_ms"],
            "vector": emb,
            "duration_ms": seg["duration_ms"],
            "speech_confidence": 0.9,  # diarization confidence placeholder
            "quality_score": min(1.0, seg["duration_ms"] / 1000.0),  # longer = better
        })
        
        # Store segment_id in segment for later linkage
        seg["segment_id"] = segment_id.hex()
    
    if records:
        write_voice_embeddings(db, video_id, records)
    
    return len(records)


def _diarize_asry(audio_path: str) -> list[dict]:
    """Try ASRY (WhisperX wrapper) for diarization."""
    import importlib
    # ASRY may not be directly importable, try common patterns
    try:
        asry = importlib.import_module("asry")
        # ASRY API varies; adapt as needed
        raise ImportError("ASRY direct import not supported yet")
    except ImportError:
        raise


def _diarize_pyannote(audio_path: str) -> list[dict]:
    """Use pyannote directly for speaker diarization."""
    from pyannote.audio import Pipeline
    
    # Use pretrained pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=None  # may need HF token
    )
    
    diarization = pipeline(audio_path)
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker_id": speaker,
            "start_ms": round(turn.start * 1000),
            "end_ms": round(turn.end * 1000),
            "duration_ms": round((turn.end - turn.start) * 1000),
            "start_time": round(turn.start, 3),
            "end_time": round(turn.end, 3),
        })
    
    return segments


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "Desktop/Kino Demo Render.mp4")
    output = sys.argv[2] if len(sys.argv) > 2 else "output/speaker_segments.json"
    diarize_speakers(video, output)
