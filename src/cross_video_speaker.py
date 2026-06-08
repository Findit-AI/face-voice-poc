"""Cross-video speaker embedding extraction and matching."""
import json
import numpy as np
import librosa
from pathlib import Path


def extract_speaker_embeddings(audio_path: str, segments_path: str, output_path: str):
    """Extract speaker embeddings for each diarized segment."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    
    # Load SpeechBrain ECAPA-TDNN
    print("Loading SpeechBrain ECAPA-TDNN...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=Path.home() / ".cache" / "speechbrain" / "ecapa",
    )
    
    y, sr = librosa.load(audio_path, sr=16000)
    
    with open(segments_path) as f:
        segments = json.load(f)
    
    embeddings = []
    for i, seg in enumerate(segments):
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        chunk = y[start:end]
        
        if len(chunk) < sr * 0.3:  # skip < 300ms
            continue
        
        # SpeechBrain expects torch tensor
        tensor = torch.FloatTensor(chunk).unsqueeze(0)
        with torch.no_grad():
            emb = classifier.encode_batch(tensor)
        emb_np = emb.squeeze().numpy()
        emb_normalized = emb_np / np.linalg.norm(emb_np)
        
        embeddings.append({
            "segment_idx": i,
            "speaker_id": seg.get("speaker_id", "unknown"),
            "start": seg["start"],
            "end": seg["end"],
            "duration_ms": seg["duration_ms"],
            "embedding": emb_normalized.tolist(),
        })
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(segments)} segments processed")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(embeddings, f)
    
    print(f"Extracted {len(embeddings)} speaker embeddings (192D)")
    return embeddings


def match_speakers_cross_video(emb1_path: str, emb2_path: str, 
                                clusters1_path: str, clusters2_path: str,
                                output_path: str):
    """Match speakers across two videos using voice embedding similarity."""
    from sklearn.metrics.pairwise import cosine_distances
    
    with open(emb1_path) as f:
        emb1 = json.load(f)
    with open(emb2_path) as f:
        emb2 = json.load(f)
    with open(clusters1_path) as f:
        fc1 = json.load(f)
    with open(clusters2_path) as f:
        fc2 = json.load(f)
    
    # Group embeddings by speaker_id
    def group_by_speaker(embeddings):
        groups = {}
        for e in embeddings:
            sid = e["speaker_id"]
            groups.setdefault(sid, []).append(np.array(e["embedding"]))
        # Centroid per speaker
        centroids = {}
        for sid, embs in groups.items():
            c = np.mean(embs, axis=0)
            centroids[sid] = c / np.linalg.norm(c)
        return centroids
    
    spk1 = group_by_speaker(emb1)
    spk2 = group_by_speaker(emb2)
    
    print(f"Video 1 speakers: {list(spk1.keys())}")
    print(f"Video 2 speakers: {list(spk2.keys())}")
    
    # Cross-video speaker distance matrix
    sids1 = sorted(spk1.keys())
    sids2 = sorted(spk2.keys())
    embs1 = np.array([spk1[s] for s in sids1])
    embs2 = np.array([spk2[s] for s in sids2])
    
    dist = cosine_distances(embs1, embs2)
    
    print(f"\nCross-video speaker distance matrix:")
    header = "         " + "  ".join([f"{s:>12}" for s in sids2])
    print(header)
    for i, s1 in enumerate(sids1):
        row = f"{s1:>8}"
        for j, s2 in enumerate(sids2):
            d = dist[i][j]
            marker = " <<<" if d < 0.3 else "  *" if d < 0.5 else ""
            row += f"  {d:>8.4f}{marker}"
        print(row)
    
    # Find matches
    matches = []
    for i, s1 in enumerate(sids1):
        for j, s2 in enumerate(sids2):
            d = dist[i][j]
            if d < 0.3:
                matches.append({"speaker1": s1, "speaker2": s2, "distance": round(d, 4)})
    
    print(f"\nSpeaker matches (distance < 0.3):")
    for m in matches:
        print(f"  {m['speaker1']} <-> {m['speaker2']}  (distance={m['distance']})")
    
    # Now: connect face clusters to cross-video speaker IDs
    print(f"\n=== Full Cross-Video Identity Chain ===")
    print(f"Face → Speaker (per video) → Voice (cross-video) → Global Person")
    
    # This requires face-speaker bindings from both videos
    # Which we have in candidates.json
    
    result = {
        "speaker_distances": dist.tolist(),
        "speaker_ids_1": sids1,
        "speaker_ids_2": sids2,
        "matches": matches,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    return result


if __name__ == "__main__":
    import sys
    
    print("=== Video 1: Speaker Embeddings ===")
    extract_speaker_embeddings(
        "output/plane/audio.wav",
        "output/plane/speaker_segments.json",
        "output/plane/speaker_embeddings.json",
    )
    
    print("\n=== Video 2: Speaker Embeddings ===")
    extract_speaker_embeddings(
        "output/car/audio.wav",
        "output/car/speaker_segments.json",
        "output/car/speaker_embeddings.json",
    )
    
    print("\n=== Cross-Video Speaker Matching ===")
    match_speakers_cross_video(
        "output/plane/speaker_embeddings.json",
        "output/car/speaker_embeddings.json",
        "output/plane/face_clusters.json",
        "output/car/face_clusters.json",
        "output/cross_video_speakers.json",
    )
