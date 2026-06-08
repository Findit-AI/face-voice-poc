# Face-Voice Binding: Technical Design Review

## Context

We are building a **cross-modal person identity system** for FinDIT, a video indexing pipeline. The goal is to bind face identities (from video) with voice identities (from audio) so that:
1. Given a person's face → find their voice/audio recordings
2. Given a voice → find the person's visual identity

This document is the technical design for review. We want you to critically evaluate:
- Is the approach sound?
- Are there failure modes we haven't considered?
- Is the architecture appropriate?
- Are there better alternatives?

## Current System

**FinDIT** is a video indexer with:
- **mediaschema** (130K lines Rust): Domain-driven schema library with multi-backend support (SQLite/PostgreSQL/MySQL/MongoDB/proto3). Already has `Person`, `Speaker`, `VoiceFingerprint`, `FaceDetection` aggregates.
- **ASRY**: WhisperX wrapper for speaker diarization (pyannote-based)
- **LanceDB**: External vector store for embeddings (referenced by mediaschema's `VoiceFingerprint.vector_id`)
- **Hardware**: Apple M1, 8GB RAM, macOS, no Docker

**Test video**: `Kino Demo Render.mp4` — 1920×1080, 24fps, 105s, cinematic footage with dark lighting, multiple people, 35% of frames have faces.

## Proposed Architecture

```
Video → Frame extraction (1fps)
         ↓
    InsightFace buffalo_l (512D face embeddings)
         ↓
    DBSCAN clustering → face_persons: {P0: 64 faces, P1: 5 faces, ...}

Audio → ASRY/pyannote speaker diarization
         ↓
    speaker_segments: {S0: [21s-25s], S1: [32s-34s], ...}

Video → TalkNet-ASD (Active Speaker Detection)
         ↓
    per-frame scores: {frame_504: {P0: 3.8, P1: -2.1, ...}}

Binding:
    For each speaker_segment:
        Find overlapping face_tracks
        Use TalkNet scores to identify who is speaking
        Confidence = top_score - second_score
        High confidence (>1.0) → bind face_person ↔ speaker
        Exclusion: bound speakers are removed from subsequent matching

Storage:
    mediaschema Person.faceprint (new field) ← face embedding centroid
    mediaschema Person.voiceprint (existing) ← voice embedding centroid
    mediaschema FaceDetection.person_id (new field) ← FK to Person
    LanceDB ← actual vectors
```

## Specific Questions for Review

### 1. Face Detection Pipeline

We use InsightFace buffalo_l (SCRFD-10G detector + ArcFace R50 recognition):
- Detection: 143ms/frame at 640×640 det_size (CPU)
- Recognition: 125ms/face (512D embedding)
- Total: ~270ms first frame + ~125ms per additional face

**Question**: Should we use InsightFace's built-in SCRFD detector, or replace it with Apple's Vision framework for detection (faster on Apple Silicon) and only use InsightFace for embeddings? Apple Vision is ~15ms per frame but does NOT provide face embeddings.

### 2. Voice Embedding Dimension

Current ASRY uses pyannote's internal speaker embeddings (dimension not documented, likely 512D). Research shows:
- ECAPA-TDNN (192D): Good general purpose
- WavLM-Large (768D): Systematically better in 2024 benchmarks
- CAM++ (512D): Lightweight, good multilingual

**Question**: Should we stick with ASRY's built-in embeddings (zero cost) or upgrade to WavLM-Large (higher accuracy but more compute on 8GB M1)?

### 3. TalkNet-ASD Limitations

TalkNet uses S3FD face detector which found 0 faces in our dark cinematic test video. We replaced it with InsightFace's detector in a hybrid pipeline. This worked but:
- Full-frame InsightFace detection at 1fps takes ~23 minutes for 105s video
- TalkNet was designed for talking-head videos, not cinematic footage

**Question**: Is TalkNet the right choice for Active Speaker Detection on cinematic footage with dark lighting, quick cuts, and reaction shots? Or should we use a simpler approach (e.g., audio energy correlation with face presence, without lip movement analysis)?

### 4. Binding Algorithm

Our proposed binding:
1. For each audio speaker segment, find temporally overlapping face tracks
2. If only 1 face present → high confidence binding
3. If multiple faces → use TalkNet ASD scores to disambiguate
4. Confidence = score_difference between top 2 candidates
5. Exclusion: bound pairs removed from subsequent matching

**Question**: Is this statistically sound? What about:
- Off-screen speakers (audio has speech, no face visible)
- Reaction shots (face visible but not speaking)
- Overlapping speech (2+ people speaking simultaneously)
- Edited video (cuts mid-sentence)

### 5. mediaschema Integration

mediaschema already has:
- `Person` aggregate (modality-neutral identity anchor)
- `Speaker` aggregate with `person_id: Option<Id>` FK
- `VoiceFingerprint` VO (vector_id → LanceDB)
- `FaceDetection` struct (bbox, confidence, euler angles)
- Schema explicitly designed for face extension: "FaceDetection.person: Option<Id>"

We plan to add:
- `FaceFingerprint` VO (copy VoiceFingerprint pattern)
- `FaceDetection.person_id` + `FaceDetection.faceprint`
- `Person.faceprint` aggregated field

**Question**: Is the VoiceFingerprint pattern (vector_id pointing to external store) the right abstraction for face embeddings too? Or should face embeddings be stored differently given they have different characteristics (512D fixed, more stable across conditions)?

### 6. Global Identity Across Videos

For cross-video person identification:
- Face embeddings: cluster all face embeddings across all videos → global person IDs
- Voice embeddings: cluster all voice embeddings across all videos → global speaker IDs
- Lip-sync binding: per-video, binds local face clusters to local voice clusters → global mapping

**Question**: Should global clustering use:
a) Incremental online clustering (process each video, update global clusters)
b) Batch re-clustering (process all videos, re-cluster everything)
c) FAISS/annoy index with threshold-based assignment

### 7. Performance Budget

On Apple M1 (8GB RAM, CPU only):
- 1000-frame video: ~57 seconds total (InsightFace being the bottleneck)
- Could optimize with frame skipping + IoU tracking

**Question**: Is this acceptable for an offline video indexing pipeline? Or do we need GPU acceleration? The system will eventually process hundreds of videos.

### 8. Failure Modes

Known risks:
- Dark/blurry faces: InsightFace SCRFD misses faces that Apple Vision detects
- Small faces (<32px): Embedding quality degrades significantly
- 128D models (SFace, Dlib, FaceNet): Proven insufficient for clustering
- Audio-visual sync in edited video: Lip movement may not correspond to the audio track

**Question**: What other failure modes should we anticipate? How should we handle them gracefully?

## What We've Already Validated

1. **Face clustering works**: InsightFace buffalo_l (512D) → 7 clusters at eps=0.8, matching ground truth (~6-8 people)
2. **128D is insufficient**: SFace produces 1 cluster regardless of eps. Dlib finds only 2 faces in 53 frames.
3. **Resolution doesn't affect speed**: Detection speed depends on det_size, not input resolution
4. **TalkNet runs on CPU**: Successfully processed synthetic video, identified main speaker (score +1.92)
5. **Simple face-voice correlation works**: 1fps sampling + energy VAD → P0 (main person) has 84% voice overlap
6. **mediaschema is ready**: Person/Speaker/VoiceFingerprint/FaceDetection all exist, designed for face extension

## Deliverable

Please review this design and tell us:
1. What is wrong or risky in this approach?
2. What are we missing?
3. What would you change?
4. Is this feasible with our constraints (M1 8GB, no Docker, Python 3.9, mediaschema Rust)?
5. What is the simplest version that proves the concept works?
