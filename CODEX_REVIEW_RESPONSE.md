# Codex Review — Response and Discussion

## Context

We are building a face-voice binding system for FinDIT, a video indexing product.
Our ultimate goal: correct global face-embedding ↔ voice-embedding binding.

## Codex's Key Critiques — Our Response

### 1. "Active-speaker association is not proof of person identity"

**We agree.** We will adopt your `BindingEvidence` approach:
- Store binding as evidence, not identity assertion
- Status: candidate | accepted | rejected
- Multiple evidence segments needed before accepting
- Human review for high-impact merges

### 2. "One face visible = high confidence is false"

**We agree.** We will:
- Remove the "single face = high confidence" rule
- Require ASD score above absolute threshold even when only one face
- Add explicit states: off_screen, ambiguous, overlapping, unbound

### 3. "Hard exclusion propagates errors"

**We agree.** We will:
- Replace exclusion with evidence accumulation in a bipartite graph
- Allow contradictory evidence (same speaker bound to different faces in different segments)
- Resolve conflicts through evidence strength, not first-come-first-served

### 4. "1fps cannot feed TalkNet"

**We agree on the principle, but need to discuss implementation.**

Our constraint: Apple M1, 8GB RAM, CPU only, no GPU.

Codex suggests:
- Sparse detection at 2-5fps with Apple Vision
- Track faces between detections (IoU tracking)
- Dense crops at source frame rate for ASD candidate windows only

**Our question:** Apple Vision does NOT provide face embeddings (confirmed: VNFacePrintDescriptor is private API). So we need InsightFace for embeddings. The pipeline would be:

1. Apple Vision detection (2-5fps, ~15ms/frame) → face bboxes
2. IoU tracking between detections
3. InsightFace embedding on best crops from each track (not every frame)
4. For ASD candidates: extract dense crops from tracked bboxes at 24fps
5. TalkNet on dense crops

Is this the right separation? Apple Vision for detection speed, InsightFace for embedding quality?

### 5. "Performance numbers contradict"

**Corrected:** The 23-minute figure was for full-frame InsightFace detection at every frame (1800 frames at ~0.7s each). At 1fps sampling (105 frames), it should be ~15 seconds. We will re-instrument with proper stage-by-stage timing.

### 6. "Face clustering validation is insufficient"

**We agree.** For the PoC, we will:
- Use pairwise precision/recall as the metric
- Manually label face identities in the test video
- Document: cosine distance, eps=0.8, min_samples=1, L2-normalized embeddings
- Report B-cubed F1 score

### 7. "Embedding dimension is not a quality metric"

**We agree.** We will:
- Stay with ASRY's built-in speaker embeddings for PoC
- Record exact model name, hash, dimension, preprocessing
- Only upgrade if PoC shows insufficient speaker separation

### 8. "Python 3.9 is a constraint"

**We agree.** The ML pipeline should run in a separate Python 3.11+ environment. mediaschema (Rust) remains independent.

### 9. "InsightFace licensing"

**Known issue.** For PoC, acceptable. For production, we have AdaFace IR101 (MIT) as alternative.

## Remaining Questions for Codex

### Q1: PoC Scope

Codex suggests 20-30 annotated clips. We currently have:
- 1 video (105s, cinematic, dark lighting, 3-5 people)
- No labeled evaluation corpus

**Can we do a staged PoC?**
- Stage 1: Run on our existing video, manually verify bindings
- Stage 2: If promising, create labeled evaluation set
- Stage 3: Full precision/recall evaluation

Or does Codex insist on labeled data from the start?

### Q2: mediaschema Integration

mediaschema already has:
- `Person` (modality-neutral identity anchor, voiceprint field)
- `Speaker` (per-track, person_id FK, voiceprint)
- `VoiceFingerprint` VO (vector_id → LanceDB)
- `FaceDetection` (bbox, confidence, euler angles, on keyframes)

Codex suggests NOT using singular Person.faceprint/voiceprint, but instead using BiometricTemplate + BindingEvidence.

**Question:** Should we:
a) Keep mediaschema's existing Person/Speaker/VoiceFingerprint as-is, and add BindingEvidence alongside?
b) Redesign Person to use BiometricTemplate instead of VoiceFingerprint?

Option (a) is additive and backward-compatible. Option (b) is a breaking change.

### Q3: Shot Detection

Codex says "detect shots first, tracks must not cross cuts."

Our video is a single continuous shot (poker table scene). But for general use:
- What shot detection tool should we use? PySceneDetect?
- Should shot boundaries reset face tracking, or just re-verify identity?

### Q4: The Binding Evidence Graph

Codex proposes:
```
LocalFaceCluster F
LocalSpeakerCluster S
BindingEvidence(F, S, segment, method, raw_score, calibrated_probability)
```

**Question:** How do we calibrate probability without labeled data? 
- Can we use a heuristic first (normalized TalkNet score as pseudo-probability)?
- When should we switch to logistic/isotonic calibration?

### Q5: Production Pipeline Architecture

For the final production system (hundreds of videos):
- Should processing be batch (process all videos, then re-cluster) or streaming (process one video at a time, update global clusters)?
- How to handle the "same person in video A and video B" problem without re-processing everything?
