# Face-Voice Binding: Final Design v6 (Production Reference)

> **Status:** PoC validated · Codex-reviewed (3 rounds) · Colleague-reviewed · Corrected
> **Date:** 2026-06-08 · v6.1
> **Goal:** Correct global face-embedding ↔ voice-embedding binding for FinDIT
> **Repo:** https://github.com/Findit-AI/face-voice-poc

---

## 1. Core Principles

### 1.1 Evidence, not assertion

All model outputs are **evidence**, never identity proof.

```
temporal overlap      ≠  this person is speaking
single visible face   ≠  high-confidence binding
lip sync score        ≠  identity proof
speaker segmentation  ≠  speaker identity
```

BindingEvidence always has:
```
status: candidate | accepted | rejected | uncertain
asserts_identity: false
```

### 1.2 False merges are worse than false splits

In a global identity system, the most severe error is:

```
Two different people incorrectly merged into one Person
```

Compared to:

```
Same person split into two Persons  (easier to fix later)
```

Early strategy: **conservative**. Prefer creating more Persons over risking incorrect merges.

### 1.3 Local labels are not global identity

Diarization outputs (`SPEAKER_00`, `SPEAKER_01`) are valid **only within their source video/audio**. They must never be used as global speaker identity.

Correct path:
```
LocalSpeakerCluster → WeSpeaker embedding → GlobalVoiceIndex → GlobalIdentityResolver
```

### 1.4 Tracking never crosses shot boundaries

```
face tracking  → NEVER crosses cuts (bbox trajectory discontinuous)
face identity  → MAY cross cuts (embedding-level recovery)
```

---

## 2. Four-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  Layer 4: Person Resolution                          │
│  Person · BindingEvidence · ConflictRecord           │
│  HumanReviewTask · IdentityMergeEvent/SplitEvent     │
├─────────────────────────────────────────────────────┤
│  Layer 3: Global Biometric                           │
│  GlobalFaceIndex · GlobalVoiceIndex                  │
│  FaceTemplate · VoiceTemplate · CandidateMatch       │
├─────────────────────────────────────────────────────┤
│  Layer 2: Local Identity                             │
│  LocalFaceCluster · LocalSpeakerCluster              │
│  FaceFingerprint · VoiceFingerprint                  │
├─────────────────────────────────────────────────────┤
│  Layer 1: Observation                                │
│  ShotBoundary · FaceDetection · FaceTrack            │
│  SpeechSegment · SpeakerTurn · ASRSegment            │
└─────────────────────────────────────────────────────┘
```

**Each layer only communicates upward.** Observations don't know about Persons. Local clusters don't know about global templates. Only the Person Resolution layer makes identity decisions.

### 2.1 Observation Layer

Records what was seen, makes no identity judgments.

| Type | Source | Output |
|------|--------|--------|
| ShotBoundary | PySceneDetect | cut timestamps |
| FaceDetection | Apple Vision | bbox + confidence + landmarks |
| FaceTrack | VNTrackObjectRequest | continuous track at native fps |
| SpeechSegment | CoreML pyannote-3.0 | start/end speaker labels |
| ASRSegment | WhisperX/ASRY | text + timestamps |

### 2.2 Local Identity Layer

Groups observations within a single video.

| Type | Algorithm | Output |
|------|-----------|--------|
| LocalFaceCluster | DBSCAN on AdaFace embeddings | face groups per video |
| LocalSpeakerCluster | DBSCAN on WeSpeaker embeddings | speaker groups per video |
| FaceFingerprint | quality-weighted centroid | 512D vector → LanceDB |
| VoiceFingerprint | quality-weighted centroid | 256D vector → LanceDB |

**LocalFaceCluster / LocalSpeakerCluster are only valid within their source video.**

### 2.3 Global Biometric Layer

Cross-video matching, but does NOT decide final Person identity.

| Component | Algorithm | Output |
|-----------|-----------|--------|
| GlobalFaceIndex | DIA Clusterer (online) | top-k face candidates |
| GlobalVoiceIndex | DIA Clusterer (online) | top-k voice candidates |
| CandidateMatch | threshold + margin | candidate_existing_template / candidate_new_template / ambiguous |

### 2.4 Person Resolution Layer

Final identity decisions, with conflict handling and human review.

| Component | Purpose |
|-----------|---------|
| Person | modality-neutral identity anchor |
| BindingEvidence | face ↔ voice association evidence |
| ConflictRecord | when evidence contradicts |
| HumanReviewTask | uncertain cases → human |
| IdentityMergeEvent | records every merge (for rollback) |
| IdentitySplitEvent | records every split (for rollback) |

---

## 3. Component Selection (Validated by PoC)

### 3.1 Face Detection: Apple Vision

| Item | Detail |
|------|--------|
| **Tool** | `VNDetectFaceLandmarksRequest` |
| **Measured** | 10ms/frame (640×360) |
| **License** | macOS built-in |
| **PoC** | ✅ Kino + MrBeast videos |

### 3.2 Face Tracking: Apple Vision VNTrackObjectRequest

| Item | Detail |
|------|--------|
| **Tool** | `VNTrackObjectRequest` |
| **Note** | Never cross shot boundaries. Redetect periodically. |

### 3.3 Face Embedding: AdaFace IR-101 ONNX

| Item | Detail |
|------|--------|
| **Tool** | AdaFace IR-101 (ONNX, standalone) |
| **Source** | [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) · [ONNX](https://huggingface.co/ludovitkramar/adaface-ir101-onnx) |
| **Dimension** | 512D |
| **Measured** | 113ms/face (aligned 112×112) |
| **License** | ✅ MIT |
| **PoC** | ✅ 9 clusters at eps=0.5 (IF: 10), correlation 0.89 |

**Why not InsightFace:** non-commercial license. **Why not IR-18:** insufficient discrimination (3 clusters).

**Preprocessing:** BGR input, `(x/255 - 0.5) / 0.5`, 5-point alignment recommended.

### 3.4 Face Clustering: DBSCAN (local) / DIA Clusterer (global)

| Scope | Algorithm | Parameters |
|-------|-----------|------------|
| Local (per video) | DBSCAN, cosine | eps=0.5-0.8, min_samples=1 (PoC) / 2-3 (production) |
| Global (cross-video) | DIA online Clusterer | threshold + margin |

### 3.5 Speaker Segmentation: CoreML pyannote-3.0

| Item | Detail |
|------|--------|
| **Tool** | CoreML SpeakerSegmentation |
| **Source** | [john-rocky/CoreML-Models](https://github.com/john-rocky/CoreML-Models) |
| **Model** | `SpeakerSegmentation.mlpackage` (5.8MB) |
| **Measured** | 12.4ms per 10s clip |
| **License** | MIT |
| **Note** | Bypass pyannote Python pipeline. Load model directly. |

### 3.6 Speaker Embedding: WeSpeaker ResNet34-LM ONNX ⭐

| Item | Detail |
|------|--------|
| **Tool** | WeSpeaker ResNet34-LM (ONNX) |
| **Source** | [onnx-community/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM) · DIA integration |
| **Model** | `wespeaker_resnet34_lm.onnx` (25MB) |
| **Dimension** | 256D |
| **Input** | fbank [200, 80] (2s window, 16kHz) |
| **Measured** | 195ms/chunk |
| **License** | MIT |
| **PoC** | ✅ gap=0.244 (1.8x better than ECAPA-TDNN's 0.138) |
| **DIA integration** | ✅ Rust ONNX Runtime, sliding window, weighted embedding, masked embedding |

**Why WeSpeaker over ECAPA-TDNN:**
- Better discrimination: gap 0.244 vs 0.138 (1.8x)
- Already integrated in DIA (Rust, ONNX Runtime, fbank, sliding window, L2 norm)
- Smaller model: 25MB vs 80MB
- Used by pyannote internally

**ECAPA-TDNN ONNX remains as benchmark reference** (192D, 51ms, gap=0.138). Not used in production pipeline.

### 3.7 Voice Clustering: DIA Clusterer (online)

DIA's `Clusterer` implements online streaming speaker clustering:

```rust
// For each new embedding:
// 1. Compute cosine similarity to all existing centroids
// 2. If best_sim >= threshold → assign to that speaker, update centroid
// 3. Else → create new speaker
// Centroid update: RollingMean or EMA (configurable)
```

This replaces DBSCAN for global voice clustering. DBSCAN remains for local (per-video) clustering.

### 3.8 Shot Detection: PySceneDetect

| Item | Detail |
|------|--------|
| **Measured** | Plane: 7.0s (384 shots), Car: 9.5s (470 shots) |
| **License** | Apache 2.0 |

### 3.9 Vector Storage: LanceDB

Already referenced by mediaschema's `VoiceFingerprint.vector_id`. Embedded, no server needed.

---

## 4. Voice Pipeline (WeSpeaker)

### 4.1 Voice Quality Filtering (before embedding)

All segments must pass quality filter before entering WeSpeaker:

```
duration >= 2.0s                    ← 1.5s is too short for stable identity
non_overlap = true                  ← skip overlapping speech
speech_confidence >= threshold
energy >= threshold
```

For global matching confidence:
```
cumulative speech per LocalSpeakerCluster >= 5-10s
high-confidence global match >= 10-20s cumulative
```

**300ms segments are for VAD/diarization only, NOT for speaker identity embedding.**

### 4.1b Face Quality Filtering (before embedding)

Equivalent rigor to voice quality filtering. All face crops must pass before entering AdaFace:

```
crop_size >= 40px               ← minimum usable face size
blur_score <= threshold         ← reject motion blur / out of focus
pose_angle <= 45°               ← reject extreme profile / tilt
occlusion_ratio <= 0.3          ← reject heavy occlusion (mask, hand, object)
vision_confidence >= 0.5        ← Apple Vision detection confidence
duplicate_iou >= 0.9            ← reject near-duplicate frames in same track
```

For global matching confidence:
```
usable_crops_per_track >= 3     ← minimum for reliable track embedding
high-confidence global match >= 5 usable crops
```

**Why equivalent rigor:** voice has duration/SNR/overlap filters; face needs crop quality/pose/blur filters. Asymmetric quality gates lead to asymmetric error rates.

### 4.2 Segment-Level Embedding Storage

Save every segment's WeSpeaker embedding, not just centroids:

```json
{
  "voice_embedding_id": "vemb_001",
  "video_id": "video_001",
  "segment_id": "seg_019",
  "local_speaker_cluster_id": "lspk_003",
  "vector_id": "lancedb_vec_001",
  "model_id": "wespeaker_resnet34",
  "dimension": 256,
  "normalization": "l2",
  "quality": {
    "duration": 2.8,
    "snr": 18.7,
    "overlap": false,
    "speech_confidence": 0.91
  }
}
```

**Why save segment-level (not just centroid):**
- Can change pooling method later
- Can delete contaminated samples
- Can re-cluster
- Can recompute centroid
- Can rollback incorrect merges
- Can do model migration

### 4.3 Quality-Weighted Centroid

Do NOT simple-average all embeddings. Use quality-weighted pooling:

```python
weight = duration × speech_confidence × non_overlap_weight × snr_weight

def weighted_centroid(embeddings, weights):
    v = sum(w * e for e, w in zip(embeddings, weights)) / sum(weights)
    return l2_normalize(v)
```

DIA's Clusterer supports:
- **RollingMean**: sum all embeddings, normalize (stable)
- **EMA**: exponential moving average (more weight on recent, flexible)

### 4.4 Open-Set Matching

Do NOT use single threshold. Use three-zone matching:

```
if top1_distance <= accept_distance
   AND (top2_distance - top1_distance) >= margin_distance
   AND local_cluster_quality >= quality_threshold:
    → matched

elif top1_distance >= reject_distance:
    → candidate_new_template

else:
    → ambiguous → human review
```

Output:
```json
{
  "top_candidates": [
    {"global_template_id": "gvoice_102", "distance": 0.21},
    {"global_template_id": "gvoice_087", "distance": 0.34}
  ],
  "decision": "matched",
  "margin": 0.13,
  "confidence": 0.86
}
```

---

## 5. BindingEvidence (Refactored)

### 5.1 Evidence Types

| Type | Strength | Description |
|------|----------|-------------|
| `temporal_overlap` | Weak | face visible while voice present |
| `only_visible_face_candidate` | Weak | only one face visible (NOT high confidence) |
| `active_speaker_detection` | Strong | ASD score from TalkNet/lip-sync |
| `voice_similarity` | Strong | cross-video voice match |
| `face_similarity` | Strong | cross-video face match |

### 5.2 Rules

- `only_visible_face_candidate` ≠ high-confidence binding (could be reaction shot, off-screen voice, B-roll)
- Single ASD evidence ≠ identity proof
- All evidence has `asserts_identity: false`
- Only `GlobalIdentityResolver` can create Person links

### 5.3 Active Speaker Detection (future)

Input: candidate window + face crop sequence + audio
Output:

```json
{
  "evidence_type": "active_speaker_detection",
  "raw_score": 2.41,
  "score_margin": 1.13,
  "valid_frames": 96,
  "status": "candidate",
  "asserts_identity": false
}
```

---

## 6. Conflict Handling

Do NOT use hard exclusion (e.g., "speaker A already bound to face X, so can't bind to face Y"). This propagates errors.

Use **evidence graph** instead:

```
speaker A -- evidence 0.82 --> face X
speaker A -- evidence 0.43 --> face Y
speaker A -- evidence 0.12 --> face Z
```

Let the resolver judge holistically.

### Common Conflicts

| Type | Description |
|------|-------------|
| voice_to_multiple_faces | same voice cluster bound to multiple face clusters |
| face_to_multiple_voices | same face cluster bound to multiple voice clusters |
| asd_voice_mismatch | ASD evidence contradicts voice similarity |
| face_voice_contradiction | face match says one person, voice match says another |

### Conflict Record

```json
{
  "conflict_id": "conf_001",
  "type": "voice_to_multiple_faces",
  "entities": {
    "voice_template_id": "gvoice_001",
    "candidate_face_person_ids": ["person_001", "person_019"]
  },
  "evidence_ids": ["bev_001", "bev_031"],
  "severity": "high",
  "status": "needs_review"
}
```

---

## 7. Human Review Queue

Cases requiring human review:

- top1/top2 margin too small
- voice match high but face binding weak
- face match high but voice match weak
- only visible face, no ASD
- multi-face scene, ASD scores close
- global merge would affect many historical records
- same Person has conflicting evidence

```json
{
  "task_id": "hrt_001",
  "type": "confirm_face_voice_binding",
  "evidence_summary": {
    "num_segments": 4,
    "total_duration": 18.2,
    "avg_asd_score": 2.1,
    "voice_distance": 0.23,
    "face_distance": 0.18
  },
  "status": "pending"
}
```

---

## 8. Rollback / Split

Every identity merge must be recorded for rollback.

```json
{
  "merge_event_id": "merge_001",
  "from_entities": ["person_018", "gvoice_034"],
  "to_person_id": "person_002",
  "reason": "face_voice_binding",
  "evidence_ids": ["bev_001", "bev_017"],
  "confidence": 0.91,
  "created_by": "auto_resolver_v1"
}
```

When error discovered:
1. Append IdentitySplitEvent (never delete merge events — append-only)
2. Recompute affected centroids from source embeddings (which are never deleted)
3. Re-evaluate downstream Person resolutions
4. Re-run batch reclustering if needed

**Append-only principle:** merge/split events are never physically deleted. All state is recomputable from the event log + source embeddings. This makes rollback lossless and auditable.

### Resolver Policy (explicit rules)

The GlobalIdentityResolver must follow these constraints:

```
MIN_EVIDENCE_FOR_AUTO_MERGE = 3       // need ≥3 pieces of evidence
MIN_CUMULATIVE_DURATION = 10s         // need ≥10s of speech/face evidence
MAX_CONFLICT_SEVERITY = "medium"      // high conflicts → always human review
MONOTONICITY = true                   // once a Person is confirmed by human, can't auto-split
PROPOSAL_ONLY = true                  // until false-merge rate < threshold
```

**Forbidden auto-merge cases:**
- Single evidence source only
- Conflicting face and voice matches
- Low-quality embeddings (below quality threshold)
- First-ever merge for a global template (needs human confirmation)
- Merge would affect >10 existing records

---

## 9. Model Versioning

All embeddings MUST record:

```
model_id                    "wespeaker_resnet34" / "adaface_ir101"
model_hash                  sha256 of model file
dimension                   256 / 512
distance_metric             cosine
normalization               l2
preprocessing_version       "audio_16k_mono_v1" / "face_aligned_112_v1"
created_at                  timestamp
source_observation_ids      [list of source IDs]
```

**Why:** different model versions produce incompatible embedding spaces. LanceDB must distinguish different vector collections. Enables migration, re-clustering, rollback.

---

## 10. mediaschema Integration

### Existing (no changes)

- `Person` — modality-neutral identity anchor
- `Speaker` — per-track speaker, `person_id: Option<Id>`
- `VoiceFingerprint` — vector_id → LanceDB, dimensions, provenance
- `FaceDetection` — bbox, confidence, euler angles

### New (additive, backward-compatible)

```rust
FaceFingerprint {
    vector_id: Id,
    dimensions: u32,         // 512
    model_id: SmolStr,       // "adaface_ir101"
    model_hash: SmolStr,
    preprocessing_version: SmolStr,
    extracted_at: Timestamp,
    confidence: Option<f32>,
    provenance: Provenance,
}

LocalFaceCluster {
    id: Id,
    video_id: Id,
    track_ids: Vec<Id>,
    faceprint: FaceFingerprint,      // centroid
    source_embedding_ids: Vec<Id>,   // individual embeddings
    pooling_method: SmolStr,         // "quality_weighted"
    quality_score: f32,
}

LocalSpeakerCluster {
    id: Id,
    video_id: Id,
    segment_ids: Vec<Id>,
    voiceprint: VoiceFingerprint,    // centroid
    source_embedding_ids: Vec<Id>,
    pooling_method: SmolStr,
    quality_score: f32,
    total_speech_duration: f32,
    num_segments: u32,
}

BindingEvidence {
    id: Id,
    face_cluster_id: Id,
    speaker_cluster_id: Id,
    video_id: Id,
    evidence_type: SmolStr,          // "temporal_overlap" | "only_visible_face_candidate"
                                     // | "active_speaker_detection" | "voice_similarity" | "face_similarity"
    // For global evidence types (voice_similarity, face_similarity):
    global_template_id: Option<Id>,  // reference to global face/voice template
    global_person_candidate: Option<Id>,  // reference to candidate Person
    segment: TimeRange,
    raw_score: Option<f32>,
    score_margin: Option<f32>,
    visible_face_fraction: f32,
    num_visible_faces: u32,
    duration_ms: u64,
    status: BindingStatus,           // Candidate | Accepted | Rejected | Uncertain
    asserts_identity: bool,          // always false
    reviewed_by: Option<SmolStr>,
    provenance: Provenance,
}

ConflictRecord { ... }              // see Section 6
HumanReviewTask { ... }             // see Section 7
IdentityMergeEvent { ... }          // see Section 8
IdentitySplitEvent { ... }          // see Section 8
```

### Production (future)

```rust
PersonBiometricTemplate {
    person_id: Id,
    modality: Modality,              // Face | Voice
    vector_id: Id,
    model_id: SmolStr,
    model_hash: SmolStr,
    preprocessing_version: SmolStr,
    dimension: u32,
    distance_metric: SmolStr,
    quality: f32,
    source_observation_ids: Vec<Id>,
    created_at: Timestamp,
}
```

One person can have multiple face/voice templates (different angles, ages, environments).

---

## 11. Performance (All Measured)

### Apple M1, 8GB RAM, CPU-only

| Step | Tool | Measured | 12min video |
|------|------|----------|-------------|
| 1. Shot Detection | PySceneDetect | 7-10s | ~8s |
| 2. Face Detection | Apple Vision (2fps) | 10ms/frame | ~10s |
| 3. Face Tracking | VNTrackObjectRequest | ~5ms/frame | ~5s |
| 4. Face Embedding | AdaFace IR-101 ONNX | 113ms/face | ~30s |
| 5. Face Clustering | DBSCAN | 3.2ms/100 | <0.1s |
| 6. Speaker Segmentation | CoreML pyannote-3.0 | **12.4ms/10s** | **~1.5s** |
| 7. Speaker Embedding | WeSpeaker ONNX | **195ms**/chunk | **~10s** |
| 8. Speaker Clustering | DBSCAN (local) / DIA (global) | <10ms | <0.1s |
| 9. Binding Assembly | Python | <1ms | <0.1s |
| **TOTAL** | | | **~65s** |

### Memory

| Model | Files | Loaded |
|-------|-------|--------|
| AdaFace IR-101 ONNX | 249MB | ~400MB |
| CoreML SpeakerSegmentation | 5.8MB | ~50MB |
| WeSpeaker ONNX | 25MB | ~100MB |
| Apple Vision | 0 | 0 |
| **Total** | | **~0.85GB** |

---

## 12. Licensing Summary

| Component | License | Production OK |
|-----------|---------|---------------|
| Apple Vision | macOS built-in | ✅ |
| AdaFace IR-101 ONNX | MIT | ✅ |
| CoreML SpeakerSegmentation | MIT | ✅ |
| WeSpeaker ResNet34-LM | MIT | ✅ |
| DIA Clusterer | MIT/Apache-2.0 | ✅ |
| PySceneDetect | Apache 2.0 | ✅ |
| LanceDB | Apache 2.0 | ✅ |
| mediaschema | FinDIT internal | ✅ |

**All components MIT or permissive license. No non-commercial restrictions.**

---

## 12b. Risks and Mitigations

### Biometric Privacy / Compliance
- Face and voice embeddings are biometric data
- Must handle consent, retention, deletion, audit logs
- Jurisdiction-specific rules (GDPR, BIPA, etc.)
- **Mitigation:** all embeddings have source_observation_ids; deletion cascades to embeddings

### Spoofing / Synthetic Media
- Voice cloning, replayed audio, deepfake faces, dubbed clips
- Can create high-confidence false bindings
- **Mitigation:** ASD evidence helps (lip-sync hard to fake); synthetic detection as future layer

### Bias and Domain Shift
- Lighting, age, gender, accent, language, compression, masks, makeup, camera angle
- Thresholds calibrated on one domain may fail on another
- **Mitigation:** per-domain calibration; evaluation set must cover diverse conditions

### Online Clustering Order Dependence
- DIA-style centroid updates are order-dependent: A→B→C may cluster differently than C→A→B
- **Mitigation:** periodic offline reclustering/reconciliation of global indexes

### Threshold Calibration Gap
- PoC gap=0.244 and correlation=0.89 are useful signals, NOT production thresholds
- **Mitigation:** production thresholds derived from evaluation set, not PoC benchmarks

### Evaluation Set Size
- 20-30 annotated clips is sufficient for smoke tests, NOT for false-merge guarantees
- **Mitigation:** start with 20-30 clips for PoC validation, expand to 100+ for production calibration

---

## 13. Key PoC Findings

1. **128D embeddings categorically insufficient** for face clustering (SFace, Dlib, FaceNet all fail)
2. **512D is the minimum** for reliable face clustering
3. **AdaFace IR-101 ONNX replaces InsightFace** — MIT, equivalent quality (9 vs 10 clusters), correlation 0.89
4. **AdaFace IR-18 CoreML too weak** — only 3 clusters (insufficient discrimination)
5. **pyannote Python pipeline is 99% overhead** — bypassing gives 89x speedup
6. **WeSpeaker beats ECAPA-TDNN** — gap 0.244 vs 0.138 (1.8x better discrimination)
7. **OpenVoice speaker encoder cannot distinguish speakers** — designed for voice conversion
8. **Temporal overlap ≠ face-voice binding** — only weak evidence
9. **Single visible face ≠ high confidence** — could be reaction shot / off-screen voice
10. **No existing production system** combines face + voice identity — genuine gap

---

## 14. Production Roadmap (Reordered per Codex Review)

**Principle: build safety nets before enabling global matching.**

| Phase | Content | Status |
|-------|---------|--------|
| **P0** | ~~Unify voice to WeSpeaker~~ | ✅ Done (gap=0.244) |
| **P1** | ~~AdaFace IR-101 as face embedding~~ | ✅ Done (MIT, 9 clusters) |
| **P2** | mediaschema schema: FaceFingerprint, model metadata, provenance | Next |
| **P3** | Segment-level embedding storage (voice + face) | Next |
| **P4** | Voice + face quality filtering | Next |
| **P5** | Evaluation set (20-30 annotated clips) + calibration harness | Next |
| **P6** | Local clustering metrics (pairwise F1) | Next |
| **P7** | GlobalVoiceIndex + GlobalFaceIndex (read-only / proposal mode) | Next |
| **P8** | BindingEvidence refactoring (all types, no high-confidence single_face) | Next |
| **P9** | Conflict records | Future |
| **P10** | Human review queue | Future |
| **P11** | Rollback/split (append-only event model) | Future |
| **P12** | GlobalIdentityResolver (proposal-only mode) | Future |
| **P13** | Auto-merge (only after measured false-merge rate is acceptable) | Future |
| **P14** | ASD integration (TalkNet) | Future |
| **P15** | Evaluation set expansion (100+ clips for production calibration) | Future |

**Key ordering constraints:**
- P2-P4 (schema + storage + quality) must come before P7 (global indexes)
- P5-P6 (evaluation + metrics) must come before P7 (global indexes are trusted)
- P9-P11 (conflict + review + rollback) must come before P12 (resolver)
- P12 (resolver) must be proposal-only until P13 (auto-merge with measured false-merge rate)

---

## 15. Evaluation Metrics

| Metric | Definition | Priority |
|--------|-----------|----------|
| **False Merge Rate** | % of Person merges that are incorrect | **#1 most important** |
| False Split Rate | % of same-person splits that are incorrect | High |
| Face Clustering F1 | pairwise precision/recall for face clusters | High |
| Speaker Clustering F1 | pairwise precision/recall for speaker clusters | High |
| Cross-video Match Accuracy | correct speaker match across videos | High |
| Binding Precision | % of accepted bindings that are correct | High |
| Binding Recall | % of correct bindings that are found | Medium |
| Uncertain Rate | % of cases requiring human review | Medium |
| Manual Review Burden | human review tasks per video | Medium |
