# Implementation Plan: P2 — mediaschema Schema Additions (v4, Codex-approved)

## Scope

P2 adds schema types and FaceDetection extensions. All additive, backward-compatible.

**P2 includes:**
1. FaceFingerprint VO (new type)
2. VoiceFingerprint model metadata fields (Option<T>)
3. FaceDetection.person_id + FaceDetection.faceprint (Option<T>, resolver-only)
4. LocalFaceCluster aggregate (new type)
5. LocalSpeakerCluster aggregate (new type)
6. SQLite tables for above (NOT BindingEvidence/GlobalBiometricMatch — those are P8)
7. cargo build + cargo test

**P2 does NOT include:** BindingEvidence, BindingStatus, GlobalBiometricMatch, resolver behavior, identity assertions. Those are P8.

## 1. FaceFingerprint VO

```rust
pub struct FaceFingerprint {
    pub vector_id: Id,
    pub dimensions: u32,                    // 512
    pub distance_metric: SmolStr,           // "cosine"
    pub normalization: SmolStr,             // "l2"
    pub model_id: SmolStr,                  // "adaface_ir101"
    pub model_hash: SmolStr,                // sha256, REQUIRED (not Option)
    pub preprocessing_version: SmolStr,     // "face_aligned_112_v1"
    pub extracted_at: Timestamp,
    pub confidence: Option<f32>,
    pub quality_score: Option<f32>,
    pub provenance: Provenance,
}
```

## 2. VoiceFingerprint Extensions

Add to existing VoiceFingerprint (all Option for backward compatibility):

```rust
pub model_id: Option<SmolStr>,
pub model_hash: Option<SmolStr>,
pub preprocessing_version: Option<SmolStr>,
pub distance_metric: Option<SmolStr>,
pub normalization: Option<SmolStr>,
```

## 3. FaceDetection Extensions

```rust
/// FK to Person. Only populated by GlobalIdentityResolver or human review.
/// Observation layer code must NEVER set this field.
pub person_id: Option<Id>,
pub faceprint: Option<FaceFingerprint>,
```

## 4. LocalFaceCluster

```rust
pub struct LocalFaceCluster {
    pub id: Id,
    pub video_id: Id,
    pub track_ids: Vec<Id>,
    pub faceprint: FaceFingerprint,
    pub source_embedding_ids: Vec<Id>,
    pub pooling_method: SmolStr,
    pub quality_score: f32,
    pub clustering_algorithm: SmolStr,
    pub clustering_threshold: f32,
    pub clustering_version: SmolStr,
    pub created_at: Timestamp,
}
```

## 5. LocalSpeakerCluster

```rust
pub struct LocalSpeakerCluster {
    pub id: Id,
    pub video_id: Id,
    pub segment_ids: Vec<Id>,
    pub voiceprint: VoiceFingerprint,
    pub source_embedding_ids: Vec<Id>,
    pub pooling_method: SmolStr,
    pub quality_score: f32,
    pub total_speech_duration_ms: u64,
    pub num_segments: u32,
    pub clustering_algorithm: SmolStr,
    pub clustering_threshold: f32,
    pub clustering_version: SmolStr,
    pub created_at: Timestamp,
}
```

## 6. SQLite Tables (P2 only)

```sql
CREATE TABLE face_fingerprints (
    id TEXT PRIMARY KEY,
    vector_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    distance_metric TEXT NOT NULL,
    normalization TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL,
    extracted_at TEXT NOT NULL,
    confidence REAL,
    quality_score REAL,
    provenance_json TEXT NOT NULL
);

CREATE TABLE local_face_clusters (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    track_ids_json TEXT NOT NULL,
    faceprint_id TEXT NOT NULL REFERENCES face_fingerprints(id),
    source_embedding_ids_json TEXT NOT NULL,
    pooling_method TEXT NOT NULL,
    quality_score REAL NOT NULL,
    clustering_algorithm TEXT NOT NULL,
    clustering_threshold REAL NOT NULL,
    clustering_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE local_speaker_clusters (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    segment_ids_json TEXT NOT NULL,
    voiceprint_id TEXT NOT NULL,
    source_embedding_ids_json TEXT NOT NULL,
    pooling_method TEXT NOT NULL,
    quality_score REAL NOT NULL,
    total_speech_duration_ms INTEGER NOT NULL,
    num_segments INTEGER NOT NULL,
    clustering_algorithm TEXT NOT NULL,
    clustering_threshold REAL NOT NULL,
    clustering_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

## 7. Verification

1. cargo build succeeds
2. cargo test passes (existing tests unchanged)
3. New types compile and construct
4. SQLite projection creates 3 new tables
5. Existing VoiceFingerprint/FaceDetection tests pass
