# Implementation Plan: P3 — Segment-Level Embedding Storage (v2)

## Goal

Store every individual face/speaker embedding (not just centroids) in LanceDB.

## Confirmed Models

| Modality | Model | Dimension | License |
|----------|-------|-----------|---------|
| Face | AdaFace IR-101 ONNX | 512D | MIT |
| Voice | WeSpeaker ResNet34-LM ONNX | 256D | MIT |

## Codex Findings Addressed

### Fix 1: Extraction granularity

Current code stores per-track centroid only. P3 must:
- Save per-frame face embedding (not just centroid)
- Save per-segment voice embedding (not just centroid)
- Store both individual embeddings AND centroid

### Fix 2: local_cluster_id lifecycle

Embeddings are written before clustering. Solution:
- `local_cluster_id` is NULL at write time
- Updated after clustering completes
- LanceDB row is updated in-place (upsert on vector_id)

### Fix 3: Dual-write consistency

- Use vector_id as idempotent upsert key
- Write to LanceDB first, then mediaschema
- Reconciliation: verify every FaceFingerprint.vector_id exists in LanceDB

### Fix 4: Stable source identifiers

Face: (video_id, track_id, frame_number)
Voice: (video_id, segment_id, start_ms, end_ms)

## LanceDB Table Schema

### face_embeddings

```python
schema = pa.schema([
    pa.field("vector_id", pa.binary(16)),      # UUID bytes
    pa.field("vector", pa.list_(pa.float32(), 512)),
    pa.field("video_id", pa.binary(16)),
    pa.field("track_id", pa.binary(16)),
    pa.field("frame_number", pa.int64()),
    pa.field("local_cluster_id", pa.binary(16)),  # NULL before clustering
    pa.field("model_id", pa.utf8()),
    pa.field("model_hash", pa.utf8()),
    pa.field("preprocessing_version", pa.utf8()),
    pa.field("distance_metric", pa.utf8()),       # "cosine"
    pa.field("normalization", pa.utf8()),          # "l2"
    pa.field("quality_score", pa.float32()),
    pa.field("bbox_size_px", pa.int32()),
    pa.field("vision_confidence", pa.float32()),
    pa.field("created_at", pa.timestamp("ms")),
])
```

### voice_embeddings

```python
schema = pa.schema([
    pa.field("vector_id", pa.binary(16)),
    pa.field("vector", pa.list_(pa.float32(), 256)),
    pa.field("video_id", pa.binary(16)),
    pa.field("segment_id", pa.binary(16)),
    pa.field("start_ms", pa.int64()),
    pa.field("end_ms", pa.int64()),
    pa.field("local_cluster_id", pa.binary(16)),  # NULL before clustering
    pa.field("model_id", pa.utf8()),
    pa.field("model_hash", pa.utf8()),
    pa.field("preprocessing_version", pa.utf8()),
    pa.field("distance_metric", pa.utf8()),
    pa.field("normalization", pa.utf8()),
    pa.field("quality_score", pa.float32()),
    pa.field("duration_ms", pa.int64()),
    pa.field("speech_confidence", pa.float32()),
    pa.field("created_at", pa.timestamp("ms")),
])
```

## Implementation Order

1. Install lancedb Python package
2. Create face_embeddings table + writer function
3. Modify step23 to save per-frame embeddings (not just centroid)
4. Create voice_embeddings table + writer function
5. Create embedding reader + centroid computation
6. Verify: re-cluster from stored embeddings matches original
7. Update local_cluster_id after clustering

## Verification

1. LanceDB tables created with correct schema
2. Per-frame face embeddings written and retrievable
3. Per-segment voice embeddings written and retrievable
4. Centroid from stored embeddings matches direct computation
5. Re-clustering produces same clusters
6. local_cluster_id updated correctly after clustering
