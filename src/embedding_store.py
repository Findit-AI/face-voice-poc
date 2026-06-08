from typing import Optional
"""LanceDB embedding store for face and voice embeddings.

Stores individual per-frame/per-segment embeddings (not just centroids)
to enable re-clustering, rollback, pooling changes, and model migration.
"""
import uuid
import time
import numpy as np
from pathlib import Path

import lancedb
import pyarrow as pa


# Model constants
# Currently using InsightFace buffalo_l for PoC; production will use AdaFace IR-101
FACE_MODEL_ID = "insightface_buffalo_l"
FACE_MODEL_HASH = ""  # TODO: compute sha256 of ONNX file
FACE_DIM = 512
FACE_PREPROCESSING = "face_aligned_112_v1"

VOICE_MODEL_ID = "wespeaker_resnet34"
VOICE_MODEL_HASH = ""  # TODO: compute sha256 of ONNX file
VOICE_DIM = 256
VOICE_PREPROCESSING = "audio_16k_mono_v1"


def get_db(db_path: str = "output/lancedb") -> lancedb.DBConnection:
    """Open or create LanceDB database."""
    Path(db_path).mkdir(parents=True, exist_ok=True)
    return lancedb.connect(db_path)


def uuid7_bytes() -> bytes:
    """Generate UUID7 as 16 bytes."""
    return uuid.uuid4().bytes  # TODO: use proper UUID7


def create_face_embeddings_table(db: lancedb.DBConnection):
    """Create face_embeddings table if not exists."""
    schema = pa.schema([
        pa.field("vector_id", pa.binary(16)),
        pa.field("vector", pa.list_(pa.float32(), FACE_DIM)),
        pa.field("video_id", pa.binary(16)),
        pa.field("track_id", pa.binary(16)),
        pa.field("frame_number", pa.int64()),
        pa.field("local_cluster_id", pa.binary(16)),
        pa.field("model_id", pa.utf8()),
        pa.field("model_hash", pa.utf8()),
        pa.field("preprocessing_version", pa.utf8()),
        pa.field("distance_metric", pa.utf8()),
        pa.field("normalization", pa.utf8()),
        pa.field("quality_score", pa.float32()),
        pa.field("bbox_size_px", pa.int32()),
        pa.field("vision_confidence", pa.float32()),
        pa.field("created_at", pa.timestamp("ms")),
    ])
    try:
        db.create_table("face_embeddings", schema=schema)
        print("Created face_embeddings table")
    except Exception:
        print("face_embeddings table already exists")


def create_voice_embeddings_table(db: lancedb.DBConnection):
    """Create voice_embeddings table if not exists."""
    schema = pa.schema([
        pa.field("vector_id", pa.binary(16)),
        pa.field("vector", pa.list_(pa.float32(), VOICE_DIM)),
        pa.field("video_id", pa.binary(16)),
        pa.field("segment_id", pa.binary(16)),
        pa.field("start_ms", pa.int64()),
        pa.field("end_ms", pa.int64()),
        pa.field("local_cluster_id", pa.binary(16)),
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
    try:
        db.create_table("voice_embeddings", schema=schema)
        print("Created voice_embeddings table")
    except Exception:
        print("voice_embeddings table already exists")


def write_face_embeddings(
    db: lancedb.DBConnection,
    video_id: bytes,
    embeddings: list,
) -> list:
    """Write per-frame face embeddings to LanceDB.
    
    Args:
        db: LanceDB connection
        video_id: 16-byte UUID
        embeddings: list of {
            "track_id": bytes,
            "frame_number": int,
            "vector": list[float],  # 512D
            "bbox_size_px": int,
            "vision_confidence": float,
            "quality_score": float,
        }
    
    Returns:
        list of vector_id bytes (for linking to FaceFingerprint)
    """
    table = db.open_table("face_embeddings")
    now_ms = int(time.time() * 1000)
    
    rows = []
    vector_ids = []
    for emb in embeddings:
        vid = uuid7_bytes()
        vector_ids.append(vid)
        rows.append({
            "vector_id": vid,
            "vector": emb["vector"],
            "video_id": video_id,
            "track_id": emb["track_id"],
            "frame_number": emb["frame_number"],
            "local_cluster_id": None,  # set after clustering
            "model_id": FACE_MODEL_ID,
            "model_hash": FACE_MODEL_HASH,
            "preprocessing_version": FACE_PREPROCESSING,
            "distance_metric": "cosine",
            "normalization": "l2",
            "quality_score": emb.get("quality_score", 0.0),
            "bbox_size_px": emb.get("bbox_size_px", 0),
            "vision_confidence": emb.get("vision_confidence", 0.0),
            "created_at": now_ms,
        })
    
    if rows:
        table.add(rows)
        print(f"Wrote {len(rows)} face embeddings to LanceDB")
    
    return vector_ids


def write_voice_embeddings(
    db: lancedb.DBConnection,
    video_id: bytes,
    embeddings: list,
) -> list:
    """Write per-segment voice embeddings to LanceDB.
    
    Args:
        db: LanceDB connection
        video_id: 16-byte UUID
        embeddings: list of {
            "segment_id": bytes,
            "start_ms": int,
            "end_ms": int,
            "vector": list[float],  # 256D
            "duration_ms": int,
            "speech_confidence": float,
            "quality_score": float,
        }
    
    Returns:
        list of vector_id bytes
    """
    table = db.open_table("voice_embeddings")
    now_ms = int(time.time() * 1000)
    
    rows = []
    vector_ids = []
    for emb in embeddings:
        vid = uuid7_bytes()
        vector_ids.append(vid)
        rows.append({
            "vector_id": vid,
            "vector": emb["vector"],
            "video_id": video_id,
            "segment_id": emb["segment_id"],
            "start_ms": emb["start_ms"],
            "end_ms": emb["end_ms"],
            "local_cluster_id": None,
            "model_id": VOICE_MODEL_ID,
            "model_hash": VOICE_MODEL_HASH,
            "preprocessing_version": VOICE_PREPROCESSING,
            "distance_metric": "cosine",
            "normalization": "l2",
            "quality_score": emb.get("quality_score", 0.0),
            "duration_ms": emb.get("duration_ms", 0),
            "speech_confidence": emb.get("speech_confidence", 0.0),
            "created_at": now_ms,
        })
    
    if rows:
        table.add(rows)
        print(f"Wrote {len(rows)} voice embeddings to LanceDB")
    
    return vector_ids


def update_cluster_ids(
    db: lancedb.DBConnection,
    table_name: str,
    vector_ids: list,
    cluster_id: bytes,
):
    """Update local_cluster_id for embeddings after clustering.
    
    This is a no-op for now — LanceDB doesn't support in-place updates.
    Cluster membership is tracked via mediaschema's source_embedding_ids.
    """
    pass  # TODO: implement when LanceDB supports updates


def compute_centroid(
    db: lancedb.DBConnection,
    table_name: str,
    vector_ids: list,
    weights: Optional[list] = None,
) -> np.ndarray:
    """Compute quality-weighted centroid from stored embeddings.
    
    Args:
        db: LanceDB connection
        table_name: "face_embeddings" or "voice_embeddings"
        vector_ids: list of vector_id bytes to include
        weights: optional per-embedding weights (default: equal)
    
    Returns:
        L2-normalized centroid vector
    """
    table = db.open_table(table_name)
    
    # Query embeddings by vector_id
    # LanceDB doesn't support IN queries directly, so we scan and filter
    df = table.to_pandas()
    id_set = set(vid for vid in vector_ids)
    mask = df["vector_id"].isin(id_set)
    matched = df[mask]
    
    if matched.empty:
        raise ValueError(f"No embeddings found for {len(vector_ids)} vector_ids")
    
    vectors = np.array(matched["vector"].tolist())
    
    if weights is not None:
        weights = np.array(weights[:len(vectors)])
        centroid = np.average(vectors, axis=0, weights=weights)
    else:
        centroid = np.mean(vectors, axis=0)
    
    # L2 normalize
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    
    return centroid


if __name__ == "__main__":
    db = get_db()
    create_face_embeddings_table(db)
    create_voice_embeddings_table(db)
    print("LanceDB tables ready at output/lancedb/")
