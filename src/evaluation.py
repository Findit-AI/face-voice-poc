"""Evaluation metrics for face-voice binding pipeline.

Computes pairwise precision/recall/F1 for clustering quality,
and binding accuracy metrics.
"""
import json
import numpy as np
from pathlib import Path


def pairwise_metrics(true_labels: list, pred_labels: list) -> dict:
    """Compute pairwise precision, recall, F1 for clustering.
    
    Args:
        true_labels: ground truth cluster assignments
        pred_labels: predicted cluster assignments
    
    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    n = len(true_labels)
    if n < 2:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_true = true_labels[i] == true_labels[j]
            same_pred = pred_labels[i] == pred_labels[j]
            
            if same_true and same_pred:
                tp += 1
            elif not same_true and same_pred:
                fp += 1
            elif same_true and not same_pred:
                fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def bcubed_metrics(true_labels: list, pred_labels: list) -> dict:
    """Compute B-cubed precision, recall, F1.
    
    Per-item precision = fraction of predicted cluster that matches true class.
    Per-item recall = fraction of true class that is in predicted cluster.
    """
    n = len(true_labels)
    if n < 2:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    precisions = []
    recalls = []
    
    for i in range(n):
        # Items in same predicted cluster
        pred_cluster = [j for j in range(n) if pred_labels[j] == pred_labels[i]]
        # Items in same true class
        true_class = [j for j in range(n) if true_labels[j] == true_labels[i]]
        
        # B-cubed precision: how many in pred_cluster are in true_class
        p = sum(1 for j in pred_cluster if true_labels[j] == true_labels[i]) / len(pred_cluster)
        # B-cubed recall: how many in true_class are in pred_cluster
        r = sum(1 for j in true_class if pred_labels[j] == pred_labels[i]) / len(true_class)
        
        precisions.append(p)
        recalls.append(r)
    
    avg_p = np.mean(precisions)
    avg_r = np.mean(recalls)
    avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0
    
    return {
        "precision": round(float(avg_p), 4),
        "recall": round(float(avg_r), 4),
        "f1": round(float(avg_f1), 4),
    }


def evaluate_face_clustering(clusters_path: str, ground_truth_path: str = None) -> dict:
    """Evaluate face clustering quality.
    
    If ground_truth_path is provided, computes metrics against it.
    Otherwise, reports internal statistics.
    """
    with open(clusters_path) as f:
        clusters = json.load(f)
    
    # Internal statistics
    sizes = [len(c.get("track_ids", [])) for c in clusters]
    result = {
        "num_clusters": len(clusters),
        "sizes": sizes,
        "mean_size": round(np.mean(sizes), 1) if sizes else 0,
        "std_size": round(np.std(sizes), 1) if sizes else 0,
    }
    
    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path) as f:
            gt = json.load(f)
        
        # Build label lists
        true_labels = []
        pred_labels = []
        for item in gt:
            true_labels.append(item["true_identity"])
            # Find which cluster this track belongs to
            track_id = item["track_id"]
            for ci, c in enumerate(clusters):
                if track_id in c.get("track_ids", []):
                    pred_labels.append(ci)
                    break
            else:
                pred_labels.append(-1)  # not clustered
        
        result["pairwise"] = pairwise_metrics(true_labels, pred_labels)
        result["bcubed"] = bcubed_metrics(true_labels, pred_labels)
    
    return result


def evaluate_binding(candidates_path: str, ground_truth_path: str = None) -> dict:
    """Evaluate binding evidence quality."""
    with open(candidates_path) as f:
        candidates = json.load(f)
    
    result = {
        "num_candidates": len(candidates),
        "single_face": sum(1 for c in candidates if c.get("method") == "single_face"),
        "multi_face": sum(1 for c in candidates if c.get("method") == "multi_face_needs_asd"),
    }
    
    if ground_truth_path and Path(ground_truth_path).exists():
        with open(ground_truth_path) as f:
            gt = json.load(f)
        
        # Match candidates to ground truth
        tp = fp = fn = 0
        for g in gt:
            found = False
            for c in candidates:
                speaker_id = c.get("speaker_cluster_id", c.get("speaker_id", ""))
                if (c.get("face_cluster_id") == g["face_cluster_id"] and
                    speaker_id == g["speaker_cluster_id"]):
                    found = True
                    break
            if g["is_binding"]:
                if found:
                    tp += 1
                else:
                    fn += 1
            else:
                if found:
                    fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        result["binding_precision"] = round(precision, 4)
        result["binding_recall"] = round(recall, 4)
        result["binding_f1"] = round(f1, 4)
    
    return result


if __name__ == "__main__":
    # Evaluate current pipeline output
    print("=== Face Clustering ===")
    result = evaluate_face_clustering("output/face_clusters.json")
    print(json.dumps(result, indent=2))
    
    print("\n=== Binding Candidates ===")
    result = evaluate_binding("output/candidates.json")
    print(json.dumps(result, indent=2))
