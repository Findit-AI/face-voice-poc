"""Step 10: Generate HTML report for manual review."""
import json
from pathlib import Path


def generate_report(face_clusters_path: str, speaker_segments_path: str,
                    evidence_path: str, face_tracks_path: str, output_path: str):
    """Generate HTML report for human review of binding evidence."""
    with open(face_clusters_path) as f:
        clusters = json.load(f)
    with open(speaker_segments_path) as f:
        segments = json.load(f)
    with open(evidence_path) as f:
        evidence = json.load(f)
    with open(face_tracks_path) as f:
        tracks = json.load(f)
    
    # Build track → cluster mapping
    track_to_cluster = {}
    for c in clusters:
        for tid in c["track_ids"]:
            track_to_cluster[tid] = c["cluster_id"]
    
    # Generate cluster summary
    cluster_rows = ""
    for c in clusters:
        cluster_rows += f"""<tr>
<td>F{c['cluster_id']}</td>
<td>{c['track_count']}</td>
<td>{c['total_faces']}</td>
<td>{c['avg_face_size']:.0f}px</td>
<td>{c['time_range'][0]:.1f}s - {c['time_range'][1]:.1f}s</td>
</tr>\n"""
    
    # Generate speaker summary
    speakers = {}
    for seg in segments:
        sid = seg["speaker_id"]
        if sid not in speakers:
            speakers[sid] = {"count": 0, "total_ms": 0}
        speakers[sid]["count"] += 1
        speakers[sid]["total_ms"] += seg["duration_ms"]
    
    speaker_rows = ""
    for sid, info in sorted(speakers.items()):
        speaker_rows += f"""<tr>
<td>{sid}</td>
<td>{info['count']}</td>
<td>{info['total_ms']/1000:.1f}s</td>
</tr>\n"""
    
    # Generate evidence rows
    evidence_rows = ""
    for ev in evidence:
        score_str = f"{ev['raw_score']:.2f}" if ev['raw_score'] is not None else "—"
        margin_str = f"{ev['score_margin']:.2f}" if ev['score_margin'] is not None else "—"
        
        evidence_rows += f"""<tr>
<td>{ev['evidence_id']}</td>
<td>F{ev['face_cluster_id']}</td>
<td>{ev['speaker_cluster_id']}</td>
<td>{ev['segment'][0]:.1f}s-{ev['segment'][1]:.1f}s</td>
<td>{ev['duration_ms']/1000:.1f}s</td>
<td>{ev['visible_face_fraction']:.0%}</td>
<td>{ev['face_size']:.0f}px</td>
<td>{score_str}</td>
<td>{margin_str}</td>
<td>{ev['method']}</td>
<td class="status-{ev['status']}">{ev['status']}</td>
</tr>\n"""
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Face-Voice Binding PoC — Manual Review</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0f;color:#e0e0e0;font-family:-apple-system,sans-serif;padding:20px}}
h1{{font-size:20px;color:#f0c060;margin-bottom:6px}}
h2{{font-size:16px;color:#f0a060;margin:24px 0 10px}}
table{{width:100%;border-collapse:collapse;margin:8px 0;font-size:12px}}
th{{background:#151520;color:#f0a060;padding:6px 8px;text-align:left}}
td{{padding:5px 8px;border-bottom:1px solid #1a1a2a}}
.status-candidate{{color:#fa0}}
.status-accepted{{color:#4f4}}
.status-rejected{{color:#f44}}
.info{{background:#0a1a2a;border-left:3px solid #2af;padding:10px 14px;margin:10px 0;font-size:12px;border-radius:0 6px 6px 0}}
.warn{{background:#2a1a0a;border-left:3px solid #f80;padding:10px 14px;margin:10px 0;font-size:12px;border-radius:0 6px 6px 0}}
</style>
</head>
<body>
<h1>🔬 Face-Voice Binding PoC — Manual Review</h1>
<p style="color:#888;font-size:12px">Kino Demo Render.mp4 · All bindings are candidates — review and mark accept/reject</p>

<h2>📊 Face Clusters</h2>
<table>
<tr><th>Cluster</th><th>Tracks</th><th>Faces</th><th>Avg Size</th><th>Time Range</th></tr>
{cluster_rows}
</table>

<h2>🔊 Speaker Segments</h2>
<table>
<tr><th>Speaker</th><th>Segments</th><th>Total Duration</th></tr>
{speaker_rows}
</table>

<h2>🔗 Binding Evidence</h2>
<div class="info">
Instructions: Review each binding candidate. Mark as <b>accepted</b> (face F is the speaker) or <b>rejected</b> (face F is NOT the speaker).
<br>Focus on: Does the face appear to be speaking during this segment? Is the face clearly the speaker or just a listener?
</div>
<table>
<tr><th>ID</th><th>Face</th><th>Speaker</th><th>Segment</th><th>Duration</th><th>Visible%</th><th>Face Size</th><th>ASD Score</th><th>Margin</th><th>Method</th><th>Status</th></tr>
{evidence_rows}
</table>

<h2>📝 Review Notes</h2>
<div class="warn">
TODO: After review, update binding_evidence.json with accepted/rejected status.
<br>Gate: zero false accepts, >50% coverage on clear speech.
</div>

</body>
</html>"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)
    print(f"Report: {output_path}")


if __name__ == "__main__":
    generate_report(
        "output/face_clusters.json",
        "output/speaker_segments.json",
        "output/binding_evidence.json",
        "output/face_tracks.json",
        "output/review_report.html",
    )
