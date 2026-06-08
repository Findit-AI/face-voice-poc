"""Step 1: Shot Detection using PySceneDetect."""
import json
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector


def detect_shots(video_path: str, output_path: str) -> list[dict]:
    """Detect shot boundaries in a video.
    
    Returns list of shots: [{start_frame, start_time, end_frame, end_time}]
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    fps = video.frame_rate
    shots = []
    for i, (start, end) in enumerate(scene_list):
        shots.append({
            "shot_id": i,
            "start_frame": start.get_frames(),
            "start_time": round(start.get_seconds(), 3),
            "end_frame": end.get_frames(),
            "end_time": round(end.get_seconds(), 3),
        })
    
    # If no scene changes detected, treat entire video as one shot
    if not shots:
        total_frames = video.duration.get_frames()
        shots.append({
            "shot_id": 0,
            "start_frame": 0,
            "start_time": 0.0,
            "end_frame": total_frames,
            "end_time": round(video.duration.get_seconds(), 3),
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(shots, f, indent=2)
    
    print(f"Detected {len(shots)} shot(s):")
    for s in shots:
        print(f"  Shot {s['shot_id']}: {s['start_time']:.1f}s - {s['end_time']:.1f}s")
    
    return shots


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "Desktop/Kino Demo Render.mp4")
    output = sys.argv[2] if len(sys.argv) > 2 else "output/shots.json"
    detect_shots(video, output)
