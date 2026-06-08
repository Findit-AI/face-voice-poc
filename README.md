# Face-Voice Binding PoC

跨模态人物身份系统 — 通过人脸嵌入和声纹嵌入的绑定，实现跨视频的人物身份识别。

## 概述

本 PoC 验证了以下技术路线的可行性：

1. **人脸检测+嵌入**: Apple Vision (检测) + AdaFace IR-101 ONNX (512D 嵌入, MIT)
2. **人脸聚类**: DBSCAN (cosine distance, eps=0.8)
3. **说话人分割**: CoreML pyannote-3.0 (绕过 Python pipeline, 12ms/10s)
4. **声纹嵌入**: WeSpeaker ResNet34-LM ONNX (256D, 195ms/chunk, 跨平台, DIA已集成)
5. **面-声绑定**: BindingEvidence 机制（证据累积，非断言）

## 性能 (Apple M1, 8GB, CPU-only)

| 管线 | 耗时 |
|------|------|
| 人脸管线 (检测+嵌入+聚类) | ~45s |
| 声纹管线 (分割+嵌入+聚类) | ~7s |
| **12 分钟视频总计** | **~60s** |

## 关键发现

- AdaFace IR-101 ONNX (512D, MIT) 可替代 InsightFace buffalo_l (非商用)：聚类质量相当 (9 vs 10 clusters)，速度相当 (113ms)
- AdaFace IR-18 CoreML 区分度不够（3 clusters vs 9），不可用于聚类
- pyannote Python pipeline 99% 是编排开销，绕过后提速 89x
- WeSpeaker ResNet34-LM ONNX (256D, MIT) 是最佳声纹嵌入：gap=0.244 (1.8x优于ECAPA-TDNN的0.138), DIA已集成Rust ONNX
- ECAPA-TDNN ONNX (192D) 作为benchmark参考：51ms/chunk, gap=0.138
- Resemblyzer GE2E (256D) 在短片段上区分度不足（gap=0.06）
- OpenVoice Speaker Encoder 不能区分说话人（设计用于语音转换）

## 文档

- [FINAL_DESIGN.md](FINAL_DESIGN.md) — 完整技术设计（含所有选型来源和 benchmark 数据）
- [DESIGN_REVIEW.md](DESIGN_REVIEW.md) — Codex 审查的原始设计
- [CODEX_REVIEW_RESPONSE.md](CODEX_REVIEW_RESPONSE.md) — Codex Q&A

## 依赖

```bash
pip install insightface opencv-python scenedetect pyannote.audio soundfile librosa sklearn
pip install pyobjc-framework-Vision  # Apple Vision (macOS only)
pip install onnxruntime              # ECAPA-TDNN ONNX inference
```

CoreML 模型（已包含在 `models/coreml/`）：
- `SpeakerSegmentation.mlpackage` — 5.8MB, MIT
- `OpenVoice_SpeakerEncoder.mlpackage` — 1.7MB, MIT (已验证不可用于说话人区分)

## 使用

```bash
# 完整管线
python3 src/step1_shot_detection.py <video.mp4> output/shots.json
python3 src/step23_detect_track_embed.py <video.mp4> output/shots.json output/face_tracks.json
python3 src/step4_face_cluster.py output/face_tracks.json output/face_clusters.json
python3 src/step56_speaker_diarize.py <video.mp4> output/speaker_segments.json
python3 src/step789_candidate_evidence.py output/face_clusters.json output/speaker_segments.json output/face_tracks.json output/candidates.json output/binding_evidence.json
python3 src/step10_report.py

# 可视化
python3 src/visualize.py
```

## License

PoC code: MIT
CoreML models: MIT (SpeakerSegmentation), MIT (OpenVoice)
WeSpeaker ResNet34-LM ONNX: MIT
AdaFace IR-101 ONNX: MIT
All components are MIT licensed for production use.
