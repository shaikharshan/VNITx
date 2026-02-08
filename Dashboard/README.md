---
title: VNITx Multimodal Security Dashboard
emoji: 🛰️
colorFrom: blue
colorTo: purple
sdk: streamlit
pinned: false
license: mit
short_description: HackIITK 2026 demo console for VNITx APIs
---

# VNITx Multimodal Security Dashboard

This Streamlit dashboard connects to the deployed VNITx Hugging Face Spaces:

- Audio: `POST /api/voice-detection`
- Image: `POST /analyze`
- Video: `POST /analyze_video`

## Environment Variables

- `AUDIO_BASE` (default: `https://arshan123-vnitx-audio.hf.space`)
- `IMAGE_BASE` (default: `https://arshan123-vnitx-image.hf.space`)
- `VIDEO_BASE` (default: `https://arshan123-vnitx-video.hf.space`)
- `AUDIO_API_KEY` (default: `sk_test_123456789`)

## Notes

- `packages.txt` installs `ffmpeg` for audio extraction and conversion.
- `requirements.txt` includes `pydub`, `streamlit`, and `httpx`.
