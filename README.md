# Open-Source Video Dubbing Pipeline

Dub your videos into Spanish in **your own voice**. This runs on RunPod, a cloud GPU service. The only cost is the GPU time while it processes your video. No subscriptions, no per-minute fees, no caps.

Created by [Liny Dee](https://www.stan.store/linydee) for creators who want to reach Spanish-speaking audiences without paying for expensive AI dubbing services.

> **Status: in validation.** Steps 1 and 2 are proven on real audio. Step 3 (voice cloning) has not yet completed end to end in this setup. See "What is verified" below.

## How it works

1. **Transcribe** your English audio with faster-whisper ✅ verified
2. **Translate** each segment to Spanish with Helsinki-NLP (MarianMT) ✅ verified
3. **Clone your voice** with VoxCPM2 and synthesize the Spanish audio in your voice ⏳ in validation
4. You get a dubbed Spanish MP3, ready to swap onto your video in any editor

No lip sync needed if your video doesn't show your face (hands-only tutorials, cooking videos, screen recordings, tarot readings, etc.). Just replace the audio track.

## What is verified

- **Transcription:** ran on an 18-minute real-world audio file, produced 228 segments correctly.
- **Translation:** all 228 segments translated English to Spanish with Helsinki-NLP opus-mt-en-es. Output inspected and correct.
- **Script logic:** transcribe, translate, and dub code paths reviewed and correct.
- **RunPod workflow:** pod creation, PyTorch template, RTX 4090 selection, Jupyter terminal access, and file upload all confirmed working.

## What is still in validation

- **Voice cloning end to end:** VoxCPM2 (39k stars, Apache 2.0) is a real, maintained project, but the full dubbing step has not yet completed in this pipeline. It was killed for memory on a CPU-only machine. The next test is on a RunPod GPU.
- **Fresh pip install on RunPod:** dependencies installed cleanly on the dev machine, but a fresh RunPod PyTorch image may have version conflicts.
- **Time and cost:** "15-20 minutes, about $0.25-$0.35 per 18-minute video" is an estimate from VoxCPM2's published inference speed and measured transcription time, not a measured run.

## Quick start on RunPod

1. Create a RunPod account at [runpod.io](https://www.runpod.io/)
2. Deploy a GPU pod: **Pods** > choose a **PyTorch** template > pick an **RTX 4090** (about $0.89/hr)
3. Open **Jupyter Notebook** on the pod, then open a **Terminal**
4. Clone this repo and install:
   ```bash
   git clone https://github.com/linydee88/dubbing-pipeline.git
   cd dubbing-pipeline
   pip install -r requirements.txt
   ```
   _Note: model downloads need network access. If downloads fail, run with `env -u no_proxy -u NO_PROXY` prefix._
5. Upload your audio file (MP3 or M4A) to the pod via the Jupyter file browser
6. Run the dubbing:
   ```bash
   python spanish_dub.py your-audio.m4a
   ```
7. Download the `your-audio_dubbed_es.mp3` file when it's done
8. Drop it onto your video in CapCut, Premiere, or any editor

An 18-minute video is estimated at roughly 15-20 minutes on an RTX 4090, costing about $0.25-$0.35. (Estimate, not yet measured. See "What is still in validation.")

**Remember to stop your pod when you're done** so you're not charged for idle time.

## Requirements

- Python 3.10+
- PyTorch (CPU works for transcription/translation, GPU strongly recommended for voice cloning)
- ffmpeg (usually pre-installed on RunPod PyTorch templates)
- See `requirements.txt` for Python packages

## Models used

| Step | Model | License |
|------|-------|---------|
| Transcription | faster-whisper (base.en) | MIT |
| Translation | Helsinki-NLP opus-mt-en-es | Apache 2.0 |
| Voice cloning | OpenBMB VoxCPM2 | Apache 2.0 |

All models are free for commercial use.

## Tips

- **Audio only is enough.** You don't need to upload the full video. Extract the audio (MP3/M4A) on your phone, dub it, then mux it back onto the video in your editor.
- **Voice reference:** the script uses the first 10 seconds of your audio as the voice cloning reference. Make sure those 10 seconds are clean speech in your natural voice.
- **Translation quality:** machine translation is good but not perfect. Spot-check the Spanish output for your niche vocabulary (tarot terms, slang, etc.) before publishing.
- **Other languages:** VoxCPM2 supports 30 languages. To dub to a different language, swap the translation model (e.g. `Helsinki-NLP/opus-mt-en-fr` for French) and adjust the script.

## License

MIT - free for personal and commercial use. See LICENSE file.
