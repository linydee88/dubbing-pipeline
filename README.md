# Open-Source Video Dubbing Pipeline

Dub your videos into Spanish, Tagalog, Portuguese, Hindi, or French in **your own voice**. This runs on RunPod, a cloud GPU service. The only cost is the GPU time while it processes your video. No subscriptions, no per-minute fees, no caps.

Created by [Liny Dee](https://www.stan.store/linydee) for creators who want to reach new audiences without paying for expensive AI dubbing services.

> **Status: verified.** All three steps completed end to end on real audio, October 14, 2026.

## How it works

1. **Transcribe** your English audio with faster-whisper ✅ verified
2. **Translate** each segment with Helsinki-NLP (MarianMT) ✅ verified
3. **Clone your voice** with VoxCPM2 and synthesize the dubbed audio in your voice ✅ verified
4. You get a dubbed MP3, ready to swap onto your video in any editor

No lip sync needed if your video doesn't show your face (hands-only tutorials, cooking videos, screen recordings, tarot readings, etc.). Just replace the audio track.

## Supported languages

| Code | Language | Translation model |
|------|----------|-------------------|
| `es` | Spanish | Helsinki-NLP/opus-mt-en-es ✅ verified |
| `tl` | Tagalog | Helsinki-NLP/opus-mt-en-tl |
| `pt` | Portuguese | Helsinki-NLP/opus-mt-tc-big-en-pt |
| `hi` | Hindi | Helsinki-NLP/opus-mt-en-hi |
| `fr` | French | Helsinki-NLP/opus-mt-en-fr |

## What is verified

- **Transcription:** ran on an 18-minute real-world audio file, produced 229 segments correctly.
- **Translation:** all 229 segments translated English to Spanish with Helsinki-NLP opus-mt-en-es. Output inspected and correct.
- **Voice cloning:** VoxCPM2 completed all 229 segments end to end on a RunPod L4 GPU, October 14, 2026. Output sounds like the speaker's own voice in Spanish.
- **Timing:** segments are concatenated in their natural flow, then the full dubbed audio is time-stretched with ffmpeg atempo to match the original audio length (slowed down if shorter, sped up if longer).
- **Proper-noun corrections:** the `CORRECTIONS` dictionary at the top of `dub.py` fixes known mistranslations per language (e.g. Spanish: "Arthur Edward peso" back to "Arthur Edward Waite") before voice synthesis. Add your own entries as you find them.
- **Script logic:** transcribe, translate, and dub code paths all run clean.
- **RunPod workflow:** pod creation, PyTorch template, GPU selection, Jupyter terminal access, and file upload all confirmed working.
- **GPU compatibility:** confirmed working on NVIDIA L4 (24GB VRAM). Any CUDA-capable NVIDIA GPU should work.

## Known issues

- **av library incompatibility:** `faster-whisper` breaks with `av` 19.x (`metadata_errors` error). Pinned to `av==12.3.0` in requirements.txt.
- **Proper nouns:** machine translation may mistranslate names (e.g. "Waite" became "peso"). Spot-check and fix niche vocabulary before publishing.

## Quick start on RunPod

1. Create a RunPod account at [runpod.io](https://www.runpod.io/)
2. Deploy a GPU pod: **Pods** > choose a **PyTorch** template > pick any available **NVIDIA GPU** (RTX 4090, 3090, A5000, etc. - any CUDA card works)
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
   python dub.py es your-audio.m4a    # Spanish
   python dub.py tl your-audio.m4a    # Tagalog
   python dub.py pt your-audio.m4a    # Portuguese
   python dub.py hi your-audio.m4a    # Hindi
   python dub.py fr your-audio.m4a    # French
   ```
7. Download the `your-audio_dubbed_es.mp3` (or `_dubbed_tl.mp3`) file when it's done
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
| Translation | Helsinki-NLP opus-mt-en-es / opus-mt-en-tl | Apache 2.0 |
| Voice cloning | OpenBMB VoxCPM2 | Apache 2.0 |

All models are free for commercial use.

## Tips

- **Audio only is enough.** You don't need to upload the full video. Extract the audio (MP3/M4A) on your phone, dub it, then mux it back onto the video in your editor.
- **Dependency note:** `faster-whisper` breaks with newer `av` versions (19.x). The `requirements.txt` pins `av==12.3.0` to avoid the `metadata_errors` error. If you install packages manually, use that version.
- **Voice reference:** the script uses the first 10 seconds of your audio as the voice cloning reference. Make sure those 10 seconds are clean speech in your natural voice.
- **Translation quality:** machine translation is good but not perfect. Spot-check the output for your niche vocabulary (tarot terms, slang, etc.) before publishing.
- **Other languages:** VoxCPM2 supports 30 languages. To add another language, add its Helsinki-NLP model to the `LANGUAGES` dict in `dub.py`.

## License

MIT - free for personal and commercial use. See LICENSE file.
