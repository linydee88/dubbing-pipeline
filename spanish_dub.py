#!/usr/bin/env python3
"""Open-source video dubbing pipeline: English -> Spanish in your own voice.

Created by Liny Dee for creators who want to reach Spanish-speaking audiences
without paying per-minute AI dubbing subscriptions.

Pipeline:
1. Transcribes English audio with faster-whisper
2. Translates each segment to Spanish with Helsinki-NLP opus-mt-en-es
3. Clones your voice with VoxCPM2 and synthesizes Spanish audio in your voice
4. Outputs dubbed Spanish MP3

No lip sync needed if your video doesn't show your face (e.g. hands-only tarot readings,
cooking videos, screen recordings). Just swap the audio track in your editor.

Usage:
    python spanish_dub.py AUDIO.mp3 [AUDIO2.mp3 ...]

Requirements: see requirements.txt. A GPU is strongly recommended for the
voice cloning step (VoxCPM2). This runs on RunPod, a cloud GPU service.

License: MIT - free for personal and commercial use.
"""

import subprocess
import sys
from pathlib import Path

WHISPER_MODEL = "base.en"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-es"
VOXCPM_MODEL = "openbmb/VoxCPM2"

# Work directory for temp files (relative to script location)
WORK_DIR = Path(__file__).parent / "work"


def transcribe(audio: Path):
    """Transcribe English audio to segments with faster-whisper."""
    from faster_whisper import WhisperModel

    # Convert to WAV first for compatibility
    wav_audio = WORK_DIR / f"{audio.stem}_transcribe.wav"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio), "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", str(wav_audio)],
        check=True, capture_output=True,
    )
    # Use GPU if available, fall back to CPU
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
    segments, _ = model.transcribe(str(wav_audio), language="en", vad_filter=True)
    return list(segments)


def translate_segments(segments):
    """Translate English segments to Spanish with MarianMT."""
    from transformers import MarianMTModel, MarianTokenizer

    print("Loading Spanish translation model...", flush=True)
    tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = MarianMTModel.from_pretrained(TRANSLATION_MODEL)
    results = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            results.append("")
            continue
        inputs = tokenizer(
            [text], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        translated = model.generate(**inputs, max_length=512)
        spanish = tokenizer.decode(translated[0], skip_special_tokens=True)
        results.append(spanish)
        print(f"  EN: {text[:60]}...", flush=True)
        print(f"  ES: {spanish[:60]}...", flush=True)
    return results


def dub_audio(audio: Path, segments, translations):
    """Clone voice and synthesize Spanish audio with VoxCPM2."""
    from voxcpm import VoxCPM
    import soundfile as sf
    import numpy as np

    print("Loading VoxCPM2 voice cloning model...", flush=True)
    model = VoxCPM.from_pretrained(VOXCPM_MODEL, load_denoiser=False)

    # Use first 10 seconds of original audio as voice reference
    ref_audio = WORK_DIR / f"{audio.stem}_ref.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio), "-t", "10", "-ar", "16000",
         "-ac", "1", "-c:a", "pcm_s16le", str(ref_audio)],
        check=True, capture_output=True,
    )

    # Generate Spanish audio for each segment
    all_audio = []
    sample_rate = None

    for i, (seg, spanish) in enumerate(zip(segments, translations)):
        if not spanish.strip():
            continue
        print(
            f"  Synthesizing segment {i+1}/{len(translations)}: "
            f"{spanish[:50]}...",
            flush=True,
        )
        wav = model.generate(
            text=spanish,
            reference_wav_path=str(ref_audio),
            cfg_value=2.0,
            inference_timesteps=10,
        )
        if sample_rate is None:
            # NOTE: attribute path unverified - VoxCPM2 API may differ.
            # If this fails, check the voxcpm docs for the correct
            # sample rate attribute.
            sample_rate = model.tts_model.sample_rate
        all_audio.append(wav)

    if not all_audio:
        raise ValueError("No audio generated - all translations were empty")

    combined = np.concatenate(all_audio)

    # Save dubbed audio as WAV then convert to MP3
    out_wav = audio.with_name(f"{audio.stem}_dubbed_es.wav")
    sf.write(str(out_wav), combined, sample_rate)
    print(f"Wrote dubbed audio: {out_wav}", flush=True)

    out_mp3 = audio.with_name(f"{audio.stem}_dubbed_es.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out_wav), "-codec:a", "libmp3lame",
         "-q:a", "2", str(out_mp3)],
        check=True, capture_output=True,
    )
    print(f"Wrote dubbed MP3: {out_mp3}", flush=True)

    out_wav.unlink()
    return out_mp3


def main(audio_files):
    for a in audio_files:
        audio = Path(a)
        if not audio.exists():
            print(f"SKIP (not found): {audio}", flush=True)
            continue
        print(f"Transcribing: {audio.name}", flush=True)
        segments = transcribe(audio)
        print(f"Translating {len(segments)} segments to Spanish...", flush=True)
        translations = translate_segments(segments)
        print("Dubbing to Spanish in your voice...", flush=True)
        out_mp3 = dub_audio(audio, segments, translations)
        print(f"Done: {out_mp3}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
