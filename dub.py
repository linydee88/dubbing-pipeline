#!/usr/bin/env python3
"""Open-source video dubbing pipeline: English -> your language, in your own voice.

Created by Liny Dee for creators who want to reach new audiences
without paying per-minute AI dubbing subscriptions.

Pipeline:
1. Transcribes English audio with faster-whisper
2. Translates each segment with Helsinki-NLP OPUS-MT
3. Clones your voice with VoxCPM2 and synthesizes dubbed audio in your voice
4. Outputs dubbed MP3

No lip sync needed if your video doesn't show your face (e.g. hands-only tarot readings,
cooking videos, screen recordings). Just swap the audio track in your editor.

Usage:
    python dub.py LANG AUDIO.mp3 [AUDIO2.mp3 ...]

    LANG is a language code, e.g.:
        es = Spanish
        tl = Tagalog (Filipino)

Examples:
    python dub.py es cancer-eng-audio.m4a
    python dub.py tl cancer-eng-audio.m4a

Requirements: see requirements.txt. A GPU is strongly recommended for the
voice cloning step (VoxCPM2). This runs on RunPod, a cloud GPU service.

License: MIT - free for personal and commercial use.
"""

import subprocess
import sys
from pathlib import Path

WHISPER_MODEL = "base.en"
VOXCPM_MODEL = "openbmb/VoxCPM2"

# Supported target languages: code -> (Helsinki model, language name, file suffix)
LANGUAGES = {
    "es": ("Helsinki-NLP/opus-mt-en-es", "Spanish", "es"),
    "tl": ("Helsinki-NLP/opus-mt-en-tl", "Tagalog", "tl"),
}

# Proper-noun corrections per language: applied to the translation before
# voice synthesis. Add entries as "wrong" -> "right". Matching is
# case-insensitive.
CORRECTIONS = {
    "es": {
        "arthur edward peso": "Arthur Edward Waite",
        "peso": "Waite",  # fallback if only the last name gets mangled
    },
    "tl": {
        # Add Tagalog corrections here as you find them
    },
}

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


def translate_segments(segments, lang):
    """Translate English segments with MarianMT."""
    from transformers import MarianMTModel, MarianTokenizer

    model_name, lang_name, _ = LANGUAGES[lang]
    print(f"Loading {lang_name} translation model...", flush=True)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
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
        out_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        results.append(out_text)
        print(f"  EN: {text[:60]}...", flush=True)
        print(f"  {lang.upper()}: {out_text[:60]}...", flush=True)
    return results


def apply_corrections(text, lang):
    """Apply proper-noun corrections (case-insensitive)."""
    import re
    for wrong, right in CORRECTIONS.get(lang, {}).items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return text


def dub_audio(audio: Path, segments, translations, lang):
    """Clone voice and synthesize dubbed audio with VoxCPM2."""
    from voxcpm import VoxCPM
    import soundfile as sf
    import numpy as np

    _, lang_name, suffix = LANGUAGES[lang]

    # Fix known proper-noun mistranslations before synthesis
    translations = [apply_corrections(t, lang) for t in translations]

    print("Loading VoxCPM2 voice cloning model...", flush=True)
    model = VoxCPM.from_pretrained(VOXCPM_MODEL, load_denoiser=False)

    # Use first 10 seconds of original audio as voice reference
    ref_audio = WORK_DIR / f"{audio.stem}_ref.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio), "-t", "10", "-ar", "16000",
         "-ac", "1", "-c:a", "pcm_s16le", str(ref_audio)],
        check=True, capture_output=True,
    )

    # Generate dubbed audio for each segment, concatenated in order.
    # No timing adjustments: the output flows naturally at whatever
    # length the translation comes out to. Adjust to your video in your
    # editor if needed.
    all_audio = []
    sample_rate = None

    for i, (seg, dubbed) in enumerate(zip(segments, translations)):
        if not dubbed.strip():
            continue
        print(
            f"  Synthesizing segment {i+1}/{len(translations)}: "
            f"{dubbed[:50]}...",
            flush=True,
        )
        wav = model.generate(
            text=dubbed,
            reference_wav_path=str(ref_audio),
            cfg_value=2.0,
            inference_timesteps=10,
        )
        if sample_rate is None:
            # NOTE: attribute path unverified - VoxCPM2 API may differ.
            # If this fails, check the voxcpm docs for the correct
            # sample rate attribute.
            sample_rate = model.tts_model.sample_rate
        all_audio.append(np.asarray(wav, dtype=np.float32))

    if not all_audio:
        raise ValueError("No audio generated - all translations were empty")

    combined = np.concatenate(all_audio)

    # Save dubbed audio as WAV then convert to MP3
    out_wav = audio.with_name(f"{audio.stem}_dubbed_{suffix}.wav")
    sf.write(str(out_wav), combined, sample_rate)
    print(f"Wrote dubbed audio: {out_wav}", flush=True)

    out_mp3 = audio.with_name(f"{audio.stem}_dubbed_{suffix}.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(out_wav), "-codec:a", "libmp3lame",
         "-q:a", "2", str(out_mp3)],
        check=True, capture_output=True,
    )
    print(f"Wrote dubbed MP3: {out_mp3}", flush=True)

    out_wav.unlink()
    return out_mp3


def main(lang, audio_files):
    _, lang_name, _ = LANGUAGES[lang]
    for a in audio_files:
        audio = Path(a)
        if not audio.exists():
            print(f"SKIP (not found): {audio}", flush=True)
            continue
        print(f"Transcribing: {audio.name}", flush=True)
        segments = transcribe(audio)
        print(
            f"Translating {len(segments)} segments to {lang_name}...",
            flush=True,
        )
        translations = translate_segments(segments, lang)
        print(f"Dubbing to {lang_name} in your voice...", flush=True)
        out_mp3 = dub_audio(audio, segments, translations, lang)
        print(f"Done: {out_mp3}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print(f"Supported languages: {', '.join(sorted(LANGUAGES))}")
        sys.exit(1)
    lang = sys.argv[1].lower()
    if lang not in LANGUAGES:
        print(f"Unknown language '{lang}'. Supported: {', '.join(sorted(LANGUAGES))}")
        sys.exit(1)
    main(lang, sys.argv[2:])
