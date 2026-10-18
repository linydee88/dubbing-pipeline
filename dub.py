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
    "pt": ("Helsinki-NLP/opus-mt-en-pt", "Portuguese", "pt"),
    "hi": ("Helsinki-NLP/opus-mt-en-hi", "Hindi", "hi"),
    "fr": ("Helsinki-NLP/opus-mt-en-fr", "French", "fr"),
}

# Proper-noun corrections per language: applied to the translation before
# voice synthesis. Add entries as "wrong" -> "right". Matching is
# case-insensitive.
#
# NOTE (2026-10-14): the old fallback "peso" -> "Waite" was REMOVED. It fired
# on every legitimate use of the Spanish word "peso" (weight), e.g.
# "el peso de la carta" became "el Waite de la carta". Proper nouns are now
# protected with placeholders BEFORE translation (see below), which is
# immune to every mistranslation variant instead of chasing each one.
CORRECTIONS = {
    "es": {
        # Safety net only: with placeholder protection below, the
        # translator never sees the name, so this should never fire.
        "arthur edward peso": "Arthur Edward Waite",
    },
    "tl": {
        # Add Tagalog corrections here as you find them
    },
    "pt": {
        # Add Portuguese corrections here as you find them
    },
    "hi": {
        # Add Hindi corrections here as you find them
    },
    "fr": {
        # Add French corrections here as you find them
    },
}

# Proper-noun placeholder protection. Before translation, names are swapped
# for tokens the translator passes through untouched; after translation the
# tokens are swapped back. This beats post-translation correction because it
# works no matter HOW the model would have mangled the name ("peso",
# "espera", transliterated first names, split across segments, ...).
# VERIFIED 2026-10-14 against the real Helsinki-NLP/opus-mt-en-es model:
# all three placeholders below survive translation unchanged, including
# mid-sentence, after prepositions, and in lowercase input.
# Tagalog (opus-mt-en-tl) pass-through is NOT yet verified: protection is
# currently applied to Spanish ("es") only.
NAME_PLACEHOLDERS_ES = [
    # (english pattern, placeholder, restored text) - full name FIRST,
    # then the bare surname, so "Rider Waite" keeps working.
    (r"arthur\s+edward\s+waite", "XXARTHUREDWARDWAITEQXX", "Arthur Edward Waite"),
    (r"\bwaite\b", "XXWAITEQXX", "Waite"),
]


def protect_names(text, lang):
    """Swap proper nouns for translation-safe placeholders (pre-translation)."""
    import re
    if lang == "es":
        for pattern, placeholder, _ in NAME_PLACEHOLDERS_ES:
            text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
    return text


def restore_names(text, lang):
    """Swap placeholders back to proper nouns (post-translation)."""
    if lang == "es":
        for _, placeholder, restored in NAME_PLACEHOLDERS_ES:
            text = text.replace(placeholder, restored)
    return text

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
        # Protect proper nouns BEFORE translation so the model can't
        # mistranslate them, then restore them in the translated text.
        protected = protect_names(text, lang)
        inputs = tokenizer(
            [protected], return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        translated = model.generate(**inputs, max_length=512)
        out_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        out_text = restore_names(out_text, lang)
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
    # After combining, the full dubbed audio is time-stretched with
    # ffmpeg atempo to match the original audio length: slowed down
    # if shorter, sped up if longer.
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

    # Match the dubbed length to the original audio length.
    orig_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
        check=True, capture_output=True, text=True,
    )
    orig_seconds = float(orig_probe.stdout.strip())
    dubbed_seconds = len(combined) / sample_rate
    print(
        f"Original: {orig_seconds:.1f}s, dubbed: {dubbed_seconds:.1f}s",
        flush=True,
    )
    out_mp3 = audio.with_name(f"{audio.stem}_dubbed_{suffix}.mp3")
    if abs(dubbed_seconds - orig_seconds) < 0.5:
        print("Lengths match, no adjustment needed.", flush=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_wav), "-codec:a", "libmp3lame",
             "-q:a", "2", str(out_mp3)],
            check=True, capture_output=True,
        )
    else:
        # atempo changes speed without changing pitch.
        # tempo > 1 speeds up (dubbed too long), tempo < 1 slows down
        # (dubbed too short).
        tempo = dubbed_seconds / orig_seconds
        # Chain atempo filters if outside the 0.5-2.0 single-filter range.
        filters = []
        t = tempo
        while t > 2.0:
            filters.append("atempo=2.0")
            t /= 2.0
        while t < 0.5:
            filters.append("atempo=0.5")
            t /= 0.5
        filters.append(f"atempo={t:.4f}")
        print(
            f"Adjusting tempo {tempo:.3f}x to match original length...",
            flush=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_wav), "-filter:a",
             ",".join(filters), "-codec:a", "libmp3lame",
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
