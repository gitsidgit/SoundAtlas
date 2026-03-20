#!/usr/bin/env python3
"""
SoundAtlas — audio analysis pipeline
======================================
Fully local, zero tokens, zero account required.

Install
-------
    pip install resemblyzer scikit-learn numpy scipy

    resemblyzer downloads its pretrained GE2E model (~17MB) automatically
    on first run from a public URL. No account needed, ever.

Outputs (per file)
------------------
  gallery.json        — master data file for the HTML viewer, containing:
    waveData          — 200-point normalised float array for the inline SVG
    peaks             — WaveSurfer.js compatible peaks {version,data,...}
    segs              — diarized speaker segments with start/end/confidence
    speakerVectors    — per-speaker mean embedding (256-d float) for search
    svg               — pre-rendered inline SVG string (optional, --svg flag)

Usage
-----
    # Basic
    python singer_pipeline.py --audio_dir ./audio

    # Fix speaker count for better accuracy
    python singer_pipeline.py --audio_dir ./audio --num_speakers 2

    # Include pre-rendered SVGs in the JSON (eliminates client-side rendering)
    python singer_pipeline.py --audio_dir ./audio --svg

    # All options
    python singer_pipeline.py \\
        --audio_dir    ./audio \\
        --output       gallery.json \\
        --num_speakers 2 \\
        --min_speakers 2 \\
        --max_speakers 5 \\
        --svg \\
        --peaks_per_sec 20 \\
        --wavedata_pts  200

WaveSurfer.js integration
--------------------------
    const ws = WaveSurfer.create({ container: '#waveform' });
    ws.load('audio/track.wav', track.peaks.data);
    // Waveform renders instantly from pre-computed peaks.
    // Audio begins streaming only on play.

Speaker vector search
---------------------
    // Cosine similarity between two speaker vectors
    function cosineSim(a, b) {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
    // Find all tracks where Singer A appears (similarity > 0.75)
    const query = tracks[0].speakerVectors['Singer A'];
    const matches = tracks.filter(t =>
        Object.values(t.speakerVectors).some(v => cosineSim(query, v) > 0.75)
    );
"""

import argparse
import json
import os
import wave
import struct
import math
import numpy as np
from pathlib import Path


# ── Audio I/O ─────────────────────────────────────────────────────────────────

def read_wav(path: str):
    """
    Read WAV → (float32 mono, sample_rate, original_channel_count).
    Handles 8/16/32-bit PCM.
    """
    with wave.open(path, "rb") as wf:
        n_ch = wf.getnchannels()
        sw   = wf.getsampwidth()
        sr   = wf.getframerate()
        raw  = wf.readframes(wf.getnframes())

    if sw == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw} bytes")

    if n_ch > 1:
        a = a.reshape(-1, n_ch).mean(axis=1)

    return a, sr, n_ch


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample audio array from src_sr to dst_sr."""
    if src_sr == dst_sr:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_sr, dst_sr)
        return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:
        n_out = int(len(audio) * dst_sr / src_sr)
        idx   = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


# ── Waveform data (inline SVG + Tabulator viewer) ─────────────────────────────

def compute_wavedata(audio: np.ndarray, pts: int = 200) -> list:
    """
    200-point normalised float array for the inline SVG viewer.
    Picks evenly spaced samples — fast and sufficient for 30px-tall SVGs.
    """
    idx     = np.linspace(0, len(audio) - 1, pts, dtype=int)
    samples = audio[idx]
    mx      = np.max(np.abs(samples)) or 1.0
    return [round(float(v / mx), 4) for v in samples]


# ── WaveSurfer peaks ──────────────────────────────────────────────────────────

def compute_peaks(audio: np.ndarray, sr: int, pixels_per_sec: int = 20) -> dict:
    """
    Compute WaveSurfer.js compatible pre-decoded peaks.

    WaveSurfer expects the BBC audiowaveform JSON format:
        {
          "version": 2,
          "channels": 1,
          "sample_rate": 16000,
          "samples_per_pixel": 800,
          "bits": 8,
          "length": 1234,
          "data": [-45, 62, -38, 71, ...]   ← interleaved min/max per pixel
        }

    Each pixel = one [min, max] pair, both as 8-bit signed integers [-128,127].
    This is exactly what wavesurfer.load(url, peaks.data) expects.

    Load in WaveSurfer:
        wavesurfer.load('audio.wav', peaksObj.data);
    """
    samples_per_pixel = max(1, sr // pixels_per_sec)
    n_pixels          = math.ceil(len(audio) / samples_per_pixel)
    data              = []

    for i in range(n_pixels):
        chunk = audio[i * samples_per_pixel : (i + 1) * samples_per_pixel]
        if len(chunk) == 0:
            data.extend([0, 0])
            continue
        mn = int(np.min(chunk) * 127)
        mx = int(np.max(chunk) * 127)
        # Clamp to [-128, 127]
        data.append(max(-128, min(127, mn)))
        data.append(max(-128, min(127, mx)))

    return {
        "version":          2,
        "channels":         1,
        "sample_rate":      sr,
        "samples_per_pixel": samples_per_pixel,
        "bits":             8,
        "length":           n_pixels,
        "data":             data
    }


# ── SVG generation ────────────────────────────────────────────────────────────

def build_svg(wavedata: list, segments: list, duration: float,
              width: int = 400, height: int = 30) -> str:
    """
    Build a self-contained SVG string with speaker-coloured segment bands.
    Same visual as the JS viewer but pre-rendered server-side.

    Stored in gallery.json and rendered directly via:
        cell.getElement().innerHTML = row.svg;
    """
    COLORS = ["#5b7cfa","#e74c7d","#27ae82","#f39c12",
              "#9b59b6","#e74c3c","#16a085","#d35400"]

    pts     = len(wavedata)
    mid     = height / 2
    amp     = height * 0.44
    xs      = (width - 1) / (pts - 1)

    # Map singer names to colours
    singers   = list({s["singer"] for s in segments})
    color_map = {s: COLORS[i % len(COLORS)] for i, s in enumerate(singers)}

    # Ghost polyline (full waveform, grey)
    ghost = " ".join(
        f"{i*xs:.1f},{max(0, min(height, mid - wavedata[i]*amp)):.1f}"
        for i in range(pts)
    )

    # Per-singer filled bands
    fills  = ""
    lines  = ""
    for seg in segments:
        col = color_map[seg["singer"]]
        i0  = round(seg["start"] / duration * (pts - 1))
        i1  = min(pts - 1, round(seg["end"] / duration * (pts - 1)))
        if i1 <= i0:
            continue
        top = ""
        bot = ""
        for pi in range(i0, i1 + 1):
            px  = f"{pi*xs:.1f}"
            py  = f"{max(0, min(height, mid - wavedata[pi]*amp)):.1f}"
            pyb = f"{max(0, min(height, mid + wavedata[pi]*amp)):.1f}"
            top += f"{px},{py} "
            bot  = f"{px},{pyb} " + bot
        x0 = f"{i0*xs:.1f}"
        x1 = f"{i1*xs:.1f}"
        fills += (f'<polygon points="{x0},{mid} {top}{x1},{mid} {bot}" '
                  f'fill="{col}" fill-opacity="0.28"/>')
        lines += (f'<polyline points="{x0},{mid} {top}{x1},{mid}" '
                  f'fill="none" stroke="{col}" stroke-width="1.5"/>')

    return (
        f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="#f7f8fa"/>'
        f'<line x1="0" y1="{mid}" x2="{width}" y2="{mid}" '
        f'stroke="#ddd" stroke-width="0.8"/>'
        f'<polyline points="{ghost}" fill="none" stroke="#ccc" stroke-width="0.9"/>'
        f'{fills}{lines}</svg>'
    )


# ── Diarization with resemblyzer ──────────────────────────────────────────────

def diarize(audio: np.ndarray, sr: int, args) -> tuple:
    """
    Speaker diarization + per-speaker embedding vectors using resemblyzer.

    Uses the official resemblyzer API:
        encoder.embed_utterance(wav, return_partials=True, rate=N)
    which returns (utterance_embed, cont_embeds, wav_splits).
    cont_embeds is an (n_windows, 256) array of rolling embeddings.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import normalize
    except ImportError:
        raise SystemExit(
            "\nERROR: resemblyzer or scikit-learn not installed.\n"
            "Fix:  pip install resemblyzer scikit-learn\n"
        )

    # resemblyzer requires 16kHz mono — resample if needed
    audio16 = resample(audio, sr, 16000)

    # Write to temp WAV so preprocess_wav can load it
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    _write_wav16(audio16, tmp_path)

    try:
        wav = preprocess_wav(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Rate=16 means one embedding per ~0.0625s — good granularity for diarization
    encoder = VoiceEncoder()
    _, cont_embeds, wav_splits = encoder.embed_utterance(
        wav, return_partials=True, rate=16
    )
    # cont_embeds: (n_windows, 256)

    n_windows  = len(cont_embeds)
    hop_secs   = len(wav) / 16000 / n_windows  # actual hop in seconds

    if n_windows < 2:
        # Audio too short to diarize — treat as single speaker
        embed     = encoder.embed_utterance(wav)
        segments  = [{"singer": "Singer A", "start": 0.0,
                       "end": round(len(audio)/sr, 3), "confidence": 0.88}]
        vectors   = {"Singer A": embed.tolist()}
        return segments, vectors

    # ── Determine speaker count ────────────────────────────────────────────
    n_speakers = args.num_speakers
    if n_speakers is None:
        lo = max(2, args.min_speakers or 2)
        hi = min(8, args.max_speakers or 6, n_windows - 1)
        if lo >= hi:
            n_speakers = lo
        else:
            best_k, best_score = lo, -1
            for k in range(lo, hi + 1):
                km    = KMeans(n_clusters=k, random_state=0, n_init=10)
                lbls  = km.fit_predict(cont_embeds)
                score = silhouette_score(cont_embeds, lbls)
                if score > best_score:
                    best_k, best_score = k, score
            n_speakers = best_k

    # ── Cluster window embeddings ──────────────────────────────────────────
    clust  = AgglomerativeClustering(n_clusters=n_speakers)
    labels = clust.fit_predict(cont_embeds)

    # ── Build segments from label runs ─────────────────────────────────────
    raw_segs  = []
    cur_label = labels[0]
    cur_start = 0.0

    for i in range(1, n_windows):
        if labels[i] != cur_label:
            raw_segs.append({
                "speaker": f"SPEAKER_{cur_label:02d}",
                "start":   round(cur_start, 3),
                "end":     round(i * hop_secs, 3)
            })
            cur_label = labels[i]
            cur_start = i * hop_secs

    raw_segs.append({
        "speaker": f"SPEAKER_{cur_label:02d}",
        "start":   round(cur_start, 3),
        "end":     round(len(audio) / sr, 3)
    })

    raw_segs = _merge(raw_segs)
    segments = _name(raw_segs)

    # ── Per-speaker mean vectors (256-d, L2 normalised) ────────────────────
    spk_embeds = {}
    for win_idx, lbl in enumerate(labels):
        spk_embeds.setdefault(f"SPEAKER_{lbl:02d}", []).append(cont_embeds[win_idx])

    chars     = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    spk_ids   = sorted(spk_embeds.keys())
    name_map  = {sid: f"Singer {chars[i % 26]}" for i, sid in enumerate(spk_ids)}
    spk_vectors = {
        name_map[k]: normalize(
            np.mean(v, axis=0, keepdims=True)
        )[0].tolist()
        for k, v in spk_embeds.items()
    }

    return segments, spk_vectors


def _write_wav16(audio: np.ndarray, path: str):
    """Write float32 mono array as 16kHz 16-bit WAV."""
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())


def _merge(segments: list, gap: float = 0.4) -> list:
    """Merge consecutive same-speaker segments within `gap` seconds."""
    if not segments:
        return []
    out = [segments[0].copy()]
    for s in segments[1:]:
        p = out[-1]
        if s["speaker"] == p["speaker"] and s["start"] - p["end"] <= gap:
            p["end"] = s["end"]
        else:
            out.append(s.copy())
    return out


def _name(segments: list) -> list:
    """Map SPEAKER_00 → Singer A, etc."""
    mapping, chars = {}, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for s in segments:
        sid = s["speaker"]
        if sid not in mapping:
            i = len(mapping)
            mapping[sid] = f"Singer {chars[i] if i < 26 else f'S{i}'}"
    return [{
        "singer":     mapping[s["speaker"]],
        "start":      round(s["start"], 3),
        "end":        round(s["end"],   3),
        "confidence": 0.88
    } for s in segments]


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    wav_files = sorted(Path(args.audio_dir).glob("**/*.wav"))
    if not wav_files:
        raise SystemExit(f"No WAV files found in {args.audio_dir}")

    print(f"Found {len(wav_files)} WAV file(s)")
    print(f"SVG generation : {'yes' if args.svg else 'no (client-side)'}")
    print(f"Peaks res      : {args.peaks_per_sec} px/sec")
    print(f"Wavedata pts   : {args.wavedata_pts}")
    print()

    tracks = []
    errors = []

    for i, path in enumerate(wav_files):
        print(f"[{i+1}/{len(wav_files)}] {path.name}")
        try:
            audio, sr, n_ch = read_wav(str(path))
            duration        = len(audio) / sr

            print(f"        {duration:.1f}s  {sr}Hz  "
                  f"{'Stereo' if n_ch > 1 else 'Mono'}")

            # Waveform data (200pt float array for inline SVG viewer)
            wavedata = compute_wavedata(audio, args.wavedata_pts)

            # WaveSurfer peaks
            peaks = compute_peaks(audio, sr, args.peaks_per_sec)

            # Diarization + speaker vectors
            segments, spk_vectors = diarize(audio, sr, args)

            speakers = sorted({s["singer"] for s in segments})
            print(f"        {len(speakers)} speaker(s): {', '.join(speakers)}"
                  f"  ({len(segments)} segments)")
            print(f"        Vector dims: {len(next(iter(spk_vectors.values()), []))}d"
                  f" × {len(spk_vectors)} speaker(s)")

            track = {
                "id":             i + 1,
                "idx":            i + 1,
                "name":           path.name,
                "genre":          "unknown",
                "durRaw":         round(duration),
                "sr":             sr,
                "ch":             n_ch,
                "segs":           segments,
                "waveData":       wavedata,
                "peaks":          peaks,
                "speakerVectors": spk_vectors,
                "singers":        "_"
            }

            # Optional: pre-render SVG server-side
            if args.svg:
                track["svg"] = build_svg(wavedata, segments, duration)

            tracks.append(track)

        except Exception as exc:
            print(f"        WARNING: skipped — {exc}")
            errors.append((path.name, str(exc)))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tracks, f, indent=2)


    print(f"\n{'='*52}")
    print(f"Done.  {len(tracks)} tracks → {args.output}")
    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    print(f"{'='*52}\n")


def main():
    p = argparse.ArgumentParser(
        description="SoundAtlas pipeline — resemblyzer, fully local, zero tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--audio_dir",     required=True,
                   help="Folder of WAV files (searched recursively)")
    p.add_argument("--output",        default="gallery.json",
                   help="Output JSON path (default: gallery.json)")
    p.add_argument("--num_speakers",  type=int, default=None,
                   help="Fix speaker count (improves accuracy when known)")
    p.add_argument("--min_speakers",  type=int, default=None,
                   help="Minimum speakers to detect")
    p.add_argument("--max_speakers",  type=int, default=None,
                   help="Maximum speakers to detect")
    p.add_argument("--svg",           action="store_true",
                   help="Pre-render SVGs server-side and embed in JSON")
    p.add_argument("--peaks_per_sec", type=int, default=20,
                   help="WaveSurfer peaks resolution in pixels/sec (default: 20)")
    p.add_argument("--wavedata_pts",  type=int, default=200,
                   help="Points for inline SVG wavedata (default: 200)")

    run(p.parse_args())


if __name__ == "__main__":
    main()