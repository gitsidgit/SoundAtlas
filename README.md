# SoundAtlas

> An audio anthology platform for querying, analysing, and exploring spoken and musical audio content at scale — with built-in synthetic speech detection.

SoundAtlas turns a folder of audio files — podcasts, interviews, music, recordings — into a richly navigable, searchable, and queryable library. It combines accurate multi-scale speaker diarization, segment-level synthetic speech detection, waveform visualisation, pre-computed peaks for instant playback, and speaker embedding vectors for similarity search into a single interactive gallery.

---

## Vision

Most audio content sits in flat file systems, unsearchable and unanalysed. SoundAtlas treats audio as structured data: who speaks when, what they say, how the audio is composed, and critically — whether any portion of it has been synthetically generated or tampered with.

Two parallel use cases share the same core pipeline:

- **Music** — identify singers, visualise vocal segments, catalogue stems and multi-artist collaborations
- **Podcasts / spoken word** — diarise hosts and guests, detect synthetic speech insertions, surface tampered content

The synthetic speech detection capability specifically targets the threat of localised insertion — where a single word or phrase (e.g. "yes" → "no") is replaced with TTS-generated audio within an otherwise genuine recording. This is distinct from whole-file deepfake detection and is underserved by existing tools.

---

## Project Files

```
SoundAtlas/
├── singer_waveform_tabulator.html   # Interactive gallery viewer (single file, no build)
├── integrity_pipeline.py            # Diarization + integrity analysis pipeline
├── download_data.sh                 # Test data downloader (FMA, LibriSpeech, AMI, MUSDB)
└── gallery.json                     # Pipeline output — loaded by the viewer at runtime
```

Audio files are referenced by absolute path in `gallery.json` (`audio_path` field) so they can live anywhere on disk.

---

## Current State

### 1. `singer_waveform_tabulator.html` — Interactive Gallery Viewer

A single-file browser application built on [Tabulator 6](https://tabulator.info) with Bootstrap 3 theming. No build step — serve via a local HTTP server and open in a browser.

**Data loading**
- Automatically fetches `gallery.json` on load and populates the table with real pipeline data
- Falls back to 120 synthetic demo tracks if `gallery.json` is not present, with a clear "Demo mode" subtitle

**Table columns**
- `#` / File / Duration / Channels / Sample rate
- **Integrity** — colour-coded verdict badge: green `GENUINE`, amber `PARTIAL FAKE`, red `SYNTHETIC`, with percentage confidence score
- **Speakers** — pill badges per detected speaker, colour-coded
- **Waveform ▶** — inline SVG with per-speaker colour bands and red fake-segment overlays; click to open the audio player

**Table features**
- Multi-column sort builder with add/remove sort levels, synced bidirectionally with shift+click header sorting
- Toggleable column header filters including verdict dropdown (All / Genuine / Partial fake / Synthetic)
- Column visibility toggle, reorder (drag header), and resize (drag edge), all persisted to `localStorage`
- Amplitude zoom slider, live text search, configurable pagination (10 / 20 / 50 / 100)
- Speaker legend — click any speaker to mute/unmute their segments across all waveforms

**WaveSurfer player modal**
- Opens on waveform cell click; renders instantly from pre-computed peaks — no audio download needed for the visual
- Speaker segment regions and fake segment regions (red, `FAKE` labelled) overlaid simultaneously
- Click-to-seek progress bar, play/pause toggle, elapsed/total time, language/gender/verdict in meta line
- Streams audio from `audio_path` (absolute path) — audio files do not need to be co-located with the viewer
- Close via ✕, backdrop click, or Escape

### 2. `integrity_pipeline.py` — Diarization + Integrity Pipeline

Replaces `singer_pipeline.py`. Processes a folder of WAV files and writes `gallery.json`. All models are fully local — no tokens, no accounts, no internet connection required after first run.

**Diarization — NeMo MSDD**

Uses NVIDIA NeMo's Multi-Scale Diarization Decoder, a significant accuracy improvement over the previous resemblyzer-based pipeline:

- MarbleNet VAD removes silence and segments speech regions
- TitaNet-Large extracts speaker embeddings at five simultaneous scales (1.5s, 1.25s, 1.0s, 0.75s, 0.5s)
- MSDD neural decoder dynamically weights each scale to resolve overlapping speech and close boundaries
- The 0.5s shortest scale is the critical resolution for detecting single-word substitutions
- All three models download automatically from NVIDIA NGC on first run (~300MB total), then cached permanently at `~/.cache/torch/NeMo/`

**Integrity analysis — AASIST sliding window**

Runs a sliding window (default 1s window, 0.25s hop) over the audio and scores each frame for synthetic speech artefacts using either:

- Full AASIST (graph attention network, ~6MB weights) if `~/aasist` is cloned from GitHub
- Lightweight proxy spectral classifier (built-in, zero download) as fallback

Splice point detection identifies real→fake boundaries independently of the frame scores, useful for subtle insertions where the fake content itself may score borderline.

**Language identification — AmberNet**

NeMo AmberNet covers 107 languages (VoxLingua107 trained). Uses direct forward pass on the audio tensor via `EncDecSpeakerLabelModel`.

**Outputs per track (stored in `gallery.json`)**

| Field | Description |
|---|---|
| `name` | Filename |
| `audio_path` | Absolute path to the WAV file |
| `durRaw` | Duration in seconds |
| `sr` / `ch` | Sample rate / channels |
| `segs` | Speaker segments: `singer`, `start`, `end`, `duration`, `confidence` |
| `waveData` | 200-point normalised float array for inline SVG |
| `peaks` | WaveSurfer.js pre-decoded peaks `{version, channels, data, ...}` |
| `speakerVectors` | Per-speaker 192-d L2-normalised TitaNet embedding for cosine similarity search |
| `totalSpeechSec` / `silenceSec` / `speechRatio` | File-level speech activity statistics |
| `speakerStats` | Per-speaker `{totalSec, segments, sharePercent}` |
| `language` / `languageConfidence` | AmberNet language ID (107 languages) |
| `gender` / `genderConfidence` | Gender detection (`unknown` unless SpeechBrain installed) |
| `integrity` | File-level fake probability 0.0 → 1.0 (requires `--integrity` flag) |
| `verdict` | `GENUINE` / `PARTIAL_FAKE` / `SYNTHETIC` / `UNKNOWN` |
| `fakeSegments` | `[{start, end, score, method}]` — localised fake regions |
| `splicePoints` | `[float]` — detected real→fake boundary times in seconds |
| `svg` | Pre-rendered SVG with speaker + fake overlays (requires `--svg` flag) |

**Install**

```bash
# Core NeMo stack
pip install nemo_toolkit[asr]
pip install webrtcvad-wheels        # fixes webrtcvad on Python 3.12+
pip install "huggingface_hub>=0.34.0,<1.0"  # fixes version conflict from NeMo install

# Optional: full AASIST for production-grade fake detection
git clone https://github.com/clovaai/aasist ~/aasist

# Optional: gender detection
pip install speechbrain
```

**Usage**

```bash
# Diarization only
python3 integrity_pipeline.py --audio_dir ./audio --output gallery.json

# With integrity analysis
python3 integrity_pipeline.py --audio_dir ./audio --integrity

# Fix speaker count when known (improves accuracy)
python3 integrity_pipeline.py --audio_dir ./audio --num_speakers 2

# Full run — diarization + integrity + pre-rendered SVGs
python3 integrity_pipeline.py \
    --audio_dir     ./audio \
    --output        gallery.json \
    --min_speakers  1 \
    --max_speakers  4 \
    --integrity \
    --svg \
    --window_sec    1.0 \
    --hop_sec       0.25

# Skip language/gender classification (faster)
python3 integrity_pipeline.py --audio_dir ./audio --skip_classify

# Full traceback on errors
python3 integrity_pipeline.py --audio_dir ./audio --verbose
```

**Speaker vector similarity search** (client-side example)

```javascript
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
// Find all tracks containing a speaker similar to Speaker A in track 0
const query = tracks[0].speakerVectors['Speaker A'];
const matches = tracks.filter(t =>
  Object.values(t.speakerVectors).some(v => cosineSim(query, v) > 0.75)
);
```

### 3. `download_data.sh` — Test Data Downloader

```bash
chmod +x download_data.sh

./download_data.sh quick     # ~500MB  — LibriSpeech mini, ideal for first pipeline test
./download_data.sh podcast   # ~1GB    — LibriSpeech dev + AMI meetings
./download_data.sh music     # ~8GB    — FMA Small + metadata
./download_data.sh all       # ~10GB   — everything above
```

For testing the integrity detector against known genuine and spoofed samples, the ASVspoof2019 LA dataset is freely available and covers 13 TTS/VC attack systems:

```bash
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
unzip LA.zip
# Genuine:  LA/ASVspoof2019_LA_eval/flac/ — bonafide entries in protocol file
# Spoofed:  LA/ASVspoof2019_LA_eval/flac/ — spoof entries, attack IDs A07–A19
```

---

## Getting Started

**1. Serve the viewer with synthetic demo data**

```bash
python3 -m http.server 8080
# open http://localhost:8080/singer_waveform_tabulator.html
```

> Opening as a `file://` URL will not work — `fetch('gallery.json')` requires HTTP.

**2. Install pipeline dependencies**

```bash
pip install nemo_toolkit[asr] webrtcvad-wheels
pip install "huggingface_hub>=0.34.0,<1.0"
```

**3. Download test audio**

```bash
chmod +x download_data.sh && ./download_data.sh quick
```

**4. Run the pipeline**

```bash
python3 integrity_pipeline.py \
  --audio_dir data/librispeech_wav \
  --output gallery.json \
  --min_speakers 1 \
  --max_speakers 3 \
  --integrity \
  --svg
```

**5. Reload the viewer**

Refresh the browser — `gallery.json` is detected automatically on load.

---

## Recommended Audio Sources

| Source | Content | Licence | Size |
|---|---|---|---|
| [LibriSpeech](https://www.openslr.org/12) | Multi-speaker read speech | CC BY 4.0 | 337MB (dev-clean) |
| [ASVspoof2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) | Genuine + 13 TTS/VC spoofed systems | Research | ~10GB |
| [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | Real multi-speaker meetings | CC BY 4.0 | ~100MB/session |
| [FMA Small](https://github.com/mdeff/fma) | 8000 × 30s music clips, 8 genres | CC | 7.2GB |
| [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) | 150 tracks with isolated stems | Academic | 30GB |

---

## Architecture

```
WAV files
    │
    ▼
integrity_pipeline.py
  ├── MarbleNet VAD              →  speech/silence segmentation
  ├── NeMo MSDD diarization      →  who speaks when
  │     TitaNet ×5 scales              192-d speaker embeddings
  │     MSDD neural decoder            multi-scale boundary resolution
  ├── AmberNet                   →  language ID (107 languages)
  ├── AASIST sliding window      →  per-frame fake probability [0,1]
  │     splice point detection         real→fake boundary times
  ├── speaker stats              →  speech/silence/share per speaker
  └── gallery.json
              │
              ▼
singer_waveform_tabulator.html
  ├── fetch('gallery.json')        auto-load or 120-track synthetic fallback
  ├── Tabulator 6                  grid, sort, filter, pagination, persistence
  ├── WaveSurfer 7                 audio player — speaker + fake regions
  ├── Bootstrap 3                  theming
  └── inline SVG waveforms         speaker colour bands + red fake overlays
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Gallery viewer | HTML / CSS / vanilla JS |
| Data grid | [Tabulator 6.3](https://tabulator.info) |
| Audio player | [WaveSurfer.js 7](https://wavesurfer.xyz) |
| Styling | Bootstrap 3.4 |
| VAD | NeMo MarbleNet |
| Diarization | NeMo MSDD + TitaNet-Large |
| Language ID | NeMo AmberNet (107 languages) |
| Integrity detection | AASIST / lightweight spectral proxy |
| Transcription (planned) | [OpenAI Whisper](https://github.com/openai/whisper) |
| AI querying (planned) | Anthropic Claude API |

---

## Known Limitations

**Integrity detection accuracy** — the built-in proxy classifier scores spectral features heuristically and produces false positives on clean studio recordings such as LibriSpeech. For better accuracy, clone the full AASIST repo (`git clone https://github.com/clovaai/aasist ~/aasist`). Even full AASIST generalises poorly to TTS systems released after its training data (ElevenLabs, Suno, Udio) — fine-tuning on self-generated samples from current systems is the intended path to production accuracy.

**Gender detection** — NeMo does not currently ship a pretrained gender model. The pipeline returns `unknown` unless SpeechBrain is installed, which provides a wav2vec2-based gender classifier.

**NeMo logging** — NeMo re-initialises its own Python logging handlers on import, partially overriding suppression. Some `[NeMo I]` informational lines may appear despite suppression. Use `2>/dev/null` or pipe through `grep` to filter if needed.

**Processing speed** — first file takes ~60–90 seconds due to model loading into CUDA. Subsequent files take ~15–30 seconds each depending on duration. The RTX 3060 12GB handles all four NeMo models simultaneously with headroom.

---

## Planned Features

### Near-term
- [ ] **Full AASIST integration** — auto-detect `~/aasist` and load weights for production-grade integrity scoring
- [ ] **ASVspoof fine-tuning** — script to fine-tune on self-generated ElevenLabs/Suno/Bark samples
- [ ] **Speaker name editor** — rename `Speaker A / B` labels inline, persisted to `localStorage`
- [ ] **Segment-level playback** — click a speaker segment to play just that portion

### Medium-term
- [ ] **Whisper transcription** — add to pipeline; transcript column + full-text search in viewer
- [ ] **Speaker similarity search UI** — click a speaker pill to find matching speakers across the collection
- [ ] **Integrity heatmap view** — per-collection timeline of fake probability
- [ ] **Episode/album grouping** — Tabulator row grouping by series, date, or speaker

### Longer-term
- [ ] **AI-powered querying** — natural language questions across the whole collection ("find all segments where Speaker A sounds synthetic")
- [ ] **Export & reporting** — filtered views to CSV, per-track integrity reports
- [ ] **Web server mode** — Flask/FastAPI backend with real-time pipeline triggering
- [ ] **Attribution** — identify which TTS system generated a detected fake segment

---

## Contributing

Issues, ideas, and pull requests are welcome. The most impactful contributions right now are:

- Testing the integrity detector against ASVspoof2019 and current TTS systems to calibrate thresholds
- Full AASIST integration and fine-tuning pipeline for current neural vocoders
- Whisper transcription integration
- Speaker name editor UI in the viewer

---

## Licence

MIT
