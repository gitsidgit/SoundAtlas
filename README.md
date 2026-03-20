# SoundAtlas

> An audio anthology platform for querying, analysing, and exploring spoken and musical audio content at scale.

SoundAtlas turns a folder of audio files — podcasts, interviews, music, recordings — into a richly navigable, searchable, and queryable library. It combines automated speaker diarization, waveform visualisation, pre-computed peaks for instant playback, and speaker embedding vectors for similarity search into a single interactive gallery.

---

## Vision

Most audio content sits in flat file systems, unsearchable and unanalysed. SoundAtlas treats audio as structured data: who speaks when, what they say, how the audio is composed, and how it relates to other content in your collection.

Two parallel use cases share the same core pipeline:

- **Music** — identify singers, visualise vocal segments, catalogue stems and multi-artist collaborations, track an artist's history of releases across time
- **Podcasts** — diarise hosts and guests, extract transcripts, surface quotes and topics across episodes

The longer-term goal is a general-purpose audio intelligence layer that works across any spoken or musical content.

---

## Project Files

```
SoundAtlas/
├── singer_waveform_tabulator.html   # Interactive gallery viewer (single file, no build)
├── singer_pipeline.py               # Speaker diarization + data pipeline
├── download_data.sh                 # Test data downloader (FMA, LibriSpeech, AMI, MUSDB)
└── gallery.json                     # Pipeline output — loaded by the viewer at runtime
```

Place your WAV audio files alongside `gallery.json`. Filenames must match the `name` field in the JSON exactly.

---

## Current State

### 1. `singer_waveform_tabulator.html` — Interactive Gallery Viewer

A single-file browser application built on [Tabulator 6](https://tabulator.info) with Bootstrap 3 theming. No build step — open directly in a browser.

**Data loading**
- Automatically fetches `gallery.json` on load and populates the table with real pipeline data
- Falls back to 120 synthetic demo tracks if `gallery.json` is not present, clearly indicated in the subtitle

**Table features**
- Inline SVG waveforms with per-speaker colour-coded segment bands
- Click any waveform cell to open the WaveSurfer audio player modal
- Speaker/singer legend — click to mute/unmute any speaker across all visible waveforms
- Multi-column sort builder — add/remove sort levels with a UI panel, synced bidirectionally with shift+click header sorting
- Toggleable column header filters — text input, numeric, and dropdown (Channels, Sample rate)
- Column visibility toggle — show/hide any column, state persisted to `localStorage`
- Column reorder (drag header) and resize (drag right edge)
- Amplitude zoom slider
- Live text search across filename, genre, and speaker names
- Pagination with configurable page size (10 / 20 / 50 / 100)
- Full persistence — sort, filters, column widths, order, and visibility all survive page reload

**WaveSurfer player modal**
- Opens on waveform cell click
- Renders waveform **instantly** from pre-computed peaks stored in `gallery.json` — no audio download needed for the visual
- Speaker segment regions overlaid on the waveform in matching colours
- Click-to-seek progress bar, play/pause toggle, elapsed/total time display
- Streams audio on demand — file must be in the same folder as `gallery.json`
- Close via ✕ button, backdrop click, or Escape key

### 2. `singer_pipeline.py` — Speaker Diarization Pipeline

Processes a folder of WAV files and writes `gallery.json`. Uses [resemblyzer](https://github.com/resemble-ai/Resemblyzer) for speaker diarization — fully local, zero tokens, zero accounts required.

**Outputs per track (all stored in `gallery.json`)**

| Field | Description |
|---|---|
| `waveData` | 200-point normalised float array for the inline SVG viewer |
| `peaks` | WaveSurfer.js compatible pre-decoded peaks `{version, data, ...}` |
| `segs` | Speaker segments with `singer`, `start`, `end`, `confidence` |
| `speakerVectors` | Per-speaker 256-d embedding for cosine similarity search |
| `svg` | Pre-rendered SVG string (optional, `--svg` flag) |

**Install**

```bash
pip install resemblyzer scikit-learn numpy scipy
pip install webrtcvad-wheels   # fixes webrtcvad on Python 3.12+
```

resemblyzer downloads its ~17MB GE2E model automatically on first run from a public URL. No account needed, ever.

**Usage**

```bash
# Basic
python3 singer_pipeline.py --audio_dir ./audio

# Fix speaker count for better accuracy when you know it
python3 singer_pipeline.py --audio_dir ./audio --num_speakers 2

# Constrain speaker range
python3 singer_pipeline.py --audio_dir ./audio --min_speakers 2 --max_speakers 5

# Pre-render SVGs server-side (faster gallery load)
python3 singer_pipeline.py --audio_dir ./audio --svg

# All options
python3 singer_pipeline.py \
    --audio_dir    ./audio \
    --output       gallery.json \
    --num_speakers 2 \
    --min_speakers 1 \
    --max_speakers 5 \
    --svg \
    --peaks_per_sec 20 \
    --wavedata_pts  200
```

**Speaker vector search** (client-side example)

```javascript
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
// Find all tracks containing a speaker similar to Singer A in track 0
const query = tracks[0].speakerVectors['Singer A'];
const matches = tracks.filter(t =>
  Object.values(t.speakerVectors).some(v => cosineSim(query, v) > 0.75)
);
```

### 3. `download_data.sh` — Test Data Downloader

```bash
chmod +x download_data.sh

./download_data.sh quick     # ~500MB  — LibriSpeech mini, ideal for first pipeline test
./download_data.sh podcast   # ~1GB    — LibriSpeech dev + AMI meetings
./download_data.sh music     # ~8GB    — FMA Small + FMA metadata
./download_data.sh all       # ~10GB   — everything above
```

Includes post-download conversion of FLAC → WAV (LibriSpeech) and MP3 → WAV (FMA) via ffmpeg.

---

## Getting Started

**1. Clone and open the viewer (no pipeline needed — uses synthetic data)**

```bash
open singer_waveform_tabulator.html   # macOS
# or just double-click the file in Windows/Linux
```

**2. Download test audio**

```bash
chmod +x download_data.sh
./download_data.sh quick
```

**3. Install pipeline dependencies**

```bash
pip install resemblyzer scikit-learn numpy scipy webrtcvad-wheels
```

**4. Run the pipeline**

```bash
python3 singer_pipeline.py \
  --audio_dir data/librispeech_wav \
  --output gallery.json \
  --min_speakers 1 \
  --max_speakers 2 \
  --svg
```

**5. Open the viewer with real data**

Place `gallery.json` (and optionally your WAV files) in the same folder as `singer_waveform_tabulator.html`, then open it in a browser. The viewer detects the JSON automatically.

> **Note:** Due to browser security restrictions, `fetch('gallery.json')` requires the page to be served over HTTP rather than opened as a local `file://` URL. Use any simple local server:
> ```bash
> python3 -m http.server 8080
> # then open http://localhost:8080/singer_waveform_tabulator.html
> ```

---

## Recommended Audio Sources

| Source | Content | Licence | Size |
|---|---|---|---|
| [LibriSpeech](https://www.openslr.org/12) | Multi-speaker read speech | CC BY 4.0 | 337MB (dev-clean) |
| [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | Real multi-speaker meetings | CC BY 4.0 | ~100MB per session |
| [FMA Small](https://github.com/mdeff/fma) | 8000 × 30s music clips, 8 genres | CC | 7.2GB |
| [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) | 150 tracks with isolated stems | Academic | 30GB |
| [ccMixter](https://ccmixter.org) | Vocal collabs and duets | CC | Varies |

---

## Architecture

```
WAV files
    │
    ▼
singer_pipeline.py
  ├── resemblyzer diarization  →  speaker segments + 256-d vectors
  ├── waveform downsampling    →  200-pt float array (SVG viewer)
  ├── peaks computation        →  WaveSurfer pre-decoded peaks
  └── gallery.json
              │
              ▼
singer_waveform_tabulator.html
  ├── fetch('gallery.json')        auto-load or synthetic fallback
  ├── Tabulator 6                  grid, sort, filter, pagination, persistence
  ├── WaveSurfer 7                 audio playback modal with speaker regions
  ├── Bootstrap 3                  theming
  └── inline SVG waveforms         zero-dependency, client or server rendered
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Gallery viewer | HTML / CSS / vanilla JS |
| Data grid | [Tabulator 6.3](https://tabulator.info) |
| Audio player | [WaveSurfer.js 7](https://wavesurfer.xyz) |
| Styling | Bootstrap 3.4 |
| Diarization | [resemblyzer](https://github.com/resemble-ai/Resemblyzer) |
| Transcription (planned) | [OpenAI Whisper](https://github.com/openai/whisper) |
| AI querying (planned) | Anthropic Claude API |

---

## Planned Features

### Near-term
- [ ] **Speaker name editor** — rename `Singer A / B / C` labels to real names inline, persisted to `localStorage`
- [ ] **Segment-level playback** — click a coloured speaker segment to play just that portion
- [ ] **Transcript column** — display Whisper-generated transcript snippets per track
- [ ] **Artist timeline** — concatenate speaker vectors by date to visualise an artist's release history

### Medium-term
- [ ] **Full-text transcript search** — client-side index (Lunr or Flexsearch) across all transcripts
- [ ] **Whisper integration** — automatic transcription added to the pipeline alongside diarization
- [ ] **Episode/album grouping** — Tabulator row grouping by series, album, or date
- [ ] **Speaker similarity search UI** — click a speaker pill to find matching speakers across the collection

### Longer-term
- [ ] **AI-powered querying** — natural language questions across the whole collection ("which episodes mention climate change?", "which songs have Adele in the bridge?")
- [ ] **Export & reporting** — filtered views to CSV, per-track or per-speaker summary reports
- [ ] **Web server mode** — lightweight Flask/FastAPI backend with real-time pipeline triggering
- [ ] **Plugin architecture** — drop-in analysers (BPM, sentiment, topic modelling) that add columns

---

## Contributing

Issues, ideas, and pull requests are welcome. The most useful contributions right now are:

- Testing the pipeline against varied audio formats, languages, and speaker counts
- Integrating Whisper transcription into `singer_pipeline.py`
- Speaker name editor UI in the viewer
- Segment-level click-to-play

---

## Licence

MIT