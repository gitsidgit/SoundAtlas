# SoundAtlas

> An audio anthology platform for querying, analysing, and exploring spoken and musical audio content at scale.

SoundAtlas is an open-source toolkit that turns a folder of audio files — podcasts, interviews, music, recordings — into a richly navigable, searchable, and queryable library. It combines automated speaker diarization, waveform visualisation, and AI-powered analysis into a single interactive gallery view.

---

## Vision

Most audio content sits in flat file systems, unsearchable and unanalysed. SoundAtlas aims to change that by treating audio as structured data: who speaks when, what they say, how the audio is composed, and how it relates to other content in your collection.

The immediate focus is two parallel use cases that share the same core pipeline:

- **Music** — identify singers, visualise vocal segments, catalogue stems and multi-artist collaborations
- **Podcasts** — diarise hosts and guests, extract transcripts, surface quotes and topics across episodes

The longer-term goal is a general-purpose audio intelligence layer that works across any spoken or musical content.

---

## Current State

The project currently consists of two components built and iterated together:

### 1. `singer_waveform_tabulator.html` — Interactive Gallery Viewer

A single-file browser application built on [Tabulator 6](https://tabulator.info) with Bootstrap 3 theming. It provides a data-grid view of an audio collection with:

- **Waveform column** — minimal SVG waveforms rendered inline, one per track, with per-singer colour-coded segments overlaid
- **Singer/speaker visualisation** — colour bands highlight which speaker is active across the timeline of each track
- **Singer legend** — click any singer to mute/unmute their segments across all visible waveforms
- **Multi-column sort** — sort builder panel with add/remove sort levels, synced bidirectionally with shift+click header sorting
- **Column header filters** — toggleable filter row with text, numeric, and dropdown filters per column
- **Column visibility** — show/hide any column with state persisted to `localStorage`
- **Column reorder & resize** — drag headers left/right to reorder; drag right edge to resize
- **Amplitude zoom** — range slider to exaggerate waveform height for quieter signals
- **Text filter** — live search across filename, genre, and singer names
- **Pagination** — configurable page size (10 / 20 / 50 / 100 rows)
- **Full persistence** — sort order, filters, column widths, column order, and visibility all survive page reload via Tabulator's built-in `localStorage` persistence

Currently running on 120 synthetically generated tracks with procedural waveforms and randomised singer segments. Ready to be fed real data from the pipeline below.

### 2. `singer_pipeline.py` — Speaker Diarization Pipeline

A Python script that processes a folder of WAV files and produces a `gallery.json` file that the viewer can load directly. Built on [pyannote.audio](https://github.com/pyannote/pyannote-audio).

**What it does:**

- Reads WAV files (mono or stereo, any sample rate)
- Runs `pyannote/speaker-diarization-community-1` for speaker detection
- Merges short same-speaker gaps (< 0.5s) to reduce fragmentation
- Downsamples waveform data to 200 points per track for lightweight SVG rendering
- Outputs structured JSON with waveform data, speaker segments, and audio metadata

**Basic usage:**

```bash
pip install pyannote.audio torch torchaudio numpy

python singer_pipeline.py \
  --audio_dir ./my_songs \
  --hf_token hf_YOURTOKEN \
  --min_speakers 2 \
  --max_speakers 4 \
  --output gallery.json
```

A HuggingFace token is required. Accept the model licence at [hf.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) before first use.

---

## Recommended Audio Sources

| Source | Content | Licence | Notes |
|---|---|---|---|
| [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) | 150 full tracks, isolated stems | Academic | Mix vocal stems to create multi-singer files |
| [FMA](https://github.com/mdeff/fma) | 106k tracks across 8 genres | CC | `fma_small` is 7GB, good for bulk testing |
| [ccMixter](https://ccmixter.org) | Vocal collabs and duets | CC | Search "duet" for known multi-singer content |
| Your own recordings | Any WAV | — | Works directly |

---

## Planned Features

### Near-term

- [ ] **Audio playback** — click a row to play the track; playhead syncs with the waveform visualisation
- [ ] **Load from `gallery.json`** — replace synthetic data with a `fetch()` call to load real pipeline output
- [ ] **Speaker name editor** — rename `Singer A / B / C` labels to real artist or guest names inline
- [ ] **Transcript column** — display Whisper-generated transcript snippets per track

### Medium-term

- [ ] **Full-text transcript search** — index transcripts with a lightweight client-side search (Lunr or Flexsearch) so you can search spoken content across the whole collection
- [ ] **Segment-level playback** — click a coloured singer segment to play just that portion
- [ ] **Episode/album grouping** — Tabulator row grouping by series, album, or date
- [ ] **Whisper integration** — add automatic transcription to the pipeline alongside diarization

### Longer-term

- [ ] **AI-powered querying** — natural language questions answered across the whole collection ("which episodes mention climate change?", "which songs have Adele singing in the bridge?")
- [ ] **Export & reporting** — export filtered views to CSV, generate per-track or per-speaker summary reports
- [ ] **Web server mode** — lightweight Flask/FastAPI backend to serve the gallery from a local or self-hosted server with real-time pipeline triggering
- [ ] **Plugin architecture** — drop-in analysers (BPM detection, sentiment, topic modelling) that add columns to the gallery

---

## Architecture

```
audio files (WAV)
      │
      ▼
singer_pipeline.py
  ├── pyannote diarization  →  speaker segments
  ├── waveform downsampling →  200-pt float array
  └── gallery.json
              │
              ▼
singer_waveform_tabulator.html
  ├── Tabulator 6 (grid, sort, filter, pagination)
  ├── Bootstrap 3 (theming)
  └── inline SVG waveforms (zero dependencies)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Gallery viewer | HTML / CSS / vanilla JS |
| Data grid | [Tabulator 6.3](https://tabulator.info) |
| Styling | Bootstrap 3.4 |
| Diarization | [pyannote.audio](https://github.com/pyannote/pyannote-audio) |
| Transcription (planned) | [OpenAI Whisper](https://github.com/openai/whisper) |
| AI querying (planned) | Anthropic Claude API |

---

## Getting Started

**1. Clone and open the viewer with synthetic data:**

```bash
# No build step needed — open directly in a browser
open singer_waveform_tabulator.html
```

**2. Run the pipeline on real audio:**

```bash
pip install pyannote.audio torch torchaudio numpy
python singer_pipeline.py --audio_dir ./audio --hf_token hf_YOURTOKEN
```

**3. Wire the viewer to real data:**

In `singer_waveform_tabulator.html`, replace the synthetic track generation block (the `for` loop that builds `tracks[]`) with:

```javascript
fetch('gallery.json')
  .then(function(r){ return r.json(); })
  .then(function(data){
    tracks = data;
    tracks.forEach(function(t){
      t.waveData = new Float32Array(t.waveData);
    });
    initTable(); // move the Tabulator init into this function
  });
```

---

## Contributing

This project is at an early stage. Issues, ideas, and pull requests are welcome. The most useful contributions right now are:

- Testing the pipeline against different audio formats and speaker counts
- Integrating Whisper transcription into `singer_pipeline.py`
- Building the audio playback layer in the viewer

---

## Licence

MIT