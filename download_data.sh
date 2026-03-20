#!/usr/bin/env bash
# =============================================================================
# SoundAtlas — test data download script
# =============================================================================
# Usage:
#   chmod +x download_data.sh
#   ./download_data.sh              # downloads everything
#   ./download_data.sh music        # music sources only
#   ./download_data.sh podcast      # podcast/speech sources only
#   ./download_data.sh quick        # smallest samples only (~500MB total)
#
# What gets downloaded:
#   data/
#   ├── fma_small/          MP3s, 8000 × 30s clips, 8 genres, CC-licensed  (7.2GB)
#   ├── fma_metadata/       Track/artist/genre metadata CSVs                (342MB)
#   ├── musdb18hq/          150 full tracks, isolated WAV stems             (30GB)  *requires access request*
#   ├── librispeech/        Multi-speaker read speech, FLAC, CC BY 4.0
#   │   ├── dev-clean/      40 speakers, 5h clean speech                    (337MB)
#   │   └── dev-other/      40 speakers, 5h challenging speech              (314MB)
#   └── ami/                AMI Meeting Corpus — real multi-speaker meetings (optional)
#
# Notes on MUSDB18-HQ:
#   The dataset requires a one-time access request at https://zenodo.org/records/3338373
#   Accept the terms, then Zenodo gives you a personal download token.
#   Replace YOUR_ZENODO_TOKEN below with that token.
#   Alternatively use `zenodo_get` (pip install zenodo-get) which handles auth:
#     zenodo_get -r 3338373 -o data/musdb18hq
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
MODE="${1:-all}"

mkdir -p "$DATA_DIR"

log() { echo -e "\n\033[1;34m▶ $1\033[0m"; }
ok()  { echo -e "\033[1;32m✓ $1\033[0m"; }
warn(){ echo -e "\033[1;33m⚠ $1\033[0m"; }

# =============================================================================
# ── MUSIC ─────────────────────────────────────────────────────────────────────
# =============================================================================

download_fma_small() {
    log "FMA Small — 8000 × 30s MP3 clips, 8 genres, CC-licensed (7.2GB)"
    mkdir -p "$DATA_DIR/fma"
    cd "$DATA_DIR/fma"

    if [ ! -f "fma_small.zip" ]; then
        wget -c -O fma_small.zip \
            "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    else
        ok "fma_small.zip already downloaded"
    fi

    log "Verifying checksum..."
    echo "ade154f733e7c09656cbe1234aa708b7b73017b3  fma_small.zip" | sha1sum -c - || \
        warn "Checksum mismatch — file may be corrupt, try re-downloading"

    log "Extracting..."
    unzip -q -n fma_small.zip
    ok "FMA Small ready at data/fma/fma_small/"
    cd "$SCRIPT_DIR"
}

download_fma_metadata() {
    log "FMA Metadata — track/artist/genre CSVs for all 106k tracks (342MB)"
    mkdir -p "$DATA_DIR/fma"
    cd "$DATA_DIR/fma"

    if [ ! -f "fma_metadata.zip" ]; then
        wget -c -O fma_metadata.zip \
            "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
    else
        ok "fma_metadata.zip already downloaded"
    fi

    log "Extracting..."
    unzip -q -n fma_metadata.zip
    ok "FMA Metadata ready at data/fma/fma_metadata/"
    cd "$SCRIPT_DIR"
}

download_musdb18hq() {
    log "MUSDB18-HQ — 150 full tracks with isolated WAV stems (30GB)"
    warn "MUSDB18-HQ requires a free access request at:"
    warn "  https://zenodo.org/records/3338373"
    warn "Accept the terms there, then either:"
    warn "  (a) Set ZENODO_TOKEN in your environment and re-run, or"
    warn "  (b) pip install zenodo-get && zenodo_get -r 3338373 -o data/musdb18hq"
    echo ""

    mkdir -p "$DATA_DIR/musdb18hq"

    if [ -n "$ZENODO_TOKEN" ]; then
        log "ZENODO_TOKEN found — attempting download via API..."
        # Zenodo API download with personal token
        wget -c --header="Authorization: Bearer $ZENODO_TOKEN" \
            -O "$DATA_DIR/musdb18hq/musdb18hq.zip" \
            "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"

        log "Extracting..."
        unzip -q -n "$DATA_DIR/musdb18hq/musdb18hq.zip" -d "$DATA_DIR/musdb18hq/"
        ok "MUSDB18-HQ ready at data/musdb18hq/"
    else
        warn "ZENODO_TOKEN not set. To download via command line:"
        echo ""
        echo "  # Option 1: zenodo_get (recommended)"
        echo "  pip install zenodo-get"
        echo "  zenodo_get -r 3338373 -o $DATA_DIR/musdb18hq"
        echo ""
        echo "  # Option 2: wget with personal token"
        echo "  export ZENODO_TOKEN=your_token_here"
        echo "  ./download_data.sh music"
        echo ""
        echo "  # Option 3: 7-second clips via musdb Python library (no token needed)"
        echo "  pip install musdb"
        echo "  python -c \""
        echo "  import musdb"
        echo "  musdb.DB(root='$DATA_DIR/musdb18hq', download=True)"
        echo "  \""
        echo ""
    fi
}

download_musdb18_clips() {
    # Fallback: download just the 7-second test clips via musdb library
    # These don't require a Zenodo token
    log "MUSDB18 — 7-second test clips (no token required, ~50MB)"
    pip install musdb --quiet
    python3 - <<EOF
import musdb, os
db = musdb.DB(root="$DATA_DIR/musdb18_clips", download=True)
print(f"Downloaded {len(db)} clips to $DATA_DIR/musdb18_clips")
EOF
    ok "MUSDB18 clips ready at data/musdb18_clips/"
}

# =============================================================================
# ── PODCAST / SPEECH ──────────────────────────────────────────────────────────
# =============================================================================

download_librispeech_dev() {
    log "LibriSpeech dev-clean — 40 speakers, 5h clean read speech, CC BY 4.0 (337MB)"
    mkdir -p "$DATA_DIR/librispeech"
    cd "$DATA_DIR/librispeech"

    # dev-clean: good starting point — 40 distinct speakers, clean recordings
    if [ ! -f "dev-clean.tar.gz" ]; then
        wget -c "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    else
        ok "dev-clean.tar.gz already downloaded"
    fi

    log "Extracting..."
    tar -xzf dev-clean.tar.gz
    ok "LibriSpeech dev-clean ready at data/librispeech/LibriSpeech/dev-clean/"

    # dev-other: same size but more varied accents/quality — better diarization test
    log "LibriSpeech dev-other — 40 speakers, more challenging speech (314MB)"
    if [ ! -f "dev-other.tar.gz" ]; then
        wget -c "https://www.openslr.org/resources/12/dev-other.tar.gz"
    else
        ok "dev-other.tar.gz already downloaded"
    fi

    log "Extracting..."
    tar -xzf dev-other.tar.gz
    ok "LibriSpeech dev-other ready at data/librispeech/LibriSpeech/dev-other/"
    cd "$SCRIPT_DIR"
}

download_librispeech_mini() {
    # Tiny subset (126MB) — useful for quick pipeline testing
    log "LibriSpeech mini (dev-clean-2) — regression test subset, 126MB"
    mkdir -p "$DATA_DIR/librispeech"
    cd "$DATA_DIR/librispeech"

    if [ ! -f "dev-clean-2.tar.gz" ]; then
        wget -c "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
    else
        ok "dev-clean-2.tar.gz already downloaded"
    fi
    tar -xzf dev-clean-2.tar.gz
    ok "LibriSpeech mini ready at data/librispeech/LibriSpeech/dev-clean-2/"
    cd "$SCRIPT_DIR"
}

download_ami_corpus() {
    log "AMI Meeting Corpus — real multi-speaker meeting recordings, CC BY 4.0"
    warn "AMI is large (~100GB full). Downloading headset mix only (~16GB)."
    warn "This is ideal podcast-like test data: 3-5 speakers per session, natural conversation."
    mkdir -p "$DATA_DIR/ami"
    cd "$DATA_DIR/ami"

    # AMI provides individual speaker headset mics + mixed recordings
    # The Headset Mix (Mix-Headset) is the most useful for diarization testing
    # Download just the first 10 meetings as a sample
    AMI_BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations"

    log "Downloading AMI sample meetings (headset mix)..."
    # These are ~100MB each — 5 meetings as a representative sample
    for meeting in ES2002a ES2002b IS1000a IS1000b EN2001a; do
        if [ ! -f "${meeting}.Mix-Headset.wav" ]; then
            wget -c -q \
                "https://groups.inf.ed.ac.uk/ami/download/temp/amicorpus/${meeting}/audio/${meeting}.Mix-Headset.wav" \
                -O "${meeting}.Mix-Headset.wav" \
                && ok "Downloaded ${meeting}" \
                || warn "Could not download ${meeting} — check AMI corpus access at https://groups.inf.ed.ac.uk/ami/corpus/"
        else
            ok "${meeting} already downloaded"
        fi
    done
    cd "$SCRIPT_DIR"
}

download_common_voice_sample() {
    # Mozilla Common Voice — free, multi-speaker, multi-accent, CC0
    # Use the validated English clip sample (no account needed for clips)
    log "Mozilla Common Voice — sample validated English clips"
    warn "Full Common Voice requires free account at voice.mozilla.org/datasets"
    warn "Showing download command — replace with your actual download URL from the site:"
    echo ""
    echo "  # After downloading from https://commonvoice.mozilla.org/en/datasets"
    echo "  # (select English, download 'Validated' set):"
    echo "  tar -xjf cv-corpus-XX.X-YYYY-MM-DD-en.tar.bz2 -C $DATA_DIR/common_voice/"
    echo ""
    warn "Alternatively, use the Hugging Face datasets library (no manual download):"
    echo ""
    echo "  pip install datasets"
    echo "  python3 -c \""
    echo "  from datasets import load_dataset"
    echo "  ds = load_dataset('mozilla-foundation/common_voice_13_0', 'en',"
    echo "                    split='validation', trust_remote_code=True)"
    echo "  print(ds[0])  # inspect first sample"
    echo "  \""
}

# =============================================================================
# ── POST-DOWNLOAD PREP ────────────────────────────────────────────────────────
# =============================================================================

prep_librispeech_for_pipeline() {
    # LibriSpeech uses FLAC + nested speaker/chapter dirs.
    # The pipeline expects WAV files. This converts and flattens.
    log "Converting LibriSpeech FLAC → WAV and flattening directory structure..."
    FLAC_DIR="$DATA_DIR/librispeech/LibriSpeech"
    OUT_DIR="$DATA_DIR/librispeech_wav"
    mkdir -p "$OUT_DIR"

    if ! command -v ffmpeg &>/dev/null; then
        warn "ffmpeg not found. Install it: sudo apt install ffmpeg  OR  brew install ffmpeg"
        warn "Skipping FLAC → WAV conversion."
        return
    fi

    find "$FLAC_DIR" -name "*.flac" | while read -r f; do
        speaker=$(basename "$(dirname "$(dirname "$f")")")
        chapter=$(basename "$(dirname "$f")")
        out="$OUT_DIR/${speaker}_${chapter}_$(basename "${f%.flac}.wav")"
        [ -f "$out" ] || ffmpeg -i "$f" -ar 16000 -ac 1 -c:a pcm_s16le "$out" -loglevel quiet
    done

    ok "WAV files ready at $OUT_DIR/"
    echo "Run the pipeline with:"
    echo "  python singer_pipeline.py --audio_dir $OUT_DIR --hf_token hf_YOURTOKEN"
}

prep_fma_for_pipeline() {
    # FMA uses MP3. Convert a sample to WAV for the pipeline.
    log "Converting FMA MP3 → WAV (first 50 tracks as sample)..."
    MP3_DIR="$DATA_DIR/fma/fma_small"
    OUT_DIR="$DATA_DIR/fma_wav_sample"
    mkdir -p "$OUT_DIR"

    if ! command -v ffmpeg &>/dev/null; then
        warn "ffmpeg not found. Install: sudo apt install ffmpeg  OR  brew install ffmpeg"
        return
    fi

    count=0
    find "$MP3_DIR" -name "*.mp3" | head -50 | while read -r f; do
        out="$OUT_DIR/$(basename "${f%.mp3}.wav")"
        [ -f "$out" ] || ffmpeg -i "$f" -ar 44100 -ac 2 -c:a pcm_s16le "$out" -loglevel quiet
        count=$((count + 1))
    done

    ok "Sample WAV files ready at $OUT_DIR/"
    echo "Run the pipeline with:"
    echo "  python singer_pipeline.py --audio_dir $OUT_DIR --hf_token hf_YOURTOKEN --min_speakers 1 --max_speakers 2"
}

# =============================================================================
# ── MAIN ──────────────────────────────────────────────────────────────────────
# =============================================================================

echo "=================================================="
echo " SoundAtlas data downloader"
echo " Mode: $MODE"
echo " Target: $DATA_DIR"
echo "=================================================="

case "$MODE" in
    quick)
        log "Quick mode — minimal downloads for pipeline testing (~500MB)"
        download_librispeech_mini
        prep_librispeech_for_pipeline
        ;;
    music)
        download_fma_metadata
        download_fma_small
        download_musdb18hq
        prep_fma_for_pipeline
        ;;
    podcast)
        download_librispeech_dev
        download_ami_corpus
        download_common_voice_sample
        prep_librispeech_for_pipeline
        ;;
    all)
        download_fma_metadata
        download_fma_small
        download_musdb18hq
        download_librispeech_dev
        download_ami_corpus
        prep_librispeech_for_pipeline
        prep_fma_for_pipeline
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [all|music|podcast|quick]"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
ok "Done. Next steps:"
echo ""
echo "  1. Get a free HuggingFace token:"
echo "     https://huggingface.co/settings/tokens"
echo ""
echo "  2. Accept the pyannote model licence:"
echo "     https://huggingface.co/pyannote/speaker-diarization-community-1"
echo ""
echo "  3. One-time model setup:"
echo "     pip install pyannote.audio torch torchaudio huggingface_hub"
echo "     huggingface-cli login"
echo "     huggingface-cli download pyannote/speaker-diarization-community-1 \\"
echo "         --local-dir ./models/pyannote"
echo ""
echo "  4. Run the pipeline (podcast):"
echo "     python singer_pipeline.py \\"
echo "       --audio_dir data/librispeech_wav \\"
echo "       --min_speakers 2 --max_speakers 6"
echo ""
echo "  5. Run the pipeline (music):"
echo "     python singer_pipeline.py \\"
echo "       --audio_dir data/fma_wav_sample \\"
echo "       --min_speakers 1 --max_speakers 3"
echo "=================================================="