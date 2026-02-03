# Music Analyzer CLI

A local-first CLI to analyze audio tracks in a folder for BPM, energy (0–100), danceability (0–100), valence (0–100), loudness, integrated loudness (LUFS), loudness range (LRA), DR value, bitrate, MP3 bitrate status, and key.

## Requirements
- macOS 15+
- `ffmpeg` available in PATH
- Python 3.14+

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage
```bash
.venv/bin/python music_analyzer.py
.venv/bin/python music_analyzer.py --csv
.venv/bin/python music_analyzer.py /path/to/music
.venv/bin/python music_analyzer.py /path/to/music --csv
.venv/bin/python music_analyzer.py "/path/to/track with spaces.flac"
.venv/bin/python music_analyzer.py "/path/to/folder with spaces" --csv
.venv/bin/python music_analyzer.py --fast --progress
.venv/bin/python music_analyzer.py --tsv
.venv/bin/python music_analyzer.py --tsv --no-header
```

## Short command (atracks)
```bash
/Users/matt/music-analyzer/atracks --csv
```
Folder with spaces:
```bash
atracks --csv "/Users/matt/Media/Matts-record-bag"
```
Fast mode with progress:
```bash
/Users/matt/music-analyzer/atracks --fast --progress
```

Terminal output format:
- Default: multi-line blocks, one track per block.
- TSV output: use `--tsv` (header on by default, suppress with `--no-header`).
- DR color bands in block output: red/yellow/green background for DR ≤5, 6–9, ≥10.
- Use `--no-color` to disable ANSI colors, or `--force-color` to force them on.

### Add atracks to PATH (optional)
```bash
chmod +x /Users/matt/music-analyzer/atracks
sudo ln -sf /Users/matt/music-analyzer/atracks /usr/local/bin/atracks
```
Then run:
```bash
atracks --csv
```

## Notes
- `analysis.csv` is written to the scanned folder when `--csv` is used.
