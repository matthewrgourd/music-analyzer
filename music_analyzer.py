#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from mutagen import File as MutagenFile
import pyloudnorm as pyln
import requests
from scipy import signal

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a"}

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CSV_FIELDS = [
    "path",
    "title",
    "artist",
    "bpm",
    "energy",
    "danceability",
    "loudness_db",
    "integrated_loudness_lufs",
    "loudness_range_lu",
    "dr_value",
    "dr_peak_db",
    "dr_rms_top20_db",
    "bitrate_kbps",
    "mp3_bitrate_status",
    "valence",
    "key",
]

OUTPUT_FIELDS = CSV_FIELDS


@dataclass
class TrackAnalysis:
    path: str
    title: Optional[str]
    artist: Optional[str]
    bpm: Optional[float]
    energy: Optional[float]
    danceability: Optional[float]
    loudness_db: Optional[float]
    integrated_loudness_lufs: Optional[float]
    loudness_range_lu: Optional[float]
    dr_value: Optional[int]
    dr_peak_db: Optional[float]
    dr_rms_top20_db: Optional[float]
    bitrate_kbps: Optional[int]
    mp3_bitrate_status: str
    valence: Optional[float]
    key: Optional[str]


def run_ffmpeg_decode(path: Path, sr: int = 22050) -> Tuple[np.ndarray, int]:
    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return np.array([], dtype=np.float32), sr
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio, sr


def frame_rms(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(audio) < frame:
        return np.array([], dtype=np.float32)
    frames = 1 + (len(audio) - frame) // hop
    rms = np.empty(frames, dtype=np.float32)
    for i in range(frames):
        start = i * hop
        segment = audio[start : start + frame]
        rms[i] = math.sqrt(np.mean(segment * segment))
    return rms


def estimate_bpm(audio: np.ndarray, sr: int) -> Tuple[Optional[float], float]:
    frame = 1024
    hop = 512
    rms = frame_rms(audio, frame, hop)
    if rms.size == 0:
        return None, 0.0
    onset_env = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    onset_env = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-9)

    corr = signal.correlate(onset_env, onset_env, mode="full")
    corr = corr[corr.size // 2 :]

    min_bpm, max_bpm = 60, 200
    min_lag = int((60.0 * sr) / (max_bpm * hop))
    max_lag = int((60.0 * sr) / (min_bpm * hop))
    if max_lag <= min_lag or max_lag >= len(corr):
        return None, 0.0

    search = corr[min_lag:max_lag]
    lag = np.argmax(search) + min_lag
    peak_strength = float(search[lag - min_lag] / (np.max(search) + 1e-9))

    bpm = 60.0 * sr / (lag * hop)
    return bpm, peak_strength


def estimate_key(audio: np.ndarray, sr: int) -> Optional[str]:
    if audio.size == 0:
        return None
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=2048, noverlap=1536)
    magnitude = np.abs(Zxx)
    chroma = np.zeros(12, dtype=np.float64)

    freqs = f
    for i, freq in enumerate(freqs):
        if freq < 20.0:
            continue
        midi = 69 + 12 * math.log2(freq / 440.0)
        pitch_class = int(round(midi)) % 12
        chroma[pitch_class] += np.mean(magnitude[i])

    if chroma.sum() == 0:
        return None

    chroma = chroma / chroma.sum()
    major_corr = np.correlate(np.roll(MAJOR_PROFILE, 0), chroma, mode="valid")[0]
    minor_corr = np.correlate(np.roll(MINOR_PROFILE, 0), chroma, mode="valid")[0]
    best_major = major_corr
    best_minor = minor_corr
    major_idx = 0
    minor_idx = 0

    for i in range(12):
        maj = np.dot(np.roll(MAJOR_PROFILE, i), chroma)
        minr = np.dot(np.roll(MINOR_PROFILE, i), chroma)
        if maj > best_major:
            best_major = maj
            major_idx = i
        if minr > best_minor:
            best_minor = minr
            minor_idx = i

    if best_major >= best_minor:
        return f"{PITCH_CLASSES[major_idx]} major"
    return f"{PITCH_CLASSES[minor_idx]} minor"


def estimate_valence(energy: float, key: Optional[str], spectral_centroid: float) -> float:
    base = 0.5 * energy + 0.5 * spectral_centroid
    if key and "major" in key:
        base += 0.1
    return float(max(0.0, min(1.0, base)))


def estimate_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    if audio.size == 0:
        return 0.0
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=2048, noverlap=1536)
    magnitude = np.abs(Zxx)
    centroid = np.sum(f[:, None] * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-9)
    centroid = np.mean(centroid)
    return float(min(1.0, centroid / (sr / 2)))


def estimate_energy(rms: np.ndarray) -> float:
    if rms.size == 0:
        return 0.0
    return float(max(0.0, min(1.0, np.mean(rms) * 4.0)))


def estimate_loudness_db(audio: np.ndarray) -> Optional[float]:
    if audio.size == 0:
        return None
    rms = math.sqrt(np.mean(audio * audio))
    if rms <= 0:
        return None
    return float(20.0 * math.log10(rms))


def round_opt(value: Optional[float], ndigits: int) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)


def estimate_loudness_ebu(audio: np.ndarray, sr: int) -> Tuple[Optional[float], Optional[float]]:
    if audio.size == 0:
        return None, None
    try:
        meter = pyln.Meter(sr)
        integrated = float(meter.integrated_loudness(audio))
        lra = float(meter.loudness_range(audio))
        return integrated, lra
    except Exception:
        return None, None


def probe_bitrate_kbps(path: Path) -> Optional[int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=bit_rate",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        data = json.loads(proc.stdout)
    except Exception:
        return None
    bit_rate = data.get("format", {}).get("bit_rate")
    if not bit_rate:
        return None
    try:
        return int(round(int(bit_rate) / 1000))
    except Exception:
        return None


def classify_mp3_bitrate(path: Path, bitrate_kbps: Optional[int], min_kbps: int) -> str:
    if path.suffix.lower() != ".mp3":
        return "na"
    if bitrate_kbps is None:
        return "unknown"
    if bitrate_kbps < min_kbps:
        return "low"
    return "ok"


def rms_db_events(audio: np.ndarray, sr: int, window_seconds: float = 3.0) -> np.ndarray:
    if audio.size == 0 or sr <= 0 or window_seconds <= 0:
        return np.array([], dtype=np.float64)
    window = int(sr * window_seconds)
    if window <= 0:
        return np.array([], dtype=np.float64)
    if audio.size <= window:
        rms = math.sqrt(float(np.mean(audio * audio)))
        return np.array([20.0 * math.log10(rms + 1e-12)], dtype=np.float64)
    target_events = 10_000
    hop = max(1, int((audio.size - window) / max(1, target_events - 1)))
    count = 1 + (audio.size - window) // hop
    rms_vals = np.empty(count, dtype=np.float64)
    idx = 0
    for start in range(0, audio.size - window + 1, hop):
        segment = audio[start : start + window]
        rms = math.sqrt(float(np.mean(segment * segment)))
        rms_vals[idx] = 20.0 * math.log10(rms + 1e-12)
        idx += 1
    return rms_vals


def estimate_dr_value(audio: np.ndarray, sr: int) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if audio.size == 0:
        return None, None, None
    abs_audio = np.abs(audio)
    if abs_audio.size == 0:
        return None, None, None
    peak = float(np.max(abs_audio))
    if peak <= 0:
        return None, None, None
    peak_db = 20.0 * math.log10(peak)
    rms_events = rms_db_events(audio, sr, window_seconds=3.0)
    if rms_events.size == 0:
        return None, round_opt(peak_db, 2), None
    rms_sorted = np.sort(rms_events)[::-1]
    top_n = max(1, int(math.ceil(rms_sorted.size * 0.2)))
    top20_avg = float(np.mean(rms_sorted[:top_n]))
    dr_value = int(round(peak_db - top20_avg))
    return dr_value, round_opt(peak_db, 2), round_opt(top20_avg, 2)


def estimate_danceability(energy: float, bpm: Optional[float], peak_strength: float) -> Optional[float]:
    if bpm is None:
        return None
    tempo_norm = max(0.0, min(1.0, (bpm - 60.0) / 120.0))
    return float(max(0.0, min(1.0, 0.5 * energy + 0.5 * tempo_norm * peak_strength)))


def read_tags(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        tags = MutagenFile(path)
    except Exception:
        return None, None
    if not tags or not tags.tags:
        return None, None

    title = None
    artist = None
    for key in ("TIT2", "title"):
        if key in tags:
            val = tags[key]
            title = str(val[0]) if isinstance(val, list) else str(val)
            break
    for key in ("TPE1", "artist"):
        if key in tags:
            val = tags[key]
            artist = str(val[0]) if isinstance(val, list) else str(val)
            break
    return title, artist


def spotify_token(client_id: str, client_secret: str) -> Optional[str]:
    try:
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception:
        return None


def spotify_lookup(title: Optional[str], artist: Optional[str], token: str) -> Optional[Dict[str, object]]:
    if not title:
        return None
    query = f"track:{title}"
    if artist:
        query += f" artist:{artist}"
    try:
        resp = requests.get(
            "https://api.spotify.com/v1/search",
            params={"q": query, "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("tracks", {}).get("items", [])
        if not items:
            return None
        track = items[0]
        track_id = track.get("id")

        features = requests.get(
            f"https://api.spotify.com/v1/audio-features/{track_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        ).json()
        return {"features": features}
    except Exception:
        return None


def analyze_file(
    path: Path,
    spotify: Optional[Dict[str, str]],
    fast: bool,
    min_mp3_kbps: int,
) -> TrackAnalysis:
    title, artist = read_tags(path)
    audio, sr = run_ffmpeg_decode(path)
    rms = frame_rms(audio, 1024, 512)

    bpm, peak_strength = estimate_bpm(audio, sr)
    energy = estimate_energy(rms)
    loudness_db = estimate_loudness_db(audio)
    if fast:
        integrated_loudness_lufs, loudness_range_lu = None, None
    else:
        integrated_loudness_lufs, loudness_range_lu = estimate_loudness_ebu(audio, sr)
    dr_value, dr_peak_db, dr_rms_top20_db = estimate_dr_value(audio, sr)
    bitrate_kbps = probe_bitrate_kbps(path)
    mp3_bitrate_status = classify_mp3_bitrate(path, bitrate_kbps, min_mp3_kbps)
    spectral_centroid = estimate_spectral_centroid(audio, sr)
    key = estimate_key(audio, sr)
    valence = estimate_valence(energy, key, spectral_centroid)
    danceability = estimate_danceability(energy, bpm, peak_strength)

    if spotify:
        token = spotify_token(spotify["id"], spotify["secret"])
        if token:
            data = spotify_lookup(title, artist, token)
            if data and "features" in data:
                feats = data["features"]
                bpm = feats.get("tempo", bpm)
                energy = feats.get("energy", energy)
                danceability = feats.get("danceability", danceability)
                loudness_db = feats.get("loudness", loudness_db)
                valence = feats.get("valence", valence)
                key_val = feats.get("key")
                mode = feats.get("mode")
                if key_val is not None and mode is not None:
                    key_name = PITCH_CLASSES[int(key_val)]
                    key = f"{key_name} {'major' if mode == 1 else 'minor'}"
    return TrackAnalysis(
        path=str(path),
        title=title,
        artist=artist,
        bpm=round_opt(bpm, 2),
        energy=round_opt(energy, 3),
        danceability=round_opt(danceability, 3),
        loudness_db=round_opt(loudness_db, 2),
        integrated_loudness_lufs=round_opt(integrated_loudness_lufs, 2),
        loudness_range_lu=round_opt(loudness_range_lu, 2),
        dr_value=dr_value,
        dr_peak_db=dr_peak_db,
        dr_rms_top20_db=dr_rms_top20_db,
        bitrate_kbps=bitrate_kbps,
        mp3_bitrate_status=mp3_bitrate_status,
        valence=round_opt(valence, 3),
        key=key,
    )


def iter_audio_files(folder: Path) -> List[Path]:
    files = []
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            files.append(path)
    return sorted(files)


def write_csv(out_path: Path, analyses: List[TrackAnalysis]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_FIELDS)
        for a in analyses:
            writer.writerow(
                [
                    a.path,
                    a.title or "",
                    a.artist or "",
                    a.bpm if a.bpm is not None else "",
                    a.energy if a.energy is not None else "",
                    a.danceability if a.danceability is not None else "",
                    a.loudness_db if a.loudness_db is not None else "",
                    a.integrated_loudness_lufs if a.integrated_loudness_lufs is not None else "",
                    a.loudness_range_lu if a.loudness_range_lu is not None else "",
                    a.dr_value if a.dr_value is not None else "",
                    a.dr_peak_db if a.dr_peak_db is not None else "",
                    a.dr_rms_top20_db if a.dr_rms_top20_db is not None else "",
                    a.bitrate_kbps if a.bitrate_kbps is not None else "",
                    a.mp3_bitrate_status,
                    a.valence if a.valence is not None else "",
                    a.key or "",
                ]
            )


def format_value(value: object) -> str:
    if value is None:
        return "null"
    return str(value)


def dr_color(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    if value <= 5:
        return "\033[41m\033[97m"
    if value <= 9:
        return "\033[43m\033[30m"
    return "\033[42m\033[30m"


def colorize(value: str, color: Optional[str], enabled: bool) -> str:
    if not enabled or not color:
        return value
    return f"{color}{value}\033[0m"


def track_to_pairs(track: TrackAnalysis) -> List[Tuple[str, object]]:
    return [
        ("path", track.path),
        ("title", track.title),
        ("artist", track.artist),
        ("bpm", track.bpm),
        ("energy", track.energy),
        ("danceability", track.danceability),
        ("loudness_db", track.loudness_db),
        ("integrated_loudness_lufs", track.integrated_loudness_lufs),
        ("loudness_range_lu", track.loudness_range_lu),
        ("dr_value", track.dr_value),
        ("dr_peak_db", track.dr_peak_db),
        ("dr_rms_top20_db", track.dr_rms_top20_db),
        ("bitrate_kbps", track.bitrate_kbps),
        ("mp3_bitrate_status", track.mp3_bitrate_status),
        ("valence", track.valence),
        ("key", track.key),
    ]


def print_blocks(analyses: List[TrackAnalysis], color: bool) -> None:
    for idx, track in enumerate(analyses):
        print(track.path)
        for key, value in track_to_pairs(track)[1:]:
            if key == "dr_value":
                colored = colorize(format_value(value), dr_color(track.dr_value), color)
                print(f"  {key}: {colored}")
            else:
                print(f"  {key}: {format_value(value)}")
        if idx < len(analyses) - 1:
            print()


def print_tsv(analyses: List[TrackAnalysis], header: bool) -> None:
    if header:
        print("\t".join(OUTPUT_FIELDS))
    for track in analyses:
        values = [format_value(val) for _, val in track_to_pairs(track)]
        print("\t".join(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze audio tracks in a folder.")
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing audio files (default: .)")
    parser.add_argument("--csv", action="store_true", help="Write analysis.csv to the folder")
    parser.add_argument("--spotify", action="store_true", help="Enrich with Spotify if credentials are set")
    parser.add_argument("--fast", action="store_true", help="Skip LUFS/LRA loudness analysis for speed")
    parser.add_argument("--progress", action="store_true", help="Print progress to stderr")
    parser.add_argument("--tsv", action="store_true", help="Print tab-separated output")
    parser.add_argument("--no-header", action="store_true", help="Suppress TSV header row")
    parser.add_argument("--no-color", action="store_true", help="Disable color output")
    parser.add_argument("--force-color", action="store_true", help="Force color output")
    parser.add_argument(
        "--min-mp3-kbps",
        type=int,
        default=320,
        help="Minimum MP3 bitrate (kbps) before flagging as low (default: 320)",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"Path not found: {folder}", file=sys.stderr)
        return 1

    if folder.is_file():
        paths = [folder]
    elif folder.is_dir():
        paths = iter_audio_files(folder)
    else:
        print(f"Unsupported path: {folder}", file=sys.stderr)
        return 1

    spotify = None
    if args.spotify:
        cid = os.getenv("SPOTIFY_CLIENT_ID")
        secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        if cid and secret:
            spotify = {"id": cid, "secret": secret}
        else:
            print("Spotify credentials not found in env vars. Skipping.", file=sys.stderr)

    analyses = []
    total = len(paths)
    for idx, path in enumerate(paths, start=1):
        if args.progress:
            print(f"[{idx}/{total}] {path.name}", file=sys.stderr)
        analyses.append(analyze_file(path, spotify, args.fast, args.min_mp3_kbps))
    color_enabled = (sys.stdout.isatty() or args.force_color) and not args.no_color
    if args.tsv:
        print_tsv(analyses, not args.no_header)
    else:
        print_blocks(analyses, color_enabled)

    if args.csv:
        out_path = folder / "analysis.csv"
        write_csv(out_path, analyses)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
