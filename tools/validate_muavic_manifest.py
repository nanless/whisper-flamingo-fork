#!/usr/bin/env python3
"""Validate MuAViC/Whisper-Flamingo TSV and label alignment."""

import argparse
import csv
import json
import subprocess
from fractions import Fraction
from pathlib import Path


def resolve_path(data_root: Path, manifest_root: str, value: str) -> Path:
    path = Path(value)
    candidates = []
    if path.is_file():
        return path
    if manifest_root:
        header = Path(manifest_root)
        candidates.append(header / value.lstrip("/") if header.is_absolute()
                          else data_root / header / value)
    candidates.append(data_root / value.lstrip("/"))
    if "/muavic/" in value:
        suffix = value.split("/muavic/", 1)[1]
        candidates.extend((data_root / suffix, data_root / "muavic" / suffix))
    deduplicated = list(dict.fromkeys(candidate.resolve(strict=False) for candidate in candidates))
    for candidate in deduplicated:
        if candidate.is_file():
            return candidate
    return deduplicated[0]


def probe(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"error": str(error)}


def validate(tsv_path: Path, label_path: Path, data_root: Path, probe_count: int) -> dict:
    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {"tsv": str(tsv_path), "labels": str(label_path), "rows": 0,
                "label_rows": 0, "empty_manifest": True, "ok": False}
    manifest_root = lines[0] or "/"
    rows = list(csv.reader(lines[1:], delimiter="\t"))
    labels = label_path.read_text(encoding="utf-8").splitlines()
    report = {
        "tsv": str(tsv_path),
        "labels": str(label_path),
        "rows": len(rows),
        "label_rows": len(labels),
        "bad_columns": 0,
        "missing_video": 0,
        "missing_audio": 0,
        "empty_labels": 0,
        "bad_numeric": 0,
        "nonpositive_counts": 0,
        "duplicate_ids": 0,
        "probe_errors": 0,
        "probes": [],
    }
    seen_ids = set()
    for index, row in enumerate(rows):
        if len(row) < 5:
            report["bad_columns"] += 1
            continue
        sample_id = row[0]
        if sample_id in seen_ids:
            report["duplicate_ids"] += 1
        seen_ids.add(sample_id)
        video = resolve_path(data_root, manifest_root, row[1])
        audio = resolve_path(data_root, manifest_root, row[2])
        if not video.is_file():
            report["missing_video"] += 1
        if not audio.is_file():
            report["missing_audio"] += 1
        if index >= len(labels) or not labels[index].strip():
            report["empty_labels"] += 1
        try:
            video_frames = int(row[-2])
            audio_samples = int(row[-1])
            if video_frames <= 0 or audio_samples <= 0:
                report["nonpositive_counts"] += 1
        except ValueError:
            report["bad_numeric"] += 1
        if index < probe_count:
            video_probe = probe(video)
            audio_probe = probe(audio)
            errors = []
            if "error" in video_probe:
                errors.append("video_ffprobe")
            if "error" in audio_probe:
                errors.append("audio_ffprobe")
            video_streams = [s for s in video_probe.get("streams", []) if s.get("codec_type") == "video"]
            audio_streams = [s for s in audio_probe.get("streams", []) if s.get("codec_type") == "audio"]
            if not video_streams:
                errors.append("no_video_stream")
            else:
                stream = video_streams[0]
                try:
                    fps = float(Fraction(stream.get("r_frame_rate", "0/1")))
                except (ValueError, ZeroDivisionError):
                    fps = 0
                if abs(fps - 25.0) > 0.01:
                    errors.append("video_fps_not_25")
                if int(stream.get("width", 0)) < 88 or int(stream.get("height", 0)) < 88:
                    errors.append("video_smaller_than_88")
            if not audio_streams:
                errors.append("no_audio_stream")
            else:
                stream = audio_streams[0]
                if int(stream.get("sample_rate", 0)) != 16000:
                    errors.append("audio_sample_rate_not_16000")
                if int(stream.get("channels", 0)) != 1:
                    errors.append("audio_not_mono")
            if errors:
                report["probe_errors"] += 1
            report["probes"].append({"row": index, "video": str(video), "audio": str(audio),
                                     "errors": errors, "video_probe": video_probe,
                                     "audio_probe": audio_probe})
    report["row_label_mismatch"] = len(rows) != len(labels)
    report["ok"] = not any(
        report[key]
        for key in (
            "bad_columns",
            "missing_video",
            "missing_audio",
            "empty_labels",
            "bad_numeric",
            "nonpositive_counts",
            "duplicate_ids",
            "probe_errors",
            "row_label_mismatch",
        )
    ) and bool(rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--probe-count", type=int, default=10)
    args = parser.parse_args()
    report = validate(args.tsv, args.labels, args.data_root, args.probe_count)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
