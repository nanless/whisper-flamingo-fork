#!/usr/bin/env python3
"""Validate MuAViC/Whisper-Flamingo TSV and label alignment."""

import argparse
import csv
import json
import subprocess
from pathlib import Path


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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


def validate(tsv_path: Path, label_path: Path, probe_count: int) -> dict:
    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    root = Path(lines[0] or "/")
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
        "probes": [],
    }
    for index, row in enumerate(rows):
        if len(row) < 5:
            report["bad_columns"] += 1
            continue
        video = resolve_path(root, row[1])
        audio = resolve_path(root, row[2])
        if not video.is_file():
            report["missing_video"] += 1
        if not audio.is_file():
            report["missing_audio"] += 1
        if index >= len(labels) or not labels[index].strip():
            report["empty_labels"] += 1
        try:
            int(row[-2])
            int(row[-1])
        except ValueError:
            report["bad_numeric"] += 1
        if index < probe_count:
            report["probes"].append(
                {"row": index, "video": probe(video), "audio": probe(audio)}
            )
    report["row_label_mismatch"] = len(rows) != len(labels)
    report["ok"] = not any(
        report[key]
        for key in (
            "bad_columns",
            "missing_video",
            "missing_audio",
            "empty_labels",
            "bad_numeric",
            "row_label_mismatch",
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--probe-count", type=int, default=10)
    args = parser.parse_args()
    report = validate(args.tsv, args.labels, args.probe_count)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
