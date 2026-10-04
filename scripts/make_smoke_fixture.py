#!/usr/bin/env python3
"""Build the Whisper-Flamingo smoke fixture data root.

config/audio/audio_smoke_run{1,2}.yaml point at ``<derived>/wf_smoke_v1`` and
read the same layout as a real run:

    muavic/en/{train,valid,test}.{tsv,en}   LRS3 splits
    muavic/en_vc2/train.{tsv,en}            LRS3 + VoxCeleb2 merged train

The fixture keeps only a handful of rows per split so the smoke exercises the
real loader, media resolution and row format without scanning the full corpus.

Media is not copied: ``muavic/en/audio`` and ``muavic/en/video`` are symlinked
to the real tree so that the official MuAViC rows resolve through
utils.resolve_media_path, and VoxCeleb2 rows already carry absolute paths.

Usage:
    python scripts/make_smoke_fixture.py \
        --source-root <derived>/muavic_wf_v1 \
        --dest-root   <derived>/wf_smoke_v1 \
        --rows 16
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def resolve(value: str, data_root: Path):
    """Mirror utils.resolve_media_path; return the resolved path or None."""
    if os.path.isfile(value):
        return value
    candidates = [os.path.join(data_root, value.lstrip("/"))]
    marker = "/muavic/"
    if marker in value:
        suffix = value.split(marker, 1)[1]
        candidates.append(os.path.join(data_root, suffix))
        candidates.append(os.path.join(data_root, "muavic", suffix))
    for c in dict.fromkeys(os.path.normpath(x) for x in candidates):
        if os.path.isfile(c):
            return c
    return None


def slice_split(
    src_dir: Path, dest_dir: Path, split: str, n: int, data_root: Path, balanced: bool = False
) -> int:
    """Write at most ``n`` resolvable rows of ``split``.

    With ``balanced`` the row budget is split between rows resolved inside the
    LRS3 tree and rows resolved elsewhere (VoxCeleb2), so a merged en_vc2 slice
    exercises both media/loader paths instead of only the LRS3 head rows.
    """
    tsv_p, txt_p = src_dir / (split + ".tsv"), src_dir / (split + ".en")
    if not tsv_p.is_file():
        return 0
    tsv = tsv_p.read_text(encoding="utf-8").splitlines()
    txt = txt_p.read_text(encoding="utf-8").splitlines()
    if len(tsv) - 1 != len(txt):
        raise SystemExit("%s: tsv %d vs en %d" % (split, len(tsv) - 1, len(txt)))
    lrs3_root = os.path.join(data_root, "muavic", "en") + os.sep
    lrs3_quota = n - n // 2 if balanced else 0
    other_quota = n // 2 if balanced else n
    lrs3_kept = other_kept = 0
    kept, texts = [], []
    for row, text in zip(tsv[1:], txt):
        f = row.split("\t")
        if len(f) < 5:
            continue
        if balanced and lrs3_kept >= lrs3_quota and "/muavic/en/" in f[2]:
            # An audio path under the LRS3 tree can never satisfy the
            # non-LRS3 quota; skip its stat calls (the manifest is huge).
            continue
        video, audio = resolve(f[1], data_root), resolve(f[2], data_root)
        if not (video and audio):
            continue
        if balanced and ("/muavic/en/" in f[2] or os.path.normpath(audio).startswith(lrs3_root)):
            if lrs3_kept >= lrs3_quota:
                continue
            lrs3_kept += 1
        else:
            if other_kept >= other_quota:
                continue
            other_kept += 1
        kept.append(row)
        texts.append(text)
        if lrs3_kept >= lrs3_quota and other_kept >= other_quota:
            break
    if not kept:
        raise SystemExit("%s: no row with resolvable media (data_root=%s)" % (split, data_root))
    if balanced and not other_kept:
        raise SystemExit(
            "%s: no non-LRS3 row with resolvable media (data_root=%s)" % (split, data_root)
        )
    (dest_dir / (split + ".tsv")).write_text("\n".join(["/"] + kept) + "\n", encoding="utf-8")
    (dest_dir / (split + ".en")).write_text("\n".join(texts) + "\n", encoding="utf-8")
    print("  %s: %d rows" % (split, len(kept)))
    return len(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--dest-root", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--val-rows", type=int, default=4)
    args = ap.parse_args()

    src = args.source_root / "muavic"
    dst = args.dest_root / "muavic"
    data_root = args.dest_root
    (dst / "en").mkdir(parents=True, exist_ok=True)

    for kind in ("audio", "video"):
        link, target = dst / "en" / kind, src / "en" / kind
        if not target.is_dir():
            raise SystemExit("missing source media dir: %s" % target)
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            raise SystemExit("%s exists and is not a symlink; refusing to replace" % link)
        link.symlink_to(target)
        print("  symlinked %s -> %s" % (link, target))

    print("LRS3 splits:")
    for split, rows in (("train", args.rows), ("valid", args.val_rows), ("test", args.val_rows)):
        tsv_path = src / "en" / (split + ".tsv")
        if not slice_split(src / "en", dst / "en", split, rows, data_root):
            raise SystemExit("missing or empty source split: %s" % tsv_path)

    print("merged train:")
    (dst / "en_vc2").mkdir(parents=True, exist_ok=True)
    tsv_path = src / "en_vc2" / "train.tsv"
    if not slice_split(
        src / "en_vc2", dst / "en_vc2", "train", args.rows, data_root, balanced=True
    ):
        raise SystemExit("missing or empty source split: %s" % tsv_path)

    print("fixture ready at %s" % args.dest_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
