#!/usr/bin/env python
"""Decode videos to the exact 25 fps frames the YOLO converter would use.

Reuses open_video / count_frames / wanted_frames from 01_autoencoder/video_to_keypoints.py,
so the saved frames are the ones `video_to_keypoints.py` feeds to YOLO (same rotation
handling, same nearest-frame mapping). Both detectors can then run on identical pixels.

Writes, per clip:  <out>/<clip stem>/<tick:06d>.jpg  and  <out>/<clip stem>/manifest.json

Usage
    python 03_app/tools/decode_frames.py "D:/path/to/Videos" --out <frames dir>

The output contains frames of real people: keep it out of Git (use a scratch folder).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "01_autoencoder"))
import video_to_keypoints as v2k  # noqa: E402


def decode(path: Path, out_dir: Path, quality: int) -> dict:
    n_src, src_fps = v2k.count_frames(path)
    needed = v2k.wanted_frames(n_src, src_fps)
    clip_dir = out_dir / path.stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    by_src = {}
    for tick, i in enumerate(needed):
        by_src.setdefault(int(i), []).append(tick)

    cap = v2k.open_video(path)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    width = height = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in by_src:
            height, width = frame.shape[:2]
            for tick in by_src[i]:
                cv2.imwrite(str(clip_dir / f"{tick:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        i += 1
    cap.release()
    manifest = {
        "clip": path.stem,
        "source_file": path.name,
        "source_frames": n_src,
        "source_fps": src_fps,
        "codec": "".join(chr((fourcc >> 8 * k) & 0xFF) for k in range(4)),
        "width": width,
        "height": height,
        "ticks": len(needed),
        "source_index_per_tick": [int(v) for v in needed],
        "jpeg_quality": quality,
    }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{path.name}: {n_src} src frames @ {src_fps:.3f} fps, {width}x{height}, "
          f"codec {manifest['codec']} -> {len(needed)} ticks")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="video file or folder")
    ap.add_argument("--out", required=True, help="output folder for frames (outside Git)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    args = ap.parse_args(argv)
    src = Path(args.input)
    videos = sorted(p for p in (src.iterdir() if src.is_dir() else [src]) if p.suffix.lower() in v2k.VIDEO_EXT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [decode(v, out, args.quality) for v in videos]
    (out / "clips.json").write_text(json.dumps([c["clip"] for c in clips], indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
