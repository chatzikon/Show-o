#!/usr/bin/env python3
"""
Create ONE combined evaluation split containing exactly one frame per original
video, using all three existing dataset splits: train, validation, and test.

Expected dataset layout
-----------------------
<dataset_root>/
    train/
        images/
        metadata.jsonl
        *.jsonl
    validation/
        images/
        metadata.jsonl
        *.jsonl
    test/
        images/
        metadata.jsonl
        *.jsonl

Output
------
<output_split>/
    images/
    metadata.jsonl
    evaluation_manifest.jsonl
    subset_summary.json
    <merged/filtered auxiliary JSONL files>

For every original video appearing in train/validation/test, the script selects
one ALREADY-EXTRACTED frame. By default it selects the frame with the largest
stored Laplacian-variance "sharpness" value.

The original split is preserved in metadata via "source_split".

The output images can be symlinked, hardlinked, or copied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


IMAGE_KEYS = (
    "image",
    "image_path",
    "img_path",
    "path",
    "file",
    "filename",
    "image_file",
    "image_name",
)

VIDEO_KEYS = (
    "video_relative_path",
    "video",
    "resolved_video_path",
    "source_video",
    "video_path",
)

DEFAULT_SPLITS = ("train", "validation", "test")

# Fallback for filenames such as:
# Abuse043_x264_0000_44273e0e4197_f01.jpg
FRAME_NAME_RE = re.compile(
    r"^(?P<video>.+?)_\d{4}_[0-9a-fA-F]+_f\d+$"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc

            if isinstance(obj, dict):
                rows.append(obj)

    return rows


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> int:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return count


def first_nonempty(
    row: dict[str, Any],
    keys: Iterable[str],
) -> Any:
    for key in keys:
        value = row.get(key)

        if (
            value is not None
            and str(value).strip() != ""
        ):
            return value

    return None


def image_basename(
    row: dict[str, Any],
) -> str | None:
    value = first_nonempty(
        row,
        IMAGE_KEYS,
    )

    if value is None:
        return None

    return Path(str(value)).name


def fallback_video_key_from_image(
    row: dict[str, Any],
) -> str | None:
    name = image_basename(row)

    if not name:
        return None

    stem = Path(name).stem
    match = FRAME_NAME_RE.match(stem)

    if match:
        return match.group("video")

    return None


def video_key(
    row: dict[str, Any],
) -> str:
    """
    Prefer explicit original-video metadata. Filename parsing is only a fallback.
    """
    for key in VIDEO_KEYS:
        value = row.get(key)

        if (
            value is not None
            and str(value).strip() != ""
        ):
            return str(value).replace("\\", "/")

    fallback = fallback_video_key_from_image(row)

    if fallback:
        return fallback

    raise ValueError(
        "Could not determine original video for row with image "
        f"{first_nonempty(row, IMAGE_KEYS)!r}"
    )


def resolve_source_image(
    row: dict[str, Any],
    source_split_dir: Path,
) -> Path:
    value = first_nonempty(
        row,
        IMAGE_KEYS,
    )

    if value is None:
        raise ValueError(
            "Metadata row has no image path."
        )

    raw = Path(str(value))
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)

    else:
        # image = "images/foo.jpg"
        candidates.append(
            source_split_dir / raw
        )

        # image = "validation/images/foo.jpg"
        candidates.append(
            source_split_dir.parent / raw
        )

        # basename fallback
        candidates.append(
            source_split_dir
            / "images"
            / raw.name
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve source image. Tried: "
        + ", ".join(
            str(x)
            for x in candidates
        )
    )


def numeric_value(
    row: dict[str, Any],
    key: str,
    default: float,
) -> float:
    value = row.get(key)

    if value in (None, ""):
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def choose_sharpest(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Select the row with the largest stored Laplacian-variance sharpness.

    Ties are broken deterministically by selected timestamp and image basename.
    """
    return max(
        rows,
        key=lambda r: (
            numeric_value(
                r,
                "sharpness",
                float("-inf"),
            ),
            -numeric_value(
                r,
                "selected_timestamp",
                float("inf"),
            ),
            image_basename(r) or "",
        ),
    )


def choose_middle(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Optional alternative: choose the available extracted frame whose timestamp
    is closest to the middle of the original video.
    """
    durations = [
        numeric_value(
            row,
            "video_duration",
            float("nan"),
        )
        for row in rows
    ]

    durations = [
        x for x in durations
        if x == x and x > 0
    ]

    if durations:
        target = durations[0] / 2.0
    else:
        times = sorted(
            numeric_value(
                row,
                "selected_timestamp",
                0.0,
            )
            for row in rows
        )
        target = times[
            len(times) // 2
        ]

    return min(
        rows,
        key=lambda r: (
            abs(
                numeric_value(
                    r,
                    "selected_timestamp",
                    0.0,
                )
                - target
            ),
            image_basename(r) or "",
        ),
    )


def choose_row(
    rows: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    if len(rows) == 1:
        return rows[0]

    if mode == "sharpest":
        return choose_sharpest(rows)

    if mode == "middle":
        return choose_middle(rows)

    raise ValueError(
        f"Unknown selection mode: {mode}"
    )


def install_image(
    src: Path,
    dst: Path,
    mode: str,
) -> None:
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        dst.exists()
        or dst.is_symlink()
    ):
        dst.unlink()

    if mode == "symlink":
        relative_target = os.path.relpath(
            src,
            start=dst.parent,
        )
        dst.symlink_to(
            relative_target
        )

    elif mode == "hardlink":
        os.link(
            src,
            dst,
        )

    elif mode == "copy":
        shutil.copy2(
            src,
            dst,
        )

    else:
        raise ValueError(
            f"Unknown link mode: {mode}"
        )


def rewrite_image_field(
    row: dict[str, Any],
    new_relative_image: str,
) -> dict[str, Any]:
    out = dict(row)
    found = False

    for key in IMAGE_KEYS:
        if (
            key in out
            and out[key] not in (
                None,
                "",
            )
        ):
            out[key] = new_relative_image
            found = True
            break

    if not found:
        out["image"] = new_relative_image

    return out


def safe_output_basename(
    source_split: str,
    original_basename: str,
    already_used: set[str],
) -> str:
    """
    Preserve the original filename unless a basename collision occurs across
    train/validation/test. If so, prefix the split name.
    """
    if original_basename not in already_used:
        return original_basename

    candidate = (
        f"{source_split}__"
        f"{original_basename}"
    )

    if candidate not in already_used:
        return candidate

    stem = Path(original_basename).stem
    suffix = Path(original_basename).suffix

    index = 2

    while True:
        candidate = (
            f"{source_split}__"
            f"{stem}__{index}"
            f"{suffix}"
        )

        if candidate not in already_used:
            return candidate

        index += 1


def auxiliary_row_matches_selected(
    row: dict[str, Any],
    old_to_new_basename: dict[str, str],
) -> tuple[bool, str | None]:
    old_basename = image_basename(row)

    if (
        old_basename is None
        or old_basename
        not in old_to_new_basename
    ):
        return False, None

    return (
        True,
        old_to_new_basename[
            old_basename
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one combined train+validation+test split with exactly "
            "one already-extracted frame per original video."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        #required=True,
        default='/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA_image_dataset',
        help=(
            "Directory containing train/, validation/, and test/."
        ),
    )

    parser.add_argument(
        "--output-split",
        type=Path,
        #required=True,
        default='/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA_image_dataset/one_frame_per_video_split',
        help=(
            "Destination directory for the new combined one-frame-per-video split."
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        # default=list(
        #     DEFAULT_SPLITS
        # ),
        default=['validation'],
        help=(
            "Source split directory names. Default: train validation test"
        ),
    )

    parser.add_argument(
        "--selection",
        choices=(
            "sharpest",
            "middle",
        ),
        default="sharpest",
        help=(
            "How to select one existing extracted frame per original video. "
            "Default: sharpest."
        ),
    )

    parser.add_argument(
        "--link-mode",
        choices=(
            "symlink",
            "hardlink",
            "copy",
        ),
        default="copy",
        help=(
            "How selected images are placed in the combined images directory."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace an existing non-empty output directory."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_root = (
        args.dataset_root.resolve()
    )

    output_split = (
        args.output_split.resolve()
    )

    source_split_dirs: dict[str, Path] = {}

    for split_name in args.splits:
        split_dir = (
            dataset_root
            / split_name
        )

        metadata_path = (
            split_dir
            / "metadata.jsonl"
        )

        images_dir = (
            split_dir
            / "images"
        )

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing split directory: {split_dir}"
            )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing metadata file: {metadata_path}"
            )

        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Missing images directory: {images_dir}"
            )

        source_split_dirs[
            split_name
        ] = split_dir

    if output_split.exists():
        has_content = any(
            output_split.iterdir()
        )

        if (
            has_content
            and not args.overwrite
        ):
            raise FileExistsError(
                f"Output split is not empty: {output_split}\n"
                "Use --overwrite if you intentionally want to rebuild it."
            )

        if (
            has_content
            and args.overwrite
        ):
            shutil.rmtree(
                output_split
            )

    (
        output_split
        / "images"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # READ ALL THREE SPLITS
    # --------------------------------------------------------

    all_rows: list[
        dict[str, Any]
    ] = []

    source_counts: dict[
        str,
        int,
    ] = {}

    for (
        split_name,
        split_dir,
    ) in source_split_dirs.items():

        rows = read_jsonl(
            split_dir
            / "metadata.jsonl"
        )

        source_counts[
            split_name
        ] = len(rows)

        for row in rows:
            enriched = dict(row)
            enriched[
                "_source_split"
            ] = split_name
            all_rows.append(
                enriched
            )

    if not all_rows:
        raise RuntimeError(
            "No metadata rows were found."
        )

    # --------------------------------------------------------
    # GROUP GLOBALLY BY ORIGINAL VIDEO
    # --------------------------------------------------------

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    splits_per_video: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in all_rows:
        key = video_key(row)

        groups[key].append(row)

        splits_per_video[
            key
        ].add(
            row["_source_split"]
        )

    cross_split_duplicates = {
        key: sorted(split_names)
        for (
            key,
            split_names,
        )
        in splits_per_video.items()
        if len(split_names) > 1
    }

    # --------------------------------------------------------
    # SELECT ONE FRAME PER ORIGINAL VIDEO
    # --------------------------------------------------------

    selected_rows: list[
        dict[str, Any]
    ] = []

    for key in sorted(groups):
        chosen = choose_row(
            groups[key],
            args.selection,
        )

        chosen = dict(chosen)
        chosen[
            "_source_video_key"
        ] = key

        selected_rows.append(
            chosen
        )

    # --------------------------------------------------------
    # MATERIALIZE IMAGES + BUILD PATH MAPS
    # --------------------------------------------------------

    used_output_basenames: set[
        str
    ] = set()

    # Mapping is split-specific because identical basenames could theoretically
    # exist in different source splits.
    split_old_to_new: dict[
        str,
        dict[str, str],
    ] = {
        split_name: {}
        for split_name
        in source_split_dirs
    }

    selected_metadata: list[
        dict[str, Any]
    ] = []

    manifest_rows: list[
        dict[str, Any]
    ] = []

    selected_per_source_split = Counter()

    for row in selected_rows:
        source_split = row[
            "_source_split"
        ]

        source_split_dir = (
            source_split_dirs[
                source_split
            ]
        )

        src = resolve_source_image(
            row,
            source_split_dir,
        )

        original_basename = (
            src.name
        )

        output_basename = (
            safe_output_basename(
                source_split,
                original_basename,
                used_output_basenames,
            )
        )

        used_output_basenames.add(
            output_basename
        )

        split_old_to_new[
            source_split
        ][
            original_basename
        ] = output_basename

        dst = (
            output_split
            / "images"
            / output_basename
        )

        install_image(
            src,
            dst,
            args.link_mode,
        )

        relative_image = (
            f"images/{output_basename}"
        )

        source_video_key = row[
            "_source_video_key"
        ]

        clean_row = {
            key: value
            for (
                key,
                value,
            )
            in row.items()
            if not key.startswith("_")
        }

        clean_row[
            "image"
        ] = relative_image

        clean_row[
            "source_split"
        ] = source_split

        clean_row[
            "source_image"
        ] = str(src)

        clean_row[
            "source_video_key"
        ] = source_video_key

        clean_row[
            "one_per_video_selection"
        ] = args.selection

        selected_metadata.append(
            clean_row
        )

        manifest_rows.append(
            {
                "image":
                    relative_image,

                "image_absolute":
                    str(dst),

                "source_image":
                    str(src),

                "source_split":
                    source_split,

                "video":
                    row.get(
                        "video"
                    ),

                "video_relative_path":
                    row.get(
                        "video_relative_path"
                    ),

                "source_video_key":
                    source_video_key,

                "label":
                    row.get(
                        "label"
                    ),

                "class_label":
                    row.get(
                        "class_label"
                    ),

                "annotation_id":
                    row.get(
                        "annotation_id"
                    ),

                "sharpness":
                    row.get(
                        "sharpness"
                    ),

                "brightness":
                    row.get(
                        "brightness"
                    ),

                "segment_start":
                    row.get(
                        "segment_start"
                    ),

                "segment_end":
                    row.get(
                        "segment_end"
                    ),

                "selected_timestamp":
                    row.get(
                        "selected_timestamp"
                    ),

                "selection":
                    args.selection,
            }
        )

        selected_per_source_split[
            source_split
        ] += 1

    write_jsonl(
        output_split
        / "metadata.jsonl",
        selected_metadata,
    )

    write_jsonl(
        output_split
        / "evaluation_manifest.jsonl",
        manifest_rows,
    )

    # --------------------------------------------------------
    # MERGE/FILTER AUXILIARY JSONL FILES FROM ALL SPLITS
    # --------------------------------------------------------

    # Example:
    # captions_verified_paligemma.jsonl from train/validation/test becomes one
    # combined captions_verified_paligemma.jsonl.
    auxiliary_names: set[
        str
    ] = set()

    for split_dir in (
        source_split_dirs.values()
    ):
        for jsonl_path in split_dir.glob(
            "*.jsonl"
        ):
            if jsonl_path.name in {
                "metadata.jsonl",
                "failures.jsonl",
            }:
                continue

            auxiliary_names.add(
                jsonl_path.name
            )

    auxiliary_summary: list[
        dict[str, Any]
    ] = []

    for filename in sorted(
        auxiliary_names
    ):
        merged_rows: list[
            dict[str, Any]
        ] = []

        source_total = 0

        for (
            split_name,
            split_dir,
        ) in source_split_dirs.items():

            source_path = (
                split_dir
                / filename
            )

            if not source_path.is_file():
                continue

            rows = read_jsonl(
                source_path
            )

            source_total += len(
                rows
            )

            basename_map = (
                split_old_to_new[
                    split_name
                ]
            )

            for row in rows:
                matched, new_basename = (
                    auxiliary_row_matches_selected(
                        row,
                        basename_map,
                    )
                )

                if not matched:
                    continue

                out_row = (
                    rewrite_image_field(
                        row,
                        f"images/{new_basename}",
                    )
                )

                out_row[
                    "source_split"
                ] = split_name

                merged_rows.append(
                    out_row
                )

        output_path = (
            output_split
            / filename
        )

        kept = write_jsonl(
            output_path,
            merged_rows,
        )

        auxiliary_summary.append(
            {
                "file":
                    filename,

                "source_rows":
                    source_total,

                "kept_rows":
                    kept,
            }
        )

    # --------------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------------

    output_video_keys = [
        row[
            "source_video_key"
        ]
        for row
        in manifest_rows
    ]

    if (
        len(output_video_keys)
        != len(
            set(
                output_video_keys
            )
        )
    ):
        raise RuntimeError(
            "Sanity check failed: "
            "the output contains duplicate source videos."
        )

    if (
        len(manifest_rows)
        != len(groups)
    ):
        raise RuntimeError(
            "Sanity check failed: "
            f"selected {len(manifest_rows)} frames for "
            f"{len(groups)} unique videos."
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "dataset_root":
            str(dataset_root),

        "source_splits":
            list(
                source_split_dirs.keys()
            ),

        "output_split":
            str(output_split),

        "selection":
            args.selection,

        "link_mode":
            args.link_mode,

        "source_metadata_rows":
            source_counts,

        "total_source_metadata_rows":
            len(all_rows),

        "unique_source_videos":
            len(groups),

        "selected_frames":
            len(manifest_rows),

        "selected_frames_by_source_split":
            dict(
                selected_per_source_split
            ),

        "videos_appearing_in_multiple_source_splits":
            cross_split_duplicates,

        "auxiliary_jsonl_files":
            auxiliary_summary,
    }

    (
        output_split
        / "subset_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        "=" * 78
    )
    print(
        "COMBINED ONE-FRAME-PER-VIDEO SPLIT CREATED"
    )
    print(
        "=" * 78
    )

    print(
        f"Dataset root          : {dataset_root}"
    )

    print(
        "Source splits         : "
        + ", ".join(
            source_split_dirs.keys()
        )
    )

    print(
        f"Total metadata rows   : {len(all_rows):,}"
    )

    print(
        f"Unique original videos: {len(groups):,}"
    )

    print(
        f"Selected frames       : {len(manifest_rows):,}"
    )

    print(
        f"Selection             : {args.selection}"
    )

    print(
        f"Image mode            : {args.link_mode}"
    )

    print(
        f"Output                : {output_split}"
    )

    print()

    print(
        "Selected frames by original split:"
    )

    for split_name in (
        source_split_dirs.keys()
    ):
        print(
            f"  {split_name:<12}: "
            f"{selected_per_source_split.get(split_name, 0):,}"
        )

    print()

    if cross_split_duplicates:
        print(
            "WARNING: some original videos were found in more than one source split."
        )
        print(
            f"Count: {len(cross_split_duplicates):,}"
        )
        print(
            "They were grouped globally, so only one frame was kept per video."
        )
        print()

    print(
        "Created:"
    )

    print(
        f"  {output_split / 'images'}"
    )

    print(
        f"  {output_split / 'metadata.jsonl'}"
    )

    print(
        f"  {output_split / 'evaluation_manifest.jsonl'}"
    )

    print(
        f"  {output_split / 'subset_summary.json'}"
    )

    for item in auxiliary_summary:
        print(
            f"  {output_split / item['file']} "
            f"[{item['kept_rows']:,}/{item['source_rows']:,} rows kept]"
        )

    print()

    print(
        "Inference image directory:"
    )

    print(
        f"  {output_split / 'images'}"
    )

    paligemma = (
        output_split
        / "captions_verified_paligemma.jsonl"
    )

    if paligemma.exists():
        print(
            "PaliGemma reference file:"
        )
        print(
            f"  {paligemma}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )