#!/usr/bin/env python3
"""Generate the RQ1 median/IQR and Scott-Knott ESD summary from raw CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import sys
from pathlib import Path


RAW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = RAW_ROOT / "rq1/accuracy"
DEFAULT_OUTPUT = RAW_ROOT / "rq1/stats/all_systems_result.csv"
MANIFEST = RAW_ROOT / "MANIFEST.csv"

SYSTEM_ORDER = [
    "sac",
    "x264",
    "storm",
    "spear",
    "sqlite",
    "nginx",
    "exastencils",
    "deeparch",
    "imagemagick",
    "libvips",
    "pulsar",
    "rabbitmq",
]

METHODS = [
    "dhda_reuse_ALL_model",
    "SeMLP",
    "BEETLE",
    "ARF",
    "SRP",
    "DaL",
    "select_regressor",
]


def quantile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated quantile (R type 7 / NumPy default)."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def median_iqr(values: list[float]) -> tuple[float, float]:
    median = statistics.median(values)
    iqr = quantile(values, 0.75) - quantile(values, 0.25)
    return median, iqr


def cohen_d(first: list[float], second: list[float]) -> float:
    """Return independent-sample Cohen's d with pooled sample variance."""
    first_variance = statistics.variance(first)
    second_variance = statistics.variance(second)
    denominator = len(first) + len(second) - 2
    pooled = math.sqrt(
        ((len(first) - 1) * first_variance + (len(second) - 1) * second_variance)
        / denominator
    )
    if pooled == 0.0:
        return 0.0 if statistics.mean(first) == statistics.mean(second) else math.inf
    return (statistics.mean(first) - statistics.mean(second)) / pooled


def scott_knott_esd_ranks(columns: dict[str, list[float]]) -> dict[str, int]:
    """Reproduce the parametric ScottKnottESD 2.x grouping used by the study."""
    ordered_methods = sorted(
        columns,
        key=lambda method: statistics.mean(columns[method]),
        reverse=True,
    )
    groups: list[list[str]] = []

    def partition(methods: list[str]) -> None:
        if len(methods) == 1:
            groups.append(methods)
            return

        # ScottKnottESD treats |d| < 0.2 as a negligible endpoint difference.
        endpoint_effect = abs(cohen_d(columns[methods[0]], columns[methods[-1]]))
        if endpoint_effect < 0.2:
            groups.append(methods)
            return

        means = [statistics.mean(columns[method]) for method in methods]
        total = sum(means)
        best_score = -math.inf
        best_split = 1
        for split in range(1, len(methods)):
            left_sum = sum(means[:split])
            right_sum = total - left_sum
            score = (
                left_sum**2 / split
                + right_sum**2 / (len(methods) - split)
                - total**2 / len(methods)
            )
            if score > best_score:
                best_score = score
                best_split = split

        if best_split == 1:
            groups.append(methods[:1])
            partition(methods[1:])
        else:
            partition(methods[:best_split])
            partition(methods[best_split:])

    partition(ordered_methods)
    group_count = len(groups)
    return {
        method: group_count - group_index
        for group_index, group in enumerate(groups)
        for method in group
    }


def read_system(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30:
        raise ValueError(f"Expected 30 runs in {path}; found {len(rows)}")

    result: dict[str, list[float]] = {}
    for method in METHODS:
        values = [float(row[method]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite {method} value in {path}")
        result[method] = values
    return result


def format_number(value: float) -> str:
    return str(round(value, 4))


def build_rows(input_dir: Path) -> list[dict[str, str]]:
    expected_files = {f"{system}_result.csv" for system in SYSTEM_ORDER}
    actual_files = {path.name for path in input_dir.glob("*_result.csv")}
    if actual_files != expected_files:
        raise ValueError(
            "RQ1 input files differ from the 12 paper systems: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )

    output_rows = []
    for system in SYSTEM_ORDER:
        columns = read_system(input_dir / f"{system}_result.csv")
        ranks = scott_knott_esd_ranks(columns)
        row = {"system": system}
        for method in METHODS:
            median, iqr = median_iqr(columns[method])
            row[method] = (
                f"{ranks[method]}_{format_number(median)}_({format_number(iqr)})"
            )
        output_rows.append(row)
    return output_rows


def render_csv(rows: list[dict[str, str]]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["system", *METHODS],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def update_manifest(output: Path) -> None:
    if output.resolve() != DEFAULT_OUTPUT.resolve() or not MANIFEST.exists():
        return
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    manifest_row = {
        "rq": "RQ1",
        "data_kind": "statistics",
        "system": "all",
        "file": output.relative_to(RAW_ROOT).as_posix(),
        "rows": str(len(SYSTEM_ORDER)),
        "columns": " | ".join(["system", *METHODS]),
        "sha256": sha256(output),
        "legacy_source": "generated from rq1/accuracy/*.csv",
    }
    for index, row in enumerate(rows):
        if row["file"] == manifest_row["file"]:
            rows[index] = manifest_row
            break
    else:
        insert_at = next(
            (
                index
                for index, row in enumerate(rows)
                if row["rq"] == "RQ1" and row["data_kind"] == "runtime"
            ),
            len(rows),
        )
        rows.insert(insert_at, manifest_row)

    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the existing output equals a fresh calculation",
    )
    args = parser.parse_args()

    content = render_csv(build_rows(args.input_dir))
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != content:
            print(f"FAIL generated statistics differ from {args.output}")
            return 1
        print("RQ1 derived statistics: 12/12 systems reproduced")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    update_manifest(args.output)
    print(f"Created {args.output} from {args.input_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
