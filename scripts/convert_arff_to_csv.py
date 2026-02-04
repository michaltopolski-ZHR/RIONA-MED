#!/usr/bin/env python3
import argparse
from pathlib import Path


def extract_data_lines(arff_path):
    lines = []
    in_data = False
    with open(arff_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            if not line:
                continue
            if line.startswith("%"):
                continue
            lines.append(line)
    return in_data, lines


def convert_file(arff_path, overwrite=False):
    found, lines = extract_data_lines(arff_path)
    if not found:
        return False, 0, "missing @data"
    if not lines:
        return False, 0, "no data rows"

    out_path = arff_path.with_suffix(".csv")
    if out_path.exists() and not overwrite:
        return False, 0, "csv exists (use --overwrite)"

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, len(lines), str(out_path)


def find_arff_files(roots):
    files = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.arff")))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Convert ARFF files to CSV with only @data rows."
    )
    parser.add_argument(
        "--root",
        action="append",
        help="Root folder to scan (can be used multiple times).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .csv files.",
    )
    args = parser.parse_args()

    roots = [Path(p) for p in (args.root or [])]
    if not roots:
        roots = [Path("data"), Path("build") / "Release"]

    arff_files = find_arff_files(roots)
    if not arff_files:
        raise SystemExit("No .arff files found in the given roots.")

    converted = 0
    skipped = 0
    for path in arff_files:
        ok, rows, info = convert_file(path, overwrite=args.overwrite)
        if ok:
            converted += 1
            print(f"OK  {path} -> {info}  rows={rows}")
        else:
            skipped += 1
            print(f"SKIP {path}  ({info})")

    print(f"Done. Converted={converted}, Skipped={skipped}")


if __name__ == "__main__":
    main()
