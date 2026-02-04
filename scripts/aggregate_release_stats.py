#!/usr/bin/env python3
import argparse
import csv
import os
import re
from pathlib import Path


def fmt_float(value, places):
    if value is None:
        return ""
    s = f"{value:.{places}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def parse_kv_tokens(tokens):
    out = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def parse_stat_file(path):
    data = {
        "input_file": None,
        "dataset": None,
        "algorithm": None,
        "mode": None,
        "k": None,
        "svdm": None,
        "times": {},
        "per_class": {},  # label -> metrics dict
    }

    times_re = re.compile(
        r"Times(?:\(ms\))?:\s*read=([^,]+),\s*preprocess=([^,]+),\s*classify=([^,]+),\s*write=([^,]+),\s*total=([^\s]+)",
        re.IGNORECASE,
    )
    timing_line_re = re.compile(r"^([A-Za-z]+)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*$")
    csharp_prec_re = re.compile(r"Precyzja_\d+\(([^)]+)\)=([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    csharp_rec_re = re.compile(r"Odzysk_\d+\(([^)]+)\)=([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    csharp_f1_re = re.compile(r"F1_\d+\(([^)]+)\)=([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    csharp_nprec_re = re.compile(r"NPrecyzja_\d+\(([^)]+)\)=([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    csharp_nrec_re = re.compile(r"NOdzysk_\d+\(([^)]+)\)=([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    csharp_nf1_re = re.compile(r"NF1_\d+\(([^)]+)\)=([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)

    in_per_class = False
    in_timings_block = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")

            if in_timings_block:
                stripped = line.strip()
                if not stripped:
                    in_timings_block = False
                    continue
                m = timing_line_re.match(stripped)
                if m:
                    key = m.group(1).lower()
                    val = float(m.group(2))
                    key_map = {
                        "read": "read",
                        "preprocess": "preprocess",
                        "classification": "classify",
                        "write": "write",
                        "total": "total",
                    }
                    if key in key_map:
                        data["times"][key_map[key]] = val
                    continue
                # stop block on unexpected non-empty line
                in_timings_block = False

            if line.lower().startswith("inputfile:"):
                data["input_file"] = line.split(":", 1)[1].strip()
                base = os.path.basename(data["input_file"])
                data["dataset"] = os.path.splitext(base)[0]
                continue
            if line.lower().startswith("algorithm:"):
                data["algorithm"] = line.split(":", 1)[1].strip()
                continue
            if line.lower().startswith("mode:"):
                data["mode"] = line.split(":", 1)[1].strip()
                continue
            if line.lower().startswith("k:"):
                try:
                    data["k"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    data["k"] = None
                continue
            if line.lower().startswith("nominaldistance:"):
                data["svdm"] = line.split(":", 1)[1].strip()
                continue

            m = times_re.match(line)
            if m:
                data["times"] = {
                    "read": float(m.group(1)),
                    "preprocess": float(m.group(2)),
                    "classify": float(m.group(3)),
                    "write": float(m.group(4)),
                    "total": float(m.group(5)),
                }
                continue

            if line.strip().lower().startswith("timings"):
                in_timings_block = True
                continue

            if re.search(r"Per.?ClassMetrics", line, re.IGNORECASE):
                in_per_class = True
                continue

            if in_per_class:
                if re.search(r"BalancedMetrics", line, re.IGNORECASE):
                    in_per_class = False
                    continue
                if not line.strip():
                    continue
                # Example:
                # High Precision=0.5 Recall=1 F1=0.66 | NPrecision=0.75 NRecall=1 NF1=0.85
                text = line.strip()
                label = text.split()[0].rstrip(":")
                pairs = re.findall(r"([A-Za-z0-9]+)=([+-]?\d+(?:\.\d+)?)", text)
                std_kv = {}
                norm_kv = {}
                for k, v in pairs:
                    try:
                        val = float(v)
                    except ValueError:
                        continue
                    if k in {"Precision", "Recall", "F1"}:
                        std_kv[k] = val
                    elif k in {"NPrecision", "NRecall", "NF1"}:
                        norm_kv[k] = val
                data["per_class"][label] = {
                    "Precision": std_kv.get("Precision"),
                    "Recall": std_kv.get("Recall"),
                    "F1": std_kv.get("F1"),
                    "NPrecision": norm_kv.get("NPrecision"),
                    "NRecall": norm_kv.get("NRecall"),
                    "NF1": norm_kv.get("NF1"),
                }

            # C#-style per-class metrics (Polish labels)
            for label, val in csharp_prec_re.findall(line):
                data["per_class"].setdefault(label, {})
                data["per_class"][label]["Precision"] = float(val)
            for label, val in csharp_rec_re.findall(line):
                data["per_class"].setdefault(label, {})
                data["per_class"][label]["Recall"] = float(val)
            for label, val in csharp_f1_re.findall(line):
                data["per_class"].setdefault(label, {})
                data["per_class"][label]["F1"] = float(val)
            for label, val in csharp_nprec_re.findall(line):
                data["per_class"].setdefault(label, {})
                data["per_class"][label]["NPrecision"] = float(val)
            for label, val in csharp_nrec_re.findall(line):
                data["per_class"].setdefault(label, {})
                data["per_class"][label]["NRecall"] = float(val)
            for label, val in csharp_nf1_re.findall(line):
                data["per_class"].setdefault(label, {})
                data["per_class"][label]["NF1"] = float(val)

    if not data["dataset"]:
        # fallback: derive from path
        data["dataset"] = path.name
    return data


def write_csv(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_table(columns, rows, right_align=None):
    if right_align is None:
        right_align = set()
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    lines = []
    header = "  ".join(
        c.rjust(widths[c]) if c in right_align else c.ljust(widths[c]) for c in columns
    )
    lines.append(header)
    lines.append("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        line = "  ".join(
            str(row.get(c, "")).rjust(widths[c]) if c in right_align else str(row.get(c, "")).ljust(widths[c])
            for c in columns
        )
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate STAT_*.txt files into comparison tables."
    )
    parser.add_argument(
        "--root",
        default="build/Release",
        help="Root folder to scan (default: build/Release).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder (default: <root>/_aggregated).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root = Path(args.root)
    if not root.is_absolute():
        root = repo_root / root
    if not root.exists():
        raise SystemExit(f"Root folder not found: {root}")

    stat_files = sorted(
        p for p in root.rglob("STAT_*.txt") if "_aggregated" not in p.parts
    )
    if not stat_files:
        raise SystemExit(f"No STAT_*.txt files found under: {root}")

    datasets = {}
    for path in stat_files:
        data = parse_stat_file(path)
        datasets.setdefault(data["dataset"], []).append(data)

    out_root = Path(args.out) if args.out else (root / "_aggregated")
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    report_lines = []

    for dataset, entries in sorted(datasets.items()):
        entries.sort(key=lambda e: (e["algorithm"], e["mode"], e["k"], e["svdm"]))
        dataset_dir = out_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # TIMINGS
        timings_cols = ["Algorithm", "Mode", "k", "read", "preprocess", "classify", "write", "total"]
        timings_rows = []
        for e in entries:
            t = e["times"]
            timings_rows.append(
                {
                    "Algorithm": e["algorithm"],
                    "Mode": e["mode"],
                    "k": e["k"],
                    "read": fmt_float(t.get("read", 0.0), 4),
                    "preprocess": fmt_float(t.get("preprocess", 0.0), 4),
                    "classify": fmt_float(t.get("classify", 0.0), 4),
                    "write": fmt_float(t.get("write", 0.0), 4),
                    "total": fmt_float(t.get("total", 0.0), 4),
                }
            )
        write_csv(dataset_dir / "TIMINGS.csv", timings_cols, timings_rows)

        # MIARY_STANDARD (CId)
        std_cols = ["Algorithm", "Mode", "k", "CId", "Precision", "Recall", "F1"]
        std_rows = []
        for e in entries:
            for label in sorted(e["per_class"].keys()):
                m = e["per_class"][label]
                std_rows.append(
                    {
                        "Algorithm": e["algorithm"],
                        "Mode": e["mode"],
                        "k": e["k"],
                        "CId": label,
                        "Precision": fmt_float(m.get("Precision", 0.0), 6),
                        "Recall": fmt_float(m.get("Recall", 0.0), 6),
                        "F1": fmt_float(m.get("F1", 0.0), 6),
                    }
                )
        write_csv(dataset_dir / "MIARY_STANDARD.csv", std_cols, std_rows)

        # MIARY_ZNORMALIZOWANE (NCId)
        norm_cols = ["Algorithm", "Mode", "k", "NCId", "NPrecision", "NRecall", "NF1"]
        norm_rows = []
        for e in entries:
            for label in sorted(e["per_class"].keys()):
                m = e["per_class"][label]
                norm_rows.append(
                    {
                        "Algorithm": e["algorithm"],
                        "Mode": e["mode"],
                        "k": e["k"],
                        "NCId": label,
                        "NPrecision": fmt_float(m.get("NPrecision", 0.0), 6),
                        "NRecall": fmt_float(m.get("NRecall", 0.0), 6),
                        "NF1": fmt_float(m.get("NF1", 0.0), 6),
                    }
                )
        write_csv(dataset_dir / "MIARY_ZNORMALIZOWANE.csv", norm_cols, norm_rows)

        # Report text
        report_lines.append(f"DATASET: {dataset}")
        report_lines.append("")
        report_lines.append("TIMINGS(ms)")
        report_lines.append(
            render_table(
                timings_cols,
                timings_rows,
                right_align={"k", "read", "preprocess", "classify", "write", "total"},
            )
        )
        report_lines.append("")
        report_lines.append("MIARY_STANDARD (CId)")
        report_lines.append(
            render_table(
                std_cols,
                std_rows,
                right_align={"k", "Precision", "Recall", "F1"},
            )
        )
        report_lines.append("")
        report_lines.append("MIARY_ZNORMALIZOWANE (NCId)")
        report_lines.append(
            render_table(
                norm_cols,
                norm_rows,
                right_align={"k", "NPrecision", "NRecall", "NF1"},
            )
        )
        report_lines.append("\n" + "=" * 80 + "\n")

    report_path = out_root / "REPORT.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote aggregated tables to: {out_root}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
