"""
Turns output/counts_log.csv into a short, readable summary printed to the
console -- handy for pasting final numbers straight into a project report
instead of reformatting a raw CSV by hand.

Usage:
    python tools/count_report.py output/counts_log.csv
"""

import csv
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "output/counts_log.csv"

    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    _header, body = rows[0], rows[1:]
    print(f"Counting summary ({path})")
    print("-" * 42)
    for class_name, in_count, out_count in body:
        print(f"{class_name:<14} in: {in_count:<6} out: {out_count:<6}")


if __name__ == "__main__":
    main()
